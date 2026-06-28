"""Torch-native static pose optimizer for the Unitree G1 (29-DoF) humanoid.

Uses pre-built .pt kinematics artifacts (built by build_kinematics_artifacts_optim.py)
for FK, Jacobians, rotation matrices, gravity Jacobian, and CoM position.
All differentiable computations are pure PyTorch — no Pinocchio in the hot path.

Execution modes (set MODE below or pass --mode CLI arg):
  single_verification  : solves for the user-defined scenario from pose_optim_config.py,
                         then visualizes the result with meshcat (requires pinocchio).
  batch_execution      : reads feasible_filtered_list.hdf5, samples NUM_PTS arm configs
                         and NUM_FORCE_PER_POINT force/pose combos each, solves all in
                         batches, writes surviving solutions to optim_pose_data.hdf5.

Frame convention (same as pose_optim.py):
  Base frame  – robot pelvis; origin at [0,0,h] in world, rotated by R(roll,pitch,0).
  YAG frame   – yaw-aligned ground frame; origin at [0,0,0], z-up, yaw=0.
  FK/Jac/Grav – all expressed in the BASE frame (kinematics library convention).
  To world    – p_world = R @ p_base + [0,0,h].
  Forces      – F_LEFT / F_RIGHT are horizontal world-frame forces on the robot hands.
  Torque      – tau_j = tG_base(q) - sum_c J_c_base^T F_c_base  (joint DOFs only, 23-dim).
  Equilibrium – force + moment balance in world frame (replaces FreeFlyer tau[:6]==0).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

torch.set_default_dtype(torch.float64)

# ── config ───────────────────────────────────────────────────────────────────
from pose_optim_config import (
    URDF_PATH, ARTIFACTS_DIR,
    POS_LEFT_ARM, POS_RIGHT_ARM, H_USER, RP_USER, F_LEFT, F_RIGHT,
    W1, W2, W3, W4, W5, W6,
    LAM_EQ, LAM_FOOTZ, LAM_ARM, LAM_BOX, LAM_QLIM, LAM_TLIM,
    LAM_RP, LAM_H, LAM_FRIC, LAM_ANKLE, LAM_ARM_ORI, LAM_HEAD, LAM_SYM,
    H0, Z_FEET_0, H_MIN, H_MAX, DEG,
    RPY_LOWER, RPY_UPPER, MU, XL, XU, YL, YU,
    HAND_RPY_LO_R, HAND_RPY_HI_R,
    OPT, LR, STEPS, PRINT_EVERY,
    VISUALIZE, FORCE_SCALE, TARGET_RADIUS,
    FEASIBLE_LIST_PATH, OUTPUT_H5_PATH,
    OPT_BATCH, LR_BATCH, STEPS_BATCH,
    NUM_PTS, NUM_FORCE_PER_POINT, BATCH_SIZE,
    FORCE_RANGE_X, FORCE_RANGE_Y,
    H_RANGE, RP_RANGE,
    MAX_PENALTY_DISCARD,
)

# ── execution mode ────────────────────────────────────────────────────────────
MODE = "single_verification"   # "single_verification" | "batch_execution"

SCRIPT_DIR = Path(__file__).resolve().parent

# ============================================================
# Load robot constants from URDF (non-differentiable, once at startup)
# ============================================================
def _load_robot_constants(urdf_path: str):
    import pinocchio as pin
    model = pin.buildModelFromUrdf(str(urdf_path), pin.JointModelFreeFlyer())
    NV  = model.nv       # 6 base + 23 joints = 29
    NQJ = NV - 6         # 23 actuated joints
    total_mass = float(sum(model.inertias[i].mass for i in range(1, model.njoints)))
    gravity_val = float(abs(model.gravity.linear[2]))

    q_lower = torch.as_tensor(model.lowerPositionLimit[7:])   # (23,)
    q_upper = torch.as_tensor(model.upperPositionLimit[7:])   # (23,)
    tau_limit = torch.as_tensor(model.effortLimit[6:6 + NQJ]) # (23,)

    # Tighten extra joint bounds (same as pose_optim.py).
    joint_names = [model.names[i] for i in range(2, model.njoints)]
    def _idx(n): return joint_names.index(n)
    for name, lo, hi in (
        ("left_hip_yaw_joint",        -60.0*DEG,  60.0*DEG),
        ("right_hip_yaw_joint",       -60.0*DEG,  60.0*DEG),
        ("left_elbow_joint",           None,       90.0*DEG),
        ("right_elbow_joint",          None,       90.0*DEG),
        ("left_shoulder_roll_joint",  -10.0*DEG,   None),
        ("right_shoulder_roll_joint",  None,       10.0*DEG),
    ):
        i = _idx(name)
        if lo is not None:
            q_lower[i] = torch.maximum(q_lower[i], torch.tensor(lo))
        if hi is not None:
            q_upper[i] = torch.minimum(q_upper[i], torch.tensor(hi))

    hip_yaw_idx = torch.tensor([_idx("left_hip_yaw_joint"),
                                 _idx("right_hip_yaw_joint")], dtype=torch.long)
    l_sroll_idx = _idx("left_shoulder_roll_joint")
    r_sroll_idx = _idx("right_shoulder_roll_joint")

    return {
        "NQJ": NQJ, "total_mass": total_mass, "gravity": gravity_val,
        "q_lower": q_lower, "q_upper": q_upper, "tau_limit": tau_limit,
        "hip_yaw_idx": hip_yaw_idx,
        "l_sroll_idx": l_sroll_idx, "r_sroll_idx": r_sroll_idx,
    }


_ROBOT = _load_robot_constants(str(URDF_PATH))
NQJ        = _ROBOT["NQJ"]
TOTAL_MASS = _ROBOT["total_mass"]
GRAVITY    = _ROBOT["gravity"]
Q_LOWER    = _ROBOT["q_lower"]
Q_UPPER    = _ROBOT["q_upper"]
TAU_LIMIT  = _ROBOT["tau_limit"]
HIP_YAW_IDX   = _ROBOT["hip_yaw_idx"]
L_SROLL_IDX   = _ROBOT["l_sroll_idx"]
R_SROLL_IDX   = _ROBOT["r_sroll_idx"]
print(f"[INFO] Total robot mass: {TOTAL_MASS:.3f} kg")

# ============================================================
# Load kinematics artifacts
# ============================================================
def _load_kinematics(artifacts_dir: str | Path, device: str):
    artifacts_dir = str(artifacts_dir)
    kin_path = os.path.join(
        os.path.dirname(__file__),
        "..", "..", "..", "isaac_humanoid_suite",
        "source", "isaac_humanoid_suite", "isaac_humanoid_suite",
        "tasks", "manager_based", "humanoid_pushing",
        "controllers", "ik_controllers", "kinematics.py",
    )
    # Resolve relative to SCRIPT_DIR for robustness.
    kin_abs = (SCRIPT_DIR / "../../../isaac_humanoid_suite/source/isaac_humanoid_suite"
               "/isaac_humanoid_suite/tasks/manager_based/humanoid_pushing"
               "/controllers/ik_controllers/kinematics.py").resolve()
    spec = importlib.util.spec_from_file_location("ik_kinematics_optim", str(kin_abs))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    RobotKinematicsCfg = mod.RobotKinematicsCfg
    RobotKinematics = mod.RobotKinematics
    cfg = RobotKinematicsCfg(
        urdf_path=str(URDF_PATH),
        ee_names=[], ee_parent_joint_names=[], b_P_EE=[],
        q_measured=[], output_groups={},
        artifacts_dir=artifacts_dir,
    )
    return RobotKinematics(cfg, device=device)


_KIN: "RobotKinematics | None" = None   # lazy-loaded


def _get_kin(device: str = "cpu"):
    global _KIN
    if _KIN is None:
        print(f"[INFO] Loading kinematics artifacts from {ARTIFACTS_DIR}")
        _KIN = _load_kinematics(ARTIFACTS_DIR, device)
    return _KIN

# ── q23 index slices (fixed-base model, verified from URDF) ──────────────────
# Left leg:  q23[0:6],   Right leg:  q23[6:12]
# Waist:     q23[12:15], Left arm:   q23[15:19], Right arm: q23[19:23]
_Q_FEET = slice(0, 12)
_Q_ARMS = slice(12, 23)
_Q_HEAD = slice(12, 15)
_Q_ANKLE_L = slice(0, 6)
_Q_ANKLE_R = slice(6, 12)
_Q_HAND_L_IDX = [12, 13, 14, 15, 16, 17, 18]   # waist + left arm
_Q_HAND_R_IDX = [12, 13, 14, 19, 20, 21, 22]   # waist + right arm

# ============================================================
# Frame math utilities
# ============================================================
def rp_to_matrix(rp: torch.Tensor) -> torch.Tensor:
    """Roll-pitch (yaw=0) to rotation matrix.

    Input: rp  (B, 2) or (2,)
    Output: R  (B, 3, 3) or (3, 3)
    """
    squeeze = rp.dim() == 1
    if squeeze:
        rp = rp.unsqueeze(0)
    B = rp.shape[0]
    r, p = rp[:, 0], rp[:, 1]
    cr, sr = torch.cos(r), torch.sin(r)
    cp, sp = torch.cos(p), torch.sin(p)
    zeros = torch.zeros(B, device=rp.device, dtype=rp.dtype)
    # R = Ry(p) @ Rx(r),  yaw=0 → Rz = I
    # [[cp,   sp*sr,  sp*cr],
    #  [0,    cr,    -sr   ],
    #  [-sp,  cp*sr,  cp*cr]]
    R = torch.stack([
        cp, sp*sr, sp*cr,
        zeros, cr, -sr,
        -sp, cp*sr, cp*cr,
    ], dim=1).reshape(B, 3, 3)
    return R.squeeze(0) if squeeze else R


def rot_mat_to_rpy(R: torch.Tensor) -> torch.Tensor:
    """Extrinsic XYZ RPY from rotation matrix.

    Input/output: (B, 3, 3) -> (B, 3)  or  (3,3) -> (3,)
    """
    squeeze = R.dim() == 2
    if squeeze:
        R = R.unsqueeze(0)
    roll  = torch.atan2( R[:, 2, 1],  R[:, 2, 2])
    pitch = torch.atan2(-R[:, 2, 0],  torch.sqrt(R[:, 2, 1]**2 + R[:, 2, 2]**2))
    yaw   = torch.atan2( R[:, 1, 0],  R[:, 0, 0])
    out = torch.stack([roll, pitch, yaw], dim=-1)
    return out.squeeze(0) if squeeze else out


def cross3(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Batched cross product: (..., 3) x (..., 3) -> (..., 3)."""
    return torch.linalg.cross(a, b, dim=-1)


# ============================================================
# Normalisation helpers (match pose_optim.py conventions)
# ============================================================
_S_H  = 0.10
_S_RP = 0.30
_S_F  = torch.tensor([100.0, 100.0, TOTAL_MASS * GRAVITY / 2.0])


def vars_from_normalized(h_n, rp_n, q23, F_n):
    """Map normalised optimiser variables to physical units (supports batches)."""
    h = H0 + _S_H * h_n
    rp = _S_RP * rp_n
    foot_forces = F_n * _S_F.to(F_n.device)
    return h, rp, q23, foot_forces


# ============================================================
# Core computation (batched)
# ============================================================
COST_KEYS = ("tau", "F_feet", "height", "upright", "hip_yaw", "shoulder_roll")
PENALTY_KEYS = (
    "equilibrium", "arm", "arm_ori", "head_behind", "foot_z", "ankle_z",
    "symmetry", "box", "qlim", "tlim", "rp", "h", "friction",
)


def compute_terms(
    h: torch.Tensor,           # (B,) or scalar
    rp: torch.Tensor,          # (B, 2) or (2,)
    q23: torch.Tensor,         # (B, 23) or (23,)
    foot_forces: torch.Tensor, # (B, 2, 3) or (2, 3)  — world frame
    tgt_l: torch.Tensor,       # (B, 3) or (3,)  — left hand target, YAG frame
    tgt_r: torch.Tensor,       # (B, 3) or (3,)  — right hand target, YAG frame
    F_left_world: torch.Tensor,  # (B, 3) or (3,)  — hand forces, world frame
    F_right_world: torch.Tensor, # (B, 3) or (3,)
    kin,
    device: str = "cpu",
) -> tuple:
    """Return (total, cost, penalty, cost_terms, penalty_terms) as scalar tensors.

    All tensors may be batched (leading B dimension) or unbatched (B=1 squeezed).
    Internally everything is promoted to B=1 so downstream code is uniform.
    """
    squeeze = q23.dim() == 1
    if squeeze:
        h = h.unsqueeze(0)
        rp = rp.unsqueeze(0)
        q23 = q23.unsqueeze(0)
        foot_forces = foot_forces.unsqueeze(0)
        tgt_l = tgt_l.unsqueeze(0)
        tgt_r = tgt_r.unsqueeze(0)
        F_left_world  = F_left_world.unsqueeze(0)
        F_right_world = F_right_world.unsqueeze(0)

    B = q23.shape[0]
    e_z = torch.tensor([0., 0., 1.], device=device, dtype=q23.dtype)

    # ── Rotation matrix: base → world ────────────────────────────────────────
    R = rp_to_matrix(rp)                      # (B, 3, 3)
    g_dir_base = (R.transpose(-2, -1) @ e_z.expand(B, 3).unsqueeze(-1)).squeeze(-1)  # (B, 3)

    # ── FK positions in base frame ────────────────────────────────────────────
    feet_base  = kin.fk("feet", q23[:, _Q_FEET])   # (B, 2, 3): [left_ankle, right_ankle]
    hands_base = kin.fk("arms", q23[:, _Q_ARMS])   # (B, 2, 3): [left_hand, right_hand]
    head_base  = kin.fk("head", q23[:, _Q_HEAD])   # (B, 3)

    foot_l_base = feet_base[:, 0, :]   # (B, 3)
    foot_r_base = feet_base[:, 1, :]   # (B, 3)
    hand_l_base = hands_base[:, 0, :]  # (B, 3)
    hand_r_base = hands_base[:, 1, :]  # (B, 3)

    # ── World-frame positions ─────────────────────────────────────────────────
    def to_world(p_base):
        # p_base: (B, 3) → (B, 3)
        return (R @ p_base.unsqueeze(-1)).squeeze(-1) + h.unsqueeze(-1) * e_z

    foot_l_world = to_world(foot_l_base)   # (B, 3)
    foot_r_world = to_world(foot_r_base)
    hand_l_world = to_world(hand_l_base)
    hand_r_world = to_world(hand_r_base)
    head_world   = to_world(head_base)
    p_com_world  = to_world(kin.com_position(q23))   # (B, 3)

    # ── Rotation matrices of ankle links ─────────────────────────────────────
    R_ankle_l = kin.rot_ee("left_ankle",  q23[:, _Q_ANKLE_L])  # (B, 3, 3)
    R_ankle_r = kin.rot_ee("right_ankle", q23[:, _Q_ANKLE_R])  # (B, 3, 3)
    # Z-axis of ankle in world frame (should be [0,0,1] for flat contact).
    ankle_z_l_world = (R @ R_ankle_l[:, :, 2].unsqueeze(-1)).squeeze(-1)  # (B, 3)
    ankle_z_r_world = (R @ R_ankle_r[:, :, 2].unsqueeze(-1)).squeeze(-1)

    # ── Hand orientation in base frame ────────────────────────────────────────
    q_hand_l = q23[:, _Q_HAND_L_IDX]
    q_hand_r = q23[:, _Q_HAND_R_IDX]
    R_hand_l = kin.rot_ee("left_hand",  q_hand_l)  # (B, 3, 3)
    R_hand_r = kin.rot_ee("right_hand", q_hand_r)  # (B, 3, 3)
    rpy_r = rot_mat_to_rpy(R_hand_r)               # (B, 3)
    rpy_l = rot_mat_to_rpy(R_hand_l)               # (B, 3)
    # Mirror left RPY for symmetric check: (-roll, pitch, -yaw).
    rpy_l_sym = torch.stack([-rpy_l[:, 0], rpy_l[:, 1], -rpy_l[:, 2]], dim=-1)

    hand_rpy_lo = torch.as_tensor(HAND_RPY_LO_R, device=device, dtype=q23.dtype)
    hand_rpy_hi = torch.as_tensor(HAND_RPY_HI_R, device=device, dtype=q23.dtype)
    rpy_r_viol = (torch.relu(rpy_r    - hand_rpy_hi) + torch.relu(hand_rpy_lo - rpy_r))
    rpy_l_viol = (torch.relu(rpy_l_sym - hand_rpy_hi) + torch.relu(hand_rpy_lo - rpy_l_sym))

    # ── Gravity & contact torques (joint DOFs only, 23-dim) ──────────────────
    tau_g = GRAVITY * (kin.gravity_jac(q23) @ g_dir_base.unsqueeze(-1)).squeeze(-1)  # (B, 23)

    # Forces in base frame: F_base = R^T @ F_world
    def to_base_force(F_world):
        return (R.transpose(-2, -1) @ F_world.unsqueeze(-1)).squeeze(-1)

    foot_f_l_base = to_base_force(foot_forces[:, 0])  # (B, 3)
    foot_f_r_base = to_base_force(foot_forces[:, 1])
    hand_f_l_base = to_base_force(F_left_world)
    hand_f_r_base = to_base_force(F_right_world)

    J_feet = kin.jacobian("feet", q23[:, _Q_FEET])  # (B, 6, 12)
    J_arms = kin.jacobian("arms", q23[:, _Q_ARMS])  # (B, 6, 11)

    # tau_c = -J^T @ F_base per contact
    def jt_f(J, F):  # J:(B,3,n), F:(B,3) → (B,n)
        return -(J.transpose(-2, -1) @ F.unsqueeze(-1)).squeeze(-1)

    tau_feet_leg  = jt_f(J_feet[:, :3, :], foot_f_l_base) + jt_f(J_feet[:, 3:, :], foot_f_r_base)  # (B,12)
    tau_hands_arm = jt_f(J_arms[:, :3, :], hand_f_l_base) + jt_f(J_arms[:, 3:, :], hand_f_r_base)  # (B,11)

    tau_j = torch.cat([tau_g[:, :12] + tau_feet_leg, tau_g[:, 12:] + tau_hands_arm], dim=-1)  # (B, 23)

    # ── Static equilibrium: force + moment balance ────────────────────────────
    F_grav = torch.tensor([0., 0., -TOTAL_MASS * GRAVITY], device=device, dtype=q23.dtype).expand(B, 3)

    all_forces = torch.stack([foot_forces[:, 0], foot_forces[:, 1],
                               F_left_world, F_right_world], dim=1)   # (B, 4, 3)
    all_pos    = torch.stack([foot_l_world, foot_r_world,
                               hand_l_world, hand_r_world], dim=1)    # (B, 4, 3)

    force_res  = all_forces.sum(dim=1) + F_grav                       # (B, 3)
    moment_res = (cross3(all_pos, all_forces).sum(dim=1)
                  + cross3(p_com_world, F_grav))                       # (B, 3)
    equil_res  = torch.cat([force_res, moment_res], dim=-1)            # (B, 6)

    # ── Individual loss terms ─────────────────────────────────────────────────
    rly  = torch.relu
    q_lo = Q_LOWER.to(device)
    q_hi = Q_UPPER.to(device)
    t_lim = TAU_LIMIT.to(device)
    rp_lo = torch.as_tensor(RPY_LOWER, device=device, dtype=q23.dtype)
    rp_hi = torch.as_tensor(RPY_UPPER, device=device, dtype=q23.dtype)

    # foot z in world frame
    foot_z_l = (R[:, 2, :] * foot_l_base).sum(-1) + h - Z_FEET_0  # (B,)
    foot_z_r = (R[:, 2, :] * foot_r_base).sum(-1) + h - Z_FEET_0

    # foot box (in base frame, x,y coordinates)
    def _box_viol(fx, fy):
        return (rly(XL - fx) + rly(fx - XU) + rly(YL - fy.abs()) + rly(fy.abs() - YU))

    fx_l, fy_l = foot_l_base[:, 0], foot_l_base[:, 1]
    fx_r, fy_r = foot_r_base[:, 0], foot_r_base[:, 1]
    box = _box_viol(fx_l, fy_l) + _box_viol(fx_r, fy_r)  # (B,)

    head_x_max = torch.minimum(tgt_l[:, 0], tgt_r[:, 0])  # (B,)
    head_behind_viol = rly(head_world[:, 0] + 0.1 - head_x_max)  # (B,)

    qviol  = rly(q23 - q_hi) + rly(q_lo - q23)              # (B, 23)
    tviol  = rly(tau_j.abs() - t_lim)                        # (B, 23)
    rpviol = rly(rp - rp_hi) + rly(rp_lo - rp)               # (B, 2)
    hviol  = rly(h - H_MAX) + rly(H_MIN - h)                  # (B,)

    fz_l = foot_forces[:, 0, 2]
    fz_r = foot_forces[:, 1, 2]
    fric = (rly(-fz_l) + rly(foot_forces[:, 0, 0].abs() - MU * fz_l)
                       + rly(foot_forces[:, 0, 1].abs() - MU * fz_l)
          + rly(-fz_r) + rly(foot_forces[:, 1, 0].abs() - MU * fz_r)
                       + rly(foot_forces[:, 1, 1].abs() - MU * fz_r))  # (B,)

    w4 = torch.as_tensor(W4, device=device, dtype=q23.dtype)

    terms = {
        # ── Objective costs ──
        "tau":          W1 * (tau_j**2).sum(-1),
        "F_feet":       W2 * (foot_forces**2).sum((-2, -1)),
        "height":       W3 * (h - H0)**2,
        "upright":      (w4 * rp**2).sum(-1),
        "hip_yaw":      W5 * q23[:, HIP_YAW_IDX.to(device)].pow(2).sum(-1),
        "shoulder_roll": W6 * (rly(-q23[:, L_SROLL_IDX]).pow(2)
                             + rly( q23[:, R_SROLL_IDX]).pow(2)),
        # ── Constraint penalties ──
        "equilibrium":  LAM_EQ    * (equil_res**2).sum(-1),
        "arm":          LAM_ARM   * ((hand_l_world - tgt_l)**2 + (hand_r_world - tgt_r)**2).sum(-1),
        "arm_ori":      LAM_ARM_ORI * (rpy_l_viol**2 + rpy_r_viol**2).sum(-1),
        "head_behind":  LAM_HEAD  * head_behind_viol**2,
        "foot_z":       LAM_FOOTZ * (foot_z_l**2 + foot_z_r**2),
        "ankle_z":      LAM_ANKLE * ((ankle_z_l_world - e_z)**2 + (ankle_z_r_world - e_z)**2).sum(-1),
        "symmetry":     LAM_SYM   * ((fx_l - fx_r)**2 + (fy_l + fy_r)**2),
        "box":          LAM_BOX   * box**2,
        "qlim":         LAM_QLIM  * qviol.pow(2).sum(-1),
        "tlim":         LAM_TLIM  * tviol.pow(2).sum(-1),
        "rp":           LAM_RP    * rpviol.pow(2).sum(-1),
        "h":            LAM_H     * hviol**2,
        "friction":     LAM_FRIC  * fric**2,
    }

    cost_terms    = {k: terms[k] for k in COST_KEYS}
    penalty_terms = {k: terms[k] for k in PENALTY_KEYS}
    cost    = sum(cost_terms.values())
    penalty = sum(penalty_terms.values())
    total   = cost + penalty

    if squeeze:
        # Reduce batch dim (B=1) to scalar for single-sample usage.
        total   = total.squeeze(0)
        cost    = cost.squeeze(0)
        penalty = penalty.squeeze(0)
        cost_terms    = {k: v.squeeze(0) for k, v in cost_terms.items()}
        penalty_terms = {k: v.squeeze(0) for k, v in penalty_terms.items()}

    return total, cost, penalty, cost_terms, penalty_terms


# ============================================================
# single_verification mode
# ============================================================

def _tgt_from_user(device: str):
    """Compute fixed YAG-frame hand targets from pose_optim_config user values."""
    import pinocchio as pin
    R_user = pin.rpy.rpyToMatrix(np.append(RP_USER, 0.0))
    tgt_l = torch.as_tensor(R_user @ POS_LEFT_ARM  + np.array([0., 0., H_USER]), device=device)
    tgt_r = torch.as_tensor(R_user @ POS_RIGHT_ARM + np.array([0., 0., H_USER]), device=device)
    return tgt_l, tgt_r


def _print_loss(step, total, cost, penalty, cost_terms, penalty_terms):
    print(f"[{step:4d}] Total loss: {float(total.detach()):.4e}")
    print(f"       Cost:     {float(cost.detach()):.4e}")
    for k in COST_KEYS:
        print(f"         {k:14s} {float(cost_terms[k].detach()):.4e}")
    print(f"       Penalty:  {float(penalty.detach()):.4e}")
    for k in PENALTY_KEYS:
        print(f"         {k:14s} {float(penalty_terms[k].detach()):.4e}")


def run_single_verification(device: str = "cpu"):
    kin = _get_kin(device)
    tgt_l, tgt_r = _tgt_from_user(device)
    F_left_world  = torch.tensor([F_LEFT[0],  F_LEFT[1],  0.0], device=device)
    F_right_world = torch.tensor([F_RIGHT[0], F_RIGHT[1], 0.0], device=device)

    # Initial guess (normalised).
    half_w  = TOTAL_MASS * GRAVITY / 2.0
    half_fx = -(F_LEFT[0] + F_RIGHT[0]) / 2.0
    half_fy = -(F_LEFT[1] + F_RIGHT[1]) / 2.0
    s_f = _S_F.to(device)

    h_n  = torch.tensor(0.0,  device=device, requires_grad=True)
    rp_n = torch.zeros(2,     device=device, requires_grad=True)
    q23  = torch.zeros(NQJ,   device=device, requires_grad=True)
    F_n  = (torch.tensor([[half_fx, half_fy, half_w],
                           [half_fx, half_fy, half_w]], device=device) / s_f
            ).requires_grad_(True)

    params = [h_n, rp_n, q23, F_n]
    if OPT == "SGD":
        opt = torch.optim.SGD(params, lr=LR)
    elif OPT == "LBFGS":
        opt = torch.optim.LBFGS(params, lr=LR)
    else:
        opt = torch.optim.Adam(params, lr=LR)

    _last: dict = {}

    def closure():
        opt.zero_grad()
        h_, rp_, q_, ff_ = vars_from_normalized(h_n, rp_n, q23, F_n)
        total, cost, penalty, ct, pt = compute_terms(
            h_, rp_, q_, ff_, tgt_l, tgt_r, F_left_world, F_right_world, kin, device
        )
        total.backward()
        _last.update(total=total, cost=cost, penalty=penalty, cost_terms=ct, penalty_terms=pt)
        return total

    for step in range(STEPS + 1):
        if step < STEPS:
            opt.step(closure)
        else:
            closure()
        if step % PRINT_EVERY == 0 or step == STEPS:
            _print_loss(step, **_last)

    h, rp, q23_sol, foot_forces = vars_from_normalized(h_n, rp_n, q23, F_n)

    import pinocchio as pin
    _model_tmp = pin.buildModelFromUrdf(str(URDF_PATH), pin.JointModelFreeFlyer())
    joint_names = [_model_tmp.names[i] for i in range(2, _model_tmp.njoints)]
    print("\nSolution:")
    print(f"  h   = {float(h):.4f} m")
    print(f"  rp  = {rp.detach().cpu().numpy()} rad  (roll, pitch; yaw fixed at 0)")
    print("  q23 (joint angles):")
    for jname, val in zip(joint_names, q23_sol.detach().cpu().numpy()):
        print(f"    {jname:42s} {val:+.4f} rad")
    print(f"  F_feet =\n{foot_forces.detach().cpu().numpy()} N")

    if VISUALIZE:
        _visualize_single(h, rp, q23_sol, foot_forces, tgt_l, tgt_r, F_left_world, F_right_world)


def _visualize_single(h, rp, q23, foot_forces, tgt_l, tgt_r, F_left_world, F_right_world):
    import pinocchio as pin
    from pinocchio.visualize import MeshcatVisualizer
    import meshcat.geometry as g

    model, collision_model, visual_model = pin.buildModelsFromUrdf(
        str(URDF_PATH), str(SCRIPT_DIR), pin.JointModelFreeFlyer()
    )
    data = model.createData()
    viz = MeshcatVisualizer(model, collision_model, visual_model)
    viz.initViewer(open=True)
    viz.loadViewerModel()

    rp_n = rp.detach().cpu().numpy()
    q = np.zeros(model.nq)
    q[2] = float(h)
    R_pin = pin.rpy.rpyToMatrix(np.append(rp_n, 0.0))
    q[3:7] = pin.Quaternion(R_pin).coeffs()
    q[7:] = q23.detach().cpu().numpy()
    viz.display(q)

    # Hand target spheres.
    for name, tgt, color in [("targets/left",  tgt_l.cpu().numpy(), 0xFF3333),
                               ("targets/right", tgt_r.cpu().numpy(), 0x3333FF)]:
        viz.viewer[name].set_object(
            g.Sphere(TARGET_RADIUS), g.MeshLambertMaterial(color=color, opacity=0.6)
        )
        T = np.eye(4); T[:3, 3] = tgt
        viz.viewer[name].set_transform(T)

    # Force arrows.
    pin.framesForwardKinematics(model, data, q)
    frame_names = ["left_ankle_roll_link", "right_ankle_roll_link",
                   "left_rubber_hand",     "right_rubber_hand"]
    frame_ids   = [model.getFrameId(n) for n in frame_names]
    ff_np = foot_forces.detach().cpu().numpy()
    hand_f = [F_left_world.cpu().numpy(), F_right_world.cpu().numpy()]
    for i in range(2):
        p = data.oMf[frame_ids[i]].translation
        _draw_arrow(viz.viewer, f"forces/foot_{i}", p, ff_np[i] * FORCE_SCALE, 0x33CC33)
    for i in range(2):
        p = data.oMf[frame_ids[2 + i]].translation
        _draw_arrow(viz.viewer, f"forces/hand_{i}", p, hand_f[i] * FORCE_SCALE, 0xFFAA00)

    print("Meshcat serving the solution; press Ctrl-C to exit.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


def _draw_arrow(viewer, name, start, vec, color):
    import meshcat.geometry as g
    end = start + vec
    pts = np.column_stack([start, end])
    viewer[name].set_object(
        g.Line(g.PointsGeometry(pts), g.LineBasicMaterial(color=color, linewidth=3))
    )


# ============================================================
# batch_execution mode
# ============================================================

def _sample_problems(feasible_path: str | Path, num_pts: int, num_force_per_pt: int, device: str):
    """Sample optimization problems from the feasible list.

    Returns a list of dicts (one per problem), each containing:
      ee_pos_l, ee_pos_r  (3,) base-frame EE positions
      q_chain_l, q_chain_r  (7,) joint warm-start
      only_straight  bool
      h_target, rp_target  (,), (2,)  — fixed targets for YAG constraint
      F_left_world, F_right_world  (3,)  — sampled world forces
      source_idx  int
    """
    with h5py.File(str(feasible_path), "r") as f:
        n_total = f["right_arm/ee_pos"].shape[0]
        # Sample num_pts indices without replacement, in random order.
        indices = np.random.choice(n_total, size=min(num_pts, n_total), replace=False)
        np.random.shuffle(indices)
        indices = np.sort(indices)   # sort for efficient HDF5 slice

        ee_pos_r  = f["right_arm/ee_pos"][indices].astype(np.float64)   # (K, 3)
        ee_pos_l  = f["left_arm/ee_pos"][indices].astype(np.float64)
        q_chain_r = f["right_arm/q_chain"][indices].astype(np.float64)
        q_chain_l = f["left_arm/q_chain"][indices].astype(np.float64)
        only_s    = f["right_arm/only_straight"][indices].astype(bool)

    K = indices.shape[0]
    problems = []
    for i in range(K):
        for _ in range(num_force_per_pt):
            # Sample base pose.
            if only_s[i]:
                h_t  = H_USER
                rp_t = RP_USER.copy()
            else:
                h_t  = np.random.uniform(*H_RANGE)
                rp_t = np.array([
                    np.random.uniform(RP_RANGE[0][0], RP_RANGE[1][0]),
                    np.random.uniform(RP_RANGE[0][1], RP_RANGE[1][1]),
                ])

            # Sample hand forces.
            fx_l = np.random.uniform(*FORCE_RANGE_X)
            fy_l = np.random.uniform(*FORCE_RANGE_Y)
            fx_r = np.random.uniform(*FORCE_RANGE_X)
            fy_r = np.random.uniform(*FORCE_RANGE_Y)

            # Bake YAG-frame targets using the rp_to_matrix numpy equivalent.
            r_, p_ = rp_t[0], rp_t[1]
            cr, sr = np.cos(r_), np.sin(r_)
            cp, sp = np.cos(p_), np.sin(p_)
            R_t = np.array([[cp, sp*sr, sp*cr],
                             [0., cr,  -sr   ],
                             [-sp, cp*sr, cp*cr]])
            tgt_l_yag = R_t @ ee_pos_l[i] + np.array([0., 0., h_t])
            tgt_r_yag = R_t @ ee_pos_r[i] + np.array([0., 0., h_t])

            problems.append({
                "ee_pos_l":     ee_pos_l[i],
                "ee_pos_r":     ee_pos_r[i],
                "q_chain_l":    q_chain_l[i],
                "q_chain_r":    q_chain_r[i],
                "only_straight": bool(only_s[i]),
                "h_target":     h_t,
                "rp_target":    rp_t,
                "tgt_l":        tgt_l_yag,
                "tgt_r":        tgt_r_yag,
                "F_left_world":  np.array([fx_l, fy_l, 0.0]),
                "F_right_world": np.array([fx_r, fy_r, 0.0]),
                "source_idx":   int(indices[i]),
            })

    return problems


def _solve_batch(problems: list, kin, device: str) -> list:
    """Solve a batch of problems simultaneously using OPT_BATCH optimizer.

    Returns a list of result dicts (one per problem, in same order).
    """
    B = len(problems)
    s_f = _S_F.to(device)

    # ── Build batched initial guess ───────────────────────────────────────────
    q23_init = torch.zeros(B, NQJ, device=device)
    for i, p in enumerate(problems):
        q23_init[i, 12:15] = torch.as_tensor(p["q_chain_l"][:3], device=device)
        q23_init[i, 15:19] = torch.as_tensor(p["q_chain_l"][3:], device=device)
        q23_init[i, 19:23] = torch.as_tensor(p["q_chain_r"][3:], device=device)

    h_targets  = torch.tensor([p["h_target"]  for p in problems], device=device)
    rp_targets = torch.tensor([p["rp_target"] for p in problems], device=device)
    tgt_l_b    = torch.tensor([p["tgt_l"]     for p in problems], device=device)
    tgt_r_b    = torch.tensor([p["tgt_r"]     for p in problems], device=device)
    F_left_b   = torch.tensor([p["F_left_world"]  for p in problems], device=device)
    F_right_b  = torch.tensor([p["F_right_world"] for p in problems], device=device)

    # Warm-start h_n and rp_n from sampled targets.
    h_n_init  = (h_targets - H0) / _S_H
    rp_n_init = rp_targets / _S_RP
    half_w    = TOTAL_MASS * GRAVITY / 2.0
    half_fx   = -(F_left_b[:, 0] + F_right_b[:, 0]) / 2.0
    half_fy   = -(F_left_b[:, 1] + F_right_b[:, 1]) / 2.0
    F_foot_init = torch.stack([half_fx, half_fy,
                                torch.full((B,), half_w, device=device)], dim=-1)
    F_n_init  = F_foot_init.unsqueeze(1).expand(B, 2, 3) / s_f

    h_n  = h_n_init.clone().requires_grad_(True)
    rp_n = rp_n_init.clone().requires_grad_(True)
    q23  = q23_init.clone().requires_grad_(True)
    F_n  = F_n_init.clone().requires_grad_(True)

    params = [h_n, rp_n, q23, F_n]
    if OPT_BATCH == "SGD":
        opt = torch.optim.SGD(params, lr=LR_BATCH)
    else:
        opt = torch.optim.Adam(params, lr=LR_BATCH)

    for step in range(STEPS_BATCH):
        opt.zero_grad()
        h_, rp_, q_, ff_ = vars_from_normalized(h_n, rp_n, q23, F_n)
        total, _, _, _, _ = compute_terms(
            h_, rp_, q_, ff_, tgt_l_b, tgt_r_b, F_left_b, F_right_b, kin, device
        )
        total.sum().backward()
        opt.step()

    # ── Evaluate final loss ───────────────────────────────────────────────────
    with torch.no_grad():
        h_sol, rp_sol, q23_sol, ff_sol = vars_from_normalized(h_n, rp_n, q23, F_n)
        _, cost_b, penalty_b, cost_terms_b, penalty_terms_b = compute_terms(
            h_sol, rp_sol, q23_sol, ff_sol,
            tgt_l_b, tgt_r_b, F_left_b, F_right_b, kin, device
        )

    # ── Collect per-sample results ────────────────────────────────────────────
    results = []
    for i, p in enumerate(problems):
        pen_i = float(penalty_b[i].cpu())
        if pen_i > MAX_PENALTY_DISCARD:
            continue
        results.append({
            # problem inputs
            "ee_pos_l":      p["ee_pos_l"],
            "ee_pos_r":      p["ee_pos_r"],
            "F_left_world":  p["F_left_world"],
            "F_right_world": p["F_right_world"],
            "h_target":      p["h_target"],
            "rp_target":     p["rp_target"],
            "only_straight": p["only_straight"],
            "source_idx":    p["source_idx"],
            # solution
            "h":           float(h_sol[i].cpu()),
            "rp":          rp_sol[i].cpu().numpy(),
            "q23":         q23_sol[i].cpu().numpy(),
            "foot_forces": ff_sol[i].cpu().numpy(),   # (2, 3)
            # losses (scalars)
            "cost":    float(cost_b[i].cpu()),
            "penalty": pen_i,
            "total":   float((cost_b[i] + penalty_b[i]).cpu()),
            **{k: float(cost_terms_b[k][i].cpu())    for k in COST_KEYS},
            **{k: float(penalty_terms_b[k][i].cpu()) for k in PENALTY_KEYS},
        })
    return results


def _write_results_hdf5(all_results: list, output_path: str | Path):
    """Write all collected results to optim_pose_data.hdf5."""
    if not all_results:
        print("[WARN] No solutions to write.")
        return

    N = len(all_results)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    def _ds(grp, key, dtype):
        arr = np.array([r[key] for r in all_results], dtype=dtype)
        grp.create_dataset(key, data=arr, compression="gzip", compression_opts=4)

    with h5py.File(str(output_path), "w") as f:
        f.attrs["n_samples"]   = N
        f.attrs["timestamp"]   = time.strftime("%Y-%m-%dT%H:%M:%S")
        f.attrs["feasible_list_path"] = str(FEASIBLE_LIST_PATH)
        f.attrs["joint_names"] = [
            "left_hip_pitch", "left_hip_roll", "left_hip_yaw",
            "left_knee", "left_ankle_pitch", "left_ankle_roll",
            "right_hip_pitch", "right_hip_roll", "right_hip_yaw",
            "right_knee", "right_ankle_pitch", "right_ankle_roll",
            "waist_yaw", "waist_roll", "waist_pitch",
            "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
            "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow",
        ]

        prb = f.create_group("problems")
        for key, dtype in [
            ("ee_pos_l",      np.float32),
            ("ee_pos_r",      np.float32),
            ("F_left_world",  np.float32),
            ("F_right_world", np.float32),
            ("h_target",      np.float32),
            ("rp_target",     np.float32),
            ("source_idx",    np.int32),
        ]:
            _ds(prb, key, dtype)
        _ds(prb, "only_straight", bool)

        sol = f.create_group("solutions")
        for key, dtype in [
            ("h",           np.float32),
            ("rp",          np.float32),
            ("q23",         np.float32),
            ("foot_forces", np.float32),
        ]:
            _ds(sol, key, dtype)

        loss = f.create_group("losses")
        for key in list(COST_KEYS) + list(PENALTY_KEYS) + ["total", "cost", "penalty"]:
            _ds(loss, key, np.float32)

    print(f"[INFO] Wrote {N} solutions to {output_path}")


def run_batch_execution(device: str = "cpu"):
    kin = _get_kin(device)
    print(f"[INFO] Sampling {NUM_PTS} arm configs × {NUM_FORCE_PER_POINT} force/pose combos "
          f"from {FEASIBLE_LIST_PATH}")

    problems = _sample_problems(FEASIBLE_LIST_PATH, NUM_PTS, NUM_FORCE_PER_POINT, device)
    print(f"[INFO] Total problems: {len(problems)}, batch size: {BATCH_SIZE}")

    all_results = []
    n_batches = (len(problems) + BATCH_SIZE - 1) // BATCH_SIZE
    t0 = time.time()
    for bi in range(n_batches):
        batch = problems[bi * BATCH_SIZE: (bi + 1) * BATCH_SIZE]
        results = _solve_batch(batch, kin, device)
        all_results.extend(results)
        elapsed = time.time() - t0
        print(f"[INFO] Batch {bi+1}/{n_batches}: "
              f"{len(results)}/{len(batch)} kept, "
              f"total kept: {len(all_results)}, "
              f"elapsed: {elapsed:.1f}s")

    _write_results_hdf5(all_results, OUTPUT_H5_PATH)


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Torch-native pose optimizer for G1.")
    parser.add_argument("--mode", choices=["single_verification", "batch_execution"],
                        default=MODE, help="Execution mode.")
    parser.add_argument("--device", default="cpu", help="Torch device (cpu / cuda:0 …).")
    args = parser.parse_args()

    if args.mode == "single_verification":
        run_single_verification(device=args.device)
    else:
        run_batch_execution(device=args.device)
