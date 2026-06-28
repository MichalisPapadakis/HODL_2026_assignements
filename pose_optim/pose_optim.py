"""Static pose optimization for the Unitree G1 (29 DoF) humanoid.

We solve, for a fixed-feet / fixed-hand-target static pose:

    min  w1||tau||^2 + w2||F_feet||^2 + w3(h-h0)^2 + w4||rp||^2
    s.t. (soft penalties)
         w_R_b(r,p) * b_FK_{l/r hand}(q) + [0;0;h] == target_{l/r}    (YAG frame)
         w_R_b(r,p) * b_FK_{l/r foot}_z  + h       == 0               (feet on flat floor)
         ankle z-axis upright in world frame                          (flat foot contact)
         FK_foot x,y in box, left/right symmetric                     (pelvis frame)
         q in joint limits (URDF + extra: hip_yaw in ±60°, elbow < 90°,
             left_shoulder_yaw > -10°, right_shoulder_roll < 10°)
         hand orientation RPY in base frame (right: filter ranges; left: mirrored roll/yaw)
         head_link x in base frame < min(POS_LEFT_ARM[0], POS_RIGHT_ARM[0])  (head behind arms)
         |tau_joints| <= URDF effort limits
         rp in [roll, pitch] bounds;  h in [H_MIN, H_MAX]
         Coulomb friction pyramid on each foot force                  (world vertical normal)
         static equilibrium of the unactuated base:  tau_base == 0
    where tau = tG(q) - sum_c J_c^T F_c  (feet: decision vars, hands: fixed inputs).

Yaw-aligned ground (YAG) frame: origin at ground (z=0), z-up, yaw=0.
Base is placed at [0, 0, h] with rotation R(roll, pitch, 0); yaw is not a
decision variable because rotating the base and the target by the same angle
leaves the arm constraint invariant (yaw is unobservable, so it is fixed at 0).

Hand targets are specified in base frame and baked into fixed YAG-frame targets
via T(H_USER, RP_USER): target_yag = R(RP_USER, 0) @ target_base + [0, 0, H_USER].

Kinematics / dynamics + their analytical derivatives come from Pinocchio and are
exposed to a generic torch optimizer through custom autograd Functions, so the
same structure ports to a batched native-torch implementation later (just swap
the Function internals for differentiable torch ops).
"""

from pathlib import Path

import numpy as np
import pinocchio as pin
import torch

torch.set_default_dtype(torch.float64)

# ============================================================
# Configuration — loaded from pose_optim_config.py
# ============================================================
from pose_optim_config import (  # noqa: E402
    URDF_PATH,
    POS_LEFT_ARM, POS_RIGHT_ARM,
    H_USER, RP_USER,
    F_LEFT, F_RIGHT,
    W1, W2, W3, W4, W5, W6,
    LAM_EQ, LAM_FOOTZ, LAM_ARM, LAM_BOX, LAM_QLIM, LAM_TLIM,
    LAM_RP, LAM_H, LAM_FRIC, LAM_ANKLE, LAM_ARM_ORI, LAM_HEAD, LAM_SYM,
    H0, Z_FEET_0, H_MIN, H_MAX, DEG,
    RPY_LOWER, RPY_UPPER, MU, XL, XU, YL, YU,
    HAND_RPY_LO_R, HAND_RPY_HI_R,
    OPT, LR, STEPS, PRINT_EVERY,
    VISUALIZE, FORCE_SCALE, TARGET_RADIUS,
)

SCRIPT_DIR = Path(__file__).resolve().parent

# ============================================================
# Model
# ============================================================
model = pin.buildModelFromUrdf(str(URDF_PATH), pin.JointModelFreeFlyer())
data = model.createData()

NV  = model.nv       # total velocity DOFs (6 base + 23 joints)
NJ  = NV - 6         # 23 actuated joints
NQJ = model.nq - 7   # 23 joint configuration entries

# Contact frames: order is [left_foot, right_foot, left_hand, right_hand].
FRAME_NAMES = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_rubber_hand",
    "right_rubber_hand",
]
FRAME_IDS = [model.getFrameId(n) for n in FRAME_NAMES]
FOOT_IDS  = FRAME_IDS[:2]

Q_LOWER   = torch.as_tensor(model.lowerPositionLimit[7:])
Q_UPPER   = torch.as_tensor(model.upperPositionLimit[7:])

# Extra joint constraints (tighter than URDF); folded into the qlim penalty.
_JOINT_NAMES_Q23 = [model.names[i] for i in range(2, model.njoints)]


def _q23_idx(name):
    return _JOINT_NAMES_Q23.index(name)


for _name, _lo, _hi in (
    ("left_hip_yaw_joint",        -60.0 * DEG,  60.0 * DEG),
    ("right_hip_yaw_joint",       -60.0 * DEG,  60.0 * DEG),
    ("left_elbow_joint",            None,       90.0 * DEG),
    ("right_elbow_joint",           None,       90.0 * DEG),
    ("left_shoulder_roll_joint",   -10.0 * DEG,    None),
    ("right_shoulder_roll_joint",   None,       10.0 * DEG),
):
    _i = _q23_idx(_name)
    if _lo is not None:
        Q_LOWER[_i] = torch.maximum(Q_LOWER[_i], torch.tensor(_lo))
    if _hi is not None:
        Q_UPPER[_i] = torch.minimum(Q_UPPER[_i], torch.tensor(_hi))

HIP_YAW_Q23_IDX = torch.tensor([
    _q23_idx("left_hip_yaw_joint"),
    _q23_idx("right_hip_yaw_joint"),
], dtype=torch.long)
LEFT_SHOULDER_ROLL_Q23_IDX = _q23_idx("left_shoulder_roll_joint")
RIGHT_SHOULDER_ROLL_Q23_IDX = _q23_idx("right_shoulder_roll_joint")

TAU_LIMIT = torch.as_tensor(model.effortLimit[6:6 + NJ])  # URDF effort [N·m]
RPY_LO    = torch.as_tensor(RPY_LOWER)   # (2,) roll, pitch
RPY_HI    = torch.as_tensor(RPY_UPPER)   # (2,)
TOTAL_MASS = float(sum(model.inertias[i].mass for i in range(1, model.njoints)))
print(f"total mass: {TOTAL_MASS}")
GRAVITY    = float(abs(model.gravity.linear[2]))

# Decision-variable scaling (optimizer works in normalized space; q23 is unscaled).
S_H = 0.10   # meters
S_RP = 0.30  # radians
S_F = torch.tensor([100.0, 100.0, TOTAL_MASS * GRAVITY / 2.0])  # per (Fx, Fy, Fz) [N]

# Fixed YAG-frame hand targets, computed from base-frame targets + user transform.
_R_user       = pin.rpy.rpyToMatrix(np.append(RP_USER, 0.0))   # yaw = 0
TGT_LEFT_YAG  = _R_user @ POS_LEFT_ARM  + np.array([0., 0., H_USER])
TGT_RIGHT_YAG = _R_user @ POS_RIGHT_ARM + np.array([0., 0., H_USER])

HAND_RPY_LO = torch.as_tensor(HAND_RPY_LO_R)
HAND_RPY_HI = torch.as_tensor(HAND_RPY_HI_R)
HAND_FRAME_IDS = FRAME_IDS[2:]   # [left_hand, right_hand]
HEAD_FRAME_ID = model.getFrameId("d435_link")
HEAD_X_MAX = float(min(TGT_LEFT_YAG[0], TGT_RIGHT_YAG[0]))   # head x must stay below this [m]


def make_config(h, rpy, q23):
    """Assemble the Pinocchio config nq = [0, 0, h, quat_xyzw, q23] (numpy)."""
    q = np.zeros(model.nq)
    q[2] = float(h)
    R = pin.rpy.rpyToMatrix(np.asarray(rpy, dtype=float))
    q[3:7] = pin.Quaternion(R).coeffs()  # (x, y, z, w)
    q[7:] = np.asarray(q23, dtype=float)
    return q


def _np(t):
    return t.detach().cpu().numpy().astype(float)


# ============================================================
# Pinocchio <-> torch bridge (analytical derivatives)
# ============================================================
class RelFramePos(torch.autograd.Function):
    """Positions of the contact frames in the pelvis frame, as a function of q23.

    The base is held at identity, so frame placements are expressed relative to
    the pelvis and depend only on the joint angles.
    """

    @staticmethod
    def forward(ctx, q23):
        q = np.zeros(model.nq)
        q[6] = 1.0  # identity quaternion w-component
        q[7:] = _np(q23)
        pin.framesForwardKinematics(model, data, q)
        pos = np.stack([data.oMf[fid].translation for fid in FRAME_IDS])  # (4, 3)
        ctx.q = q
        return torch.as_tensor(pos)

    @staticmethod
    def backward(ctx, grad_out):
        g = _np(grad_out)  # (4, 3)
        grad_q23 = np.zeros(NQJ)
        for i, fid in enumerate(FRAME_IDS):
            J = pin.computeFrameJacobian(model, data, ctx.q, fid,
                                         pin.LOCAL_WORLD_ALIGNED)
            grad_q23 += J[:3, 6:].T @ g[i]
        return torch.as_tensor(grad_q23)


class HeadPosBase(torch.autograd.Function):
    """head_link (d435_link in urdf) position in the pelvis frame (3,), from q23."""

    @staticmethod
    def forward(ctx, q23):
        q = np.zeros(model.nq)
        q[6] = 1.0
        q[7:] = _np(q23)
        pin.framesForwardKinematics(model, data, q)
        pos = data.oMf[HEAD_FRAME_ID].translation.copy()
        ctx.q = q
        return torch.as_tensor(pos)

    @staticmethod
    def backward(ctx, grad_out):
        g = _np(grad_out)
        J = pin.computeFrameJacobian(model, data, ctx.q, HEAD_FRAME_ID,
                                     pin.LOCAL_WORLD_ALIGNED)
        grad_q23 = J[:3, 6:].T @ g
        return torch.as_tensor(grad_q23)


class HandRPY(torch.autograd.Function):
    """Hand orientations in the pelvis frame as extrinsic XYZ RPY (2, 3), from q23."""

    @staticmethod
    def forward(ctx, q23):
        q = np.zeros(model.nq)
        q[6] = 1.0
        q[7:] = _np(q23)
        pin.framesForwardKinematics(model, data, q)
        rpy = np.stack([pin.rpy.matrixToRpy(data.oMf[fid].rotation)
                        for fid in HAND_FRAME_IDS])
        ctx.q23 = _np(q23)
        return torch.as_tensor(rpy)

    @staticmethod
    def backward(ctx, grad_out):
        g = _np(grad_out)  # (2, 3)

        def s(q23):
            q = np.zeros(model.nq)
            q[6] = 1.0
            q[7:] = q23
            pin.framesForwardKinematics(model, data, q)
            rpy = np.stack([pin.rpy.matrixToRpy(data.oMf[fid].rotation)
                            for fid in HAND_FRAME_IDS])
            return (g * rpy).sum()

        eps = 1e-6
        grad_q23 = np.zeros(NQJ)
        for k in range(NQJ):
            d = np.zeros(NQJ); d[k] = eps
            grad_q23[k] = (s(ctx.q23 + d) - s(ctx.q23 - d)) / (2 * eps)
        return torch.as_tensor(grad_q23)


class FootAnkleZWorld(torch.autograd.Function):
    """World-frame z-axis of each ankle_roll link (2, 3), from (rp, q23)."""

    @staticmethod
    def forward(ctx, rp, q23):
        rpy_n = np.append(_np(rp), 0.0)   # yaw = 0
        q = make_config(0.0, rpy_n, _np(q23))
        pin.framesForwardKinematics(model, data, q)
        z = np.stack([data.oMf[fid].rotation[:, 2] for fid in FOOT_IDS])
        ctx.rp, ctx.q23 = _np(rp), _np(q23)
        return torch.as_tensor(z)

    @staticmethod
    def backward(ctx, grad_out):
        g = _np(grad_out)  # (2, 3)

        def s(rp, q23):
            q = make_config(0.0, np.append(rp, 0.0), q23)
            pin.framesForwardKinematics(model, data, q)
            z = np.stack([data.oMf[fid].rotation[:, 2] for fid in FOOT_IDS])
            return (g * z).sum()

        eps = 1e-6
        grad_rp = np.zeros(2)
        for k in range(2):
            d = np.zeros(2); d[k] = eps
            grad_rp[k] = (s(ctx.rp + d, ctx.q23) - s(ctx.rp - d, ctx.q23)) / (2 * eps)
        grad_q23 = np.zeros(NQJ)
        for k in range(NQJ):
            d = np.zeros(NQJ); d[k] = eps
            grad_q23[k] = (s(ctx.rp, ctx.q23 + d) - s(ctx.rp, ctx.q23 - d)) / (2 * eps)
        return torch.as_tensor(grad_rp), torch.as_tensor(grad_q23)


def _hand_forces_world():
    return np.array([[F_LEFT[0],  F_LEFT[1],  0.0],
                     [F_RIGHT[0], F_RIGHT[1], 0.0]])


def _tau_eval(rpy, q23, foot_forces):
    """tau = tG(q) - sum_c J_{c,v}^T F_c  (world-frame point forces). Returns (tau, foot_J)."""
    q = make_config(0.0, rpy, q23)
    forces = np.vstack([foot_forces, _hand_forces_world()])
    tau = pin.computeGeneralizedGravity(model, data, q).copy()
    foot_J = []
    for k, fid in enumerate(FRAME_IDS):
        Jv = pin.computeFrameJacobian(model, data, q, fid, pin.LOCAL_WORLD_ALIGNED)[:3]
        tau = tau - Jv.T @ forces[k]
        if k < 2:
            foot_J.append(Jv)
    return tau, foot_J


class StaticTorque(torch.autograd.Function):
    """tau = tG(q) - sum_c J_c^T F_c  (full nv vector).

    Inputs: rp (2), q23 (23), foot_forces (2, 3) in world frame.
    Hand forces are fixed module inputs. Base position does not affect tau.

    Derivatives: d tau/d F = -J_foot^T is analytical and exact. The contact
    forces are constant in the *world* frame, whose exact d tau/d q needs the
    coordinate position Hessian (Pinocchio's static-torque / kinematic-Hessian
    helpers assume joint-local-constant forces, which differs). We therefore
    central-difference Pinocchio's analytical gravity+Jacobian tau function over
    (rp, q23); in the future native-torch port this block is just autograd.
    """

    @staticmethod
    def forward(ctx, rp, q23, foot_forces):
        rp_n, q23_n, ff_n = _np(rp), _np(q23), _np(foot_forces)
        tau, foot_J = _tau_eval(np.append(rp_n, 0.0), q23_n, ff_n)
        ctx.rp, ctx.q23, ctx.ff, ctx.foot_J = rp_n, q23_n, ff_n, foot_J
        return torch.as_tensor(tau)

    @staticmethod
    def backward(ctx, grad_out):
        g = _np(grad_out)  # (nv,) = dL/dtau

        def s(rp, q23):
            return g @ _tau_eval(np.append(rp, 0.0), q23, ctx.ff)[0]

        eps = 1e-6
        grad_rp = np.zeros(2)
        for k in range(2):
            d = np.zeros(2); d[k] = eps
            grad_rp[k] = (s(ctx.rp + d, ctx.q23) - s(ctx.rp - d, ctx.q23)) / (2 * eps)
        grad_q23 = np.zeros(NQJ)
        for k in range(NQJ):
            d = np.zeros(NQJ); d[k] = eps
            grad_q23[k] = (s(ctx.rp, ctx.q23 + d) - s(ctx.rp, ctx.q23 - d)) / (2 * eps)

        grad_ff = np.stack([-Jk @ g for Jk in ctx.foot_J])  # d tau/dF = -J^T
        return (torch.as_tensor(grad_rp),
                torch.as_tensor(grad_q23),
                torch.as_tensor(grad_ff))


# ============================================================
# Loss (penalty method)
# ============================================================
def rpy_to_matrix_torch(rpy):
    r, p, y = rpy[0], rpy[1], rpy[2]
    cr, sr = torch.cos(r), torch.sin(r)
    cp, sp = torch.cos(p), torch.sin(p)
    cy, sy = torch.cos(y), torch.sin(y)
    Rx = torch.stack([torch.tensor(1.0), torch.tensor(0.0), torch.tensor(0.0),
                      torch.tensor(0.0), cr, -sr,
                      torch.tensor(0.0), sr, cr]).reshape(3, 3)
    Ry = torch.stack([cp, torch.tensor(0.0), sp,
                      torch.tensor(0.0), torch.tensor(1.0), torch.tensor(0.0),
                      -sp, torch.tensor(0.0), cp]).reshape(3, 3)
    Rz = torch.stack([cy, -sy, torch.tensor(0.0),
                      sy, cy, torch.tensor(0.0),
                      torch.tensor(0.0), torch.tensor(0.0), torch.tensor(1.0)]).reshape(3, 3)
    return Rz @ Ry @ Rx


def weighted_sq(residual, weight):
    return (torch.as_tensor(weight) * residual.pow(2)).sum()


COST_KEYS = ("tau", "F_feet", "height", "upright", "hip_yaw", "shoulder_roll")
PENALTY_KEYS = (
    "equilibrium", "arm", "arm_ori", "head_behind", "foot_z", "ankle_z", "symmetry", "box",
    "qlim", "tlim", "rp", "h", "friction",
)


def compute_terms(h, rp, q23, foot_forces):
    """Return a dict of all (named) scalar loss terms for diagnostics + total."""
    rel = RelFramePos.apply(q23)               # (4, 3) pelvis frame
    tau = StaticTorque.apply(rp, q23, foot_forces)  # (nv,)

    foot_rel = rel[:2]   # (2, 3) pelvis frame
    hand_rel = rel[2:]   # (2, 3) pelvis frame

    # w_R_b: rotation from base to world (yaw = 0).
    R = rpy_to_matrix_torch(torch.cat([rp, torch.zeros(1)]))

    foot_world_z = (foot_rel @ R.T)[:, 2] + h - Z_FEET_0   # world z of each foot
    ankle_z_world = FootAnkleZWorld.apply(rp, q23)   # (2, 3)
    world_up = torch.tensor([0.0, 0.0, 1.0])

    # Hand FK in YAG/world frame: p_yag = R @ p_base + [0, 0, h].
    tgt       = torch.as_tensor(np.stack([TGT_LEFT_YAG, TGT_RIGHT_YAG]))
    hand_world = hand_rel @ R.T + torch.tensor([0., 0., 1.]) * h   # (2, 3)

    hand_rpy = HandRPY.apply(q23)   # (2, 3): [left, right] extrinsic XYZ in base frame
    rpy_r_viol = (torch.relu(hand_rpy[1] - HAND_RPY_HI)
                  + torch.relu(HAND_RPY_LO - hand_rpy[1]))
    rpy_l_sym = torch.stack([-hand_rpy[0, 0], hand_rpy[0, 1], -hand_rpy[0, 2]])
    rpy_l_viol = (torch.relu(rpy_l_sym - HAND_RPY_HI)
                  + torch.relu(HAND_RPY_LO - rpy_l_sym))

    head_x = (HeadPosBase.apply(q23)@R.T)[0]
    head_behind_viol = torch.relu(head_x+0.1 - HEAD_X_MAX)

    fx = foot_rel[:, 0]
    fy = foot_rel[:, 1]
    box = (torch.relu(XL - fx) + torch.relu(fx - XU)
           + torch.relu(YL - fy.abs()) + torch.relu(fy.abs() - YU))

    qviol  = torch.relu(q23 - Q_UPPER) + torch.relu(Q_LOWER - q23)
    tau_j  = tau[6:]
    tviol  = torch.relu(tau_j.abs() - TAU_LIMIT)
    rpviol = torch.relu(rp - RPY_HI) + torch.relu(RPY_LO - rp)
    hviol  = torch.relu(h - H_MAX) + torch.relu(H_MIN - h)

    fz = foot_forces[:, 2]
    fric = (torch.relu(-fz)
            + torch.relu(foot_forces[:, 0].abs() - MU * fz)
            + torch.relu(foot_forces[:, 1].abs() - MU * fz))

    terms = {
        "tau":         W1 * weighted_sq(tau[6:], 1.0),
        "F_feet":      W2 * weighted_sq(foot_forces, 1.0),
        "height":      W3 * (h - H0) ** 2,
        "upright":     weighted_sq(rp, W4),
        "hip_yaw":     W5 * q23[HIP_YAW_Q23_IDX].pow(2).sum(),
        "shoulder_roll": W6 * (
            torch.relu(-q23[LEFT_SHOULDER_ROLL_Q23_IDX]).pow(2)
            + torch.relu(q23[RIGHT_SHOULDER_ROLL_Q23_IDX]).pow(2)
        ),
        "equilibrium": LAM_EQ   * weighted_sq(tau[:6], 1.0),
        "arm":         LAM_ARM  * (hand_world - tgt).pow(2).sum(),
        "arm_ori":     LAM_ARM_ORI * (rpy_l_viol.pow(2).sum() + rpy_r_viol.pow(2).sum()),
        "head_behind": LAM_HEAD * head_behind_viol.pow(2),
        "foot_z":      LAM_FOOTZ * foot_world_z.pow(2).sum(),
        "ankle_z":     LAM_ANKLE * (ankle_z_world - world_up).pow(2).sum(),
        "symmetry":    LAM_SYM  * ((fx[0] - fx[1]) ** 2 + (fy[0] + fy[1]) ** 2),
        "box":         LAM_BOX  * box.pow(2).sum(),
        "qlim":        LAM_QLIM * qviol.pow(2).sum(),
        "tlim":        LAM_TLIM * tviol.pow(2).sum(),
        "rp":          LAM_RP   * rpviol.pow(2).sum(),
        "h":           LAM_H    * hviol.pow(2),
        "friction":    LAM_FRIC * fric.pow(2).sum(),
    }
    cost_terms    = {k: terms[k] for k in COST_KEYS}
    penalty_terms = {k: terms[k] for k in PENALTY_KEYS}
    cost    = sum(cost_terms.values())
    penalty = sum(penalty_terms.values())
    total   = cost + penalty
    return total, cost, penalty, cost_terms, penalty_terms


def print_loss(step, total, cost, penalty, cost_terms, penalty_terms):
    """Print total / cost / penalty and each term on separate lines."""
    print(f"[{step:4d}] Total loss: {float(total.detach()):.4e}")
    print(f"       Cost:     {float(cost.detach()):.4e}")
    for k in COST_KEYS:
        print(f"         {k:12s} {float(cost_terms[k].detach()):.4e}")
    print(f"       Penalty:  {float(penalty.detach()):.4e}")
    for k in PENALTY_KEYS:
        print(f"         {k:12s} {float(penalty_terms[k].detach()):.4e}")


# ============================================================
# Optimization
# ============================================================
def vars_from_normalized(h_n, rp_n, q23, F_n):
    """Map normalized optimizer variables to physical units."""
    h = H0 + S_H * h_n
    rp = S_RP * rp_n
    foot_forces = F_n * S_F
    return h, rp, q23, foot_forces


def initial_guess_normalized():
    """Physical initial guess; returned tensors are normalized for the optimizer."""
    h = torch.tensor(H0)
    rp = torch.zeros(2)   # (roll, pitch); yaw fixed at 0
    q23 = torch.zeros(NQJ)
    half_weight = TOTAL_MASS * GRAVITY / 2.0
    half_fx = -(F_LEFT[0] + F_RIGHT[0]) / 2
    half_fy = -(F_LEFT[1] + F_RIGHT[1]) / 2
    foot_forces = torch.tensor([[half_fx, half_fy, half_weight],
                                [half_fx, half_fy, half_weight]])

    h_n = ((h - H0) / S_H).requires_grad_(True)
    rp_n = (rp / S_RP).requires_grad_(True)
    q23 = q23.requires_grad_(True)
    F_n = (foot_forces / S_F).requires_grad_(True)
    return h_n, rp_n, q23, F_n


def solve():
    h_n, rp_n, q23, F_n = initial_guess_normalized()
    params = [h_n, rp_n, q23, F_n]
    if OPT == "SGD":
        opt = torch.optim.SGD(params, lr=LR)
    elif OPT == "LBFGS":
        opt = torch.optim.LBFGS(params, lr=LR)
    else:
        opt = torch.optim.Adam(params, lr=LR)

    # State shared with the closure so print_loss can access the latest terms.
    _last = {}

    def closure():
        opt.zero_grad()
        h, rp, _, foot_forces = vars_from_normalized(h_n, rp_n, q23, F_n)
        total, cost, penalty, cost_terms, penalty_terms = compute_terms(
            h, rp, q23, foot_forces
        )
        total.backward()
        _last.update(total=total, cost=cost, penalty=penalty,
                     cost_terms=cost_terms, penalty_terms=penalty_terms)
        return total

    for step in range(STEPS + 1):
        if step < STEPS:
            opt.step(closure)
        else:
            closure()
        if step % PRINT_EVERY == 0 or step == STEPS:
            print_loss(step, **_last)

    return vars_from_normalized(h_n, rp_n, q23, F_n)


# ============================================================
# Visualization
# ============================================================
def draw_arrow(viewer, name, start, vec, color):
    import meshcat.geometry as g
    end = start + vec
    pts = np.column_stack([start, end])
    viewer[name].set_object(
        g.Line(g.PointsGeometry(pts), g.LineBasicMaterial(color=color, linewidth=3))
    )


def visualize(h, rp, q23, foot_forces):
    import meshcat.geometry as g
    from pinocchio.visualize import MeshcatVisualizer

    _, collision_model, visual_model = pin.buildModelsFromUrdf(
        str(URDF_PATH), str(SCRIPT_DIR), pin.JointModelFreeFlyer()
    )
    viz = MeshcatVisualizer(model, collision_model, visual_model)
    viz.initViewer(open=True)
    viz.loadViewerModel()

    h_n   = float(h)
    rpy_n = np.append(_np(rp), 0.0)   # yaw = 0
    q = make_config(h_n, rpy_n, _np(q23))
    viz.display(q)

    # Hand target spheres — TGT_*_YAG are already in world/YAG frame.
    for name, tgt_yag, color in [("targets/left",  TGT_LEFT_YAG,  0xFF3333),
                                   ("targets/right", TGT_RIGHT_YAG, 0x3333FF)]:
        viz.viewer[name].set_object(
            g.Sphere(TARGET_RADIUS), g.MeshLambertMaterial(color=color, opacity=0.6)
        )
        T = np.eye(4)
        T[:3, 3] = tgt_yag
        viz.viewer[name].set_transform(T)

    # Force arrows at the hands (fixed inputs) and feet (solution reactions).
    pin.framesForwardKinematics(model, data, q)
    hand_forces = [np.array([F_LEFT[0],  F_LEFT[1],  0.0]),
                   np.array([F_RIGHT[0], F_RIGHT[1], 0.0])]
    foot_forces_n = _np(foot_forces)
    for i, fid in enumerate(FOOT_IDS):
        p_w = data.oMf[fid].translation
        draw_arrow(viz.viewer, f"forces/foot_{i}", p_w,
                   foot_forces_n[i] * FORCE_SCALE, 0x33CC33)
    for i, fid in enumerate(FRAME_IDS[2:]):
        p_w = data.oMf[fid].translation
        draw_arrow(viz.viewer, f"forces/hand_{i}", p_w,
                   hand_forces[i] * FORCE_SCALE, 0xFFAA00)

    print("Meshcat is serving the solution; press Ctrl-C to exit.")
    import time
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


# ============================================================
if __name__ == "__main__":
    h, rp, q23, foot_forces = solve()

    joint_names = [model.names[i] for i in range(2, model.njoints)]

    print("\nSolution:")
    print(f"  h   = {float(h):.4f} m")
    print(f"  rp  = {np.round(_np(rp), 4)} rad  (roll, pitch; yaw fixed at 0)")
    print("  q23 (joint angles):")
    for jname, val in zip(joint_names, _np(q23)):
        print(f"    {jname:42s} {val:+.4f} rad")
    print(f"  F_feet =\n{np.round(_np(foot_forces), 2)} N")
    if VISUALIZE:
        visualize(h, rp, q23, foot_forces)
