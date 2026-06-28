#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Visualise pose-optimisation results from optim_pose_data.hdf5 in Isaac Lab.

Controls:
  K – advance to the next solution

On each advance:
  • The robot is teleported to the solution pose (h, rp, q23).
  • Force arrows are drawn at feet (solution foot reactions) and
    hands (sampled input hand forces).
  • Hand-target spheres mark the YAG-frame targets used during optimisation.
  • All loss components are printed to the terminal.

Usage (inside the Isaac Sim python env):
  python visualize_optim_results.py [--device cuda:0] [--headless]
  python visualize_optim_results.py --results path/to/optim_pose_data.hdf5
"""

from __future__ import annotations

import argparse
import math

from isaaclab.app import AppLauncher

# ─── Config ────────────────────────────────────────────────────────────────────
from pathlib import Path

RESULTS_PATH  = str(Path(__file__).resolve().parent / "data" / "optim_pose_data.hdf5")
FORCE_SCALE   = 2.5e-2   # arrow length per Newton [m/N]
SPHERE_RADIUS = 0.03     # target sphere radius [m]
# ──────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Visualise pose-optimisation results in Isaac Lab.")
parser.add_argument("--results", default=RESULTS_PATH, help="Path to optim_pose_data.hdf5.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Everything below runs after Isaac Sim is up ────────────────────────────────
import numpy as np
import torch
import h5py
import carb          # pyright: ignore[reportMissingImports]
import omni.appwindow  # pyright: ignore[reportMissingImports]

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.visualization_markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz

import isaac_humanoid_suite.tasks  # noqa: F401
from isaac_humanoid_suite.assets.unitree import G1_LOCK_WRIST_CFG

# ─── Joint name lists (same order as q23 in optim_pose_data.hdf5) ─────────────
ALL_JOINTS = [
    "left_hip_pitch_joint",  "left_hip_roll_joint",  "left_hip_yaw_joint",
    "left_knee_joint",       "left_ankle_pitch_joint","left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint",  "right_hip_yaw_joint",
    "right_knee_joint",      "right_ankle_pitch_joint","right_ankle_roll_joint",
    "waist_yaw_joint",       "waist_roll_joint",      "waist_pitch_joint",
    "left_shoulder_pitch_joint",  "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",    "left_elbow_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",   "right_elbow_joint",
]

# ─── Scene ─────────────────────────────────────────────────────────────────────
_G1_FIXED = G1_LOCK_WRIST_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
_G1_FIXED.spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg(
    disable_gravity=True, retain_accelerations=False,
    linear_damping=0.0, angular_damping=0.0,
    max_linear_velocity=1000.0, max_angular_velocity=1000.0,
    max_depenetration_velocity=1.0,
)
_G1_FIXED.spawn.articulation_props = sim_utils.ArticulationRootPropertiesCfg(
    fix_root_link=False,   # free root so we can set arbitrary height/orientation
    enabled_self_collisions=False,
    solver_position_iteration_count=8,
    solver_velocity_iteration_count=4,
)


@configclass
class VisSceneCfg(InteractiveSceneCfg):
    dome_light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.9, 0.9, 0.9)),
    )
    ground: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    robot: ArticulationCfg = _G1_FIXED


# ─── Marker helpers ────────────────────────────────────────────────────────────

def _sphere_marker_cfg(color, prim_path) -> VisualizationMarkersCfg:
    return VisualizationMarkersCfg(
        prim_path=prim_path,
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=SPHERE_RADIUS,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            )
        },
    )


def _arrow_marker_cfg(color, prim_path) -> VisualizationMarkersCfg:
    from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG
    template = BLUE_ARROW_X_MARKER_CFG.replace(prim_path=prim_path)
    template.markers["arrow"].visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=color)
    template.markers["arrow"].scale = (0.01, 0.01, 0.01)
    return template


# ─── HDF5 loader ───────────────────────────────────────────────────────────────

def _load_results(path: str):
    with h5py.File(path, "r") as f:
        N = int(f.attrs["n_samples"])
        data = {
            # solutions
            "h":           f["solutions/h"][:],           # (N,)
            "rp":          f["solutions/rp"][:],           # (N, 2)
            "q23":         f["solutions/q23"][:],          # (N, 23)
            "foot_forces": f["solutions/foot_forces"][:],  # (N, 2, 3)
            # problems
            "ee_pos_l":      f["problems/ee_pos_l"][:],      # (N, 3)
            "ee_pos_r":      f["problems/ee_pos_r"][:],      # (N, 3)
            "F_left_world":  f["problems/F_left_world"][:],  # (N, 3)
            "F_right_world": f["problems/F_right_world"][:], # (N, 3)
            "h_target":      f["problems/h_target"][:],      # (N,)
            "rp_target":     f["problems/rp_target"][:],     # (N, 2)
            "only_straight": f["problems/only_straight"][:], # (N,)
            "source_idx":    f["problems/source_idx"][:],    # (N,)
            # losses
            "losses": {k: f[f"losses/{k}"][:] for k in f["losses"].keys()},
        }
    print(f"[INFO] Loaded {N} solutions from '{path}'")
    return data, N


# ─── Rotation helper ───────────────────────────────────────────────────────────

def _rp_to_quat_wxyz(roll: float, pitch: float) -> np.ndarray:
    """Roll-pitch (yaw=0) to quaternion [w,x,y,z] for Isaac Lab."""
    roll_t  = torch.tensor([roll],  dtype=torch.float32)
    pitch_t = torch.tensor([pitch], dtype=torch.float32)
    yaw_t   = torch.tensor([0.0],   dtype=torch.float32)
    # quat_from_euler_xyz returns (x,y,z,w); reorder to (w,x,y,z).
    q_xyzw = quat_from_euler_xyz(roll_t, pitch_t, yaw_t)[0]   # (4,) xyzw
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float32)


def _rp_to_rot3x3(roll: float, pitch: float) -> np.ndarray:
    """Roll-pitch (yaw=0) rotation matrix (3×3)."""
    cr, sr = math.cos(roll),  math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    return np.array([
        [cp,  sp*sr,  sp*cr],
        [0.,  cr,    -sr   ],
        [-sp, cp*sr,  cp*cr],
    ])


# ─── Keyboard controller ───────────────────────────────────────────────────────

class KeyboardController:
    def __init__(self, robot, device, data, N,
                 joint_ids, env_origin,
                 default_joint_pos,
                 markers):
        self._robot = robot
        self._device = device
        self._data = data
        self._N = N
        self._joint_ids = joint_ids
        self._env_origin = env_origin
        self._default_joint_pos = default_joint_pos
        self._markers = markers
        self._cursor = -1
        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._sub = self._input.subscribe_to_keyboard_events(
            self._keyboard, self._on_keyboard_event
        )
        print("[INFO] Press K to cycle through optimisation solutions.")

    def _on_keyboard_event(self, event):
        if event.type != carb.input.KeyboardEventType.KEY_PRESS:
            return
        if event.input.name.upper() == "K":
            self._advance()

    def _advance(self):
        self._cursor = (self._cursor + 1) % self._N
        idx = self._cursor
        self._apply(idx)
        self._print_losses(idx)

    def _apply(self, idx: int):
        d   = self._data
        h   = float(d["h"][idx])
        rp  = d["rp"][idx].astype(float)      # (2,)
        q23 = d["q23"][idx].astype(np.float32)  # (23,)

        # Root pose: position [env_origin + (0, 0, h)], orientation from roll/pitch.
        root_pos  = self._env_origin.cpu().numpy() + np.array([0., 0., h], dtype=np.float32)
        root_quat = _rp_to_quat_wxyz(float(rp[0]), float(rp[1]))  # wxyz

        root_state = self._robot.data.default_root_state.clone()
        root_state[0, :3]  = torch.as_tensor(root_pos, dtype=torch.float32)
        root_state[0, 3:7] = torch.as_tensor(root_quat, dtype=torch.float32)
        root_state[0, 7:]  = 0.0
        self._robot.write_root_pose_to_sim(root_state[:, :7])
        self._robot.write_root_velocity_to_sim(root_state[:, 7:])

        # Joint positions.
        joint_pos = self._default_joint_pos.clone()
        joint_pos[0, self._joint_ids] = torch.as_tensor(
            q23, device=self._device, dtype=torch.float32
        )
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel)
        self._robot.set_joint_position_target(joint_pos)

        # Update force / target markers.
        self._update_markers(idx, h, rp)

    def _update_markers(self, idx: int, h: float, rp: np.ndarray):
        d    = self._data
        R    = _rp_to_rot3x3(float(rp[0]), float(rp[1]))
        base = self._env_origin.cpu().numpy()

        # Bake YAG-frame targets: T(h, rp) @ ee_pos_base + [0,0,h]
        tgt_l = R @ d["ee_pos_l"][idx].astype(float) + np.array([0., 0., h])
        tgt_r = R @ d["ee_pos_r"][idx].astype(float) + np.array([0., 0., h])
        tgt_l_w = (base + tgt_l).astype(np.float32)
        tgt_r_w = (base + tgt_r).astype(np.float32)

        self._markers["tgt_l"].visualize(
            translations=torch.as_tensor(tgt_l_w).unsqueeze(0))
        self._markers["tgt_r"].visualize(
            translations=torch.as_tensor(tgt_r_w).unsqueeze(0))

        # Force arrows: world-frame foot and hand forces.
        # We compute arrow start positions from FK positions stored in q23 at this solution.
        # For simplicity we draw the arrows at the approximate world positions derived from
        # known base-frame foot / hand nominal offsets (visual only).
        # Foot approximate world positions: R @ foot_base + [0,0,h]
        _FOOT_L_NOM = np.array([ 0., 0.11851, -0.75686])  # left ankle in base frame at q=0
        _FOOT_R_NOM = np.array([ 0.,-0.11851, -0.75686])
        _HAND_L_NOM = np.array([0.24127, 0.15165, 0.09523])  # left hand in base frame at q=0
        _HAND_R_NOM = np.array([0.24127,-0.15165, 0.09523])

        foot_l_w = base + R @ _FOOT_L_NOM + np.array([0., 0., h])
        foot_r_w = base + R @ _FOOT_R_NOM + np.array([0., 0., h])
        hand_l_w = base + R @ _HAND_L_NOM + np.array([0., 0., h])
        hand_r_w = base + R @ _HAND_R_NOM + np.array([0., 0., h])

        ff   = d["foot_forces"][idx].astype(float)   # (2, 3)
        f_lh = d["F_left_world"][idx].astype(float)
        f_rh = d["F_right_world"][idx].astype(float)

        def _arrow_pts(start, force):
            """Return (2, 3) start/end as float32 tensor row."""
            end = start + force * FORCE_SCALE
            return np.stack([start, end], axis=0).astype(np.float32)

        # Use marker translations pairs to approximate arrows via 2-point lines.
        # Isaac Lab VisualizationMarkers don't draw lines, so we show a small sphere
        # at the tip of each force arrow instead (sufficient for visual inspection).
        for key, pos, force, marker_key in [
            ("foot_l_tip", foot_l_w, ff[0],  "foot_l"),
            ("foot_r_tip", foot_r_w, ff[1],  "foot_r"),
            ("hand_l_tip", hand_l_w, f_lh,   "hand_l"),
            ("hand_r_tip", hand_r_w, f_rh,   "hand_r"),
        ]:
            tip = pos + force * FORCE_SCALE
            if marker_key in self._markers:
                self._markers[marker_key].visualize(
                    translations=torch.as_tensor(tip.astype(np.float32)).unsqueeze(0))

    def _print_losses(self, idx: int):
        d = self._data
        print(f"\n── Solution {idx+1}/{self._N} "
              f"(source HDF5 idx {d['source_idx'][idx]}) ──────────────────")
        print(f"  h   = {d['h'][idx]:.4f} m")
        print(f"  rp  = {np.round(d['rp'][idx], 4)} rad  (roll, pitch)")
        print(f"  only_straight = {bool(d['only_straight'][idx])}")
        print(f"  F_left_world  = {np.round(d['F_left_world'][idx], 2)} N")
        print(f"  F_right_world = {np.round(d['F_right_world'][idx], 2)} N")
        print(f"  foot_forces[L]= {np.round(d['foot_forces'][idx, 0], 2)} N")
        print(f"  foot_forces[R]= {np.round(d['foot_forces'][idx, 1], 2)} N")
        losses = d["losses"]
        print(f"  ── Losses ──")
        print(f"    total      = {losses['total'][idx]:.4e}")
        print(f"    cost       = {losses['cost'][idx]:.4e}")
        print(f"    penalty    = {losses['penalty'][idx]:.4e}")
        print(f"  ── Cost terms ──")
        for k in ("tau", "F_feet", "height", "upright", "hip_yaw", "shoulder_roll"):
            print(f"    {k:16s} = {losses[k][idx]:.4e}")
        print(f"  ── Penalty terms ──")
        for k in ("equilibrium", "arm", "arm_ori", "head_behind",
                  "foot_z", "ankle_z", "symmetry", "box",
                  "qlim", "tlim", "rp", "h", "friction"):
            print(f"    {k:16s} = {losses[k][idx]:.4e}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    device = args_cli.device if args_cli.device is not None else "cuda:0"

    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[2.5, 2.5, 2.0], target=[0.0, 0.0, 1.0])

    scene_cfg = VisSceneCfg(num_envs=1, env_spacing=5.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    robot = scene["robot"]
    sim_dt = sim.get_physics_dt()
    env_origin = scene.env_origins[0].to(device)

    # Map joint names → Isaac Lab joint indices.
    joint_ids = torch.tensor(
        robot.find_joints(ALL_JOINTS, preserve_order=True)[0],
        dtype=torch.long, device=device,
    )
    default_joint_pos = robot.data.default_joint_pos.clone()

    # ── Load results ─────────────────────────────────────────────────────────
    data, N = _load_results(args_cli.results)
    if N == 0:
        print("[WARN] No solutions found in the results file. Exiting.")
        simulation_app.close()
        return

    # ── Markers ──────────────────────────────────────────────────────────────
    markers = {
        "tgt_l":   VisualizationMarkers(_sphere_marker_cfg((1.0, 0.2, 0.2), "/Visuals/TgtL")),
        "tgt_r":   VisualizationMarkers(_sphere_marker_cfg((0.2, 0.2, 1.0), "/Visuals/TgtR")),
        # Force tip spheres: feet (green), hands (orange).
        "foot_l":  VisualizationMarkers(_sphere_marker_cfg((0.2, 0.9, 0.2), "/Visuals/FtL")),
        "foot_r":  VisualizationMarkers(_sphere_marker_cfg((0.2, 0.9, 0.2), "/Visuals/FtR")),
        "hand_l":  VisualizationMarkers(_sphere_marker_cfg((1.0, 0.6, 0.1), "/Visuals/HnL")),
        "hand_r":  VisualizationMarkers(_sphere_marker_cfg((1.0, 0.6, 0.1), "/Visuals/HnR")),
    }

    kb = KeyboardController(
        robot=robot, device=device, data=data, N=N,
        joint_ids=joint_ids, env_origin=env_origin,
        default_joint_pos=default_joint_pos,
        markers=markers,
    )

    # Initial sim step.
    scene.write_data_to_sim()
    sim.step(render=False)
    scene.update(sim_dt)
    print(f"[INFO] Scene ready. {N} solutions loaded. Press K to advance.")

    # ── Sim loop ──────────────────────────────────────────────────────────────
    while simulation_app.is_running():
        robot.set_joint_position_target(robot.data.joint_pos)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)


if __name__ == "__main__":
    main()
    simulation_app.close()
