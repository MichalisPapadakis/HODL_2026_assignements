"""Shared configuration for pose_optim.py and pose_optim_torch.py.

Import with:
    from pose_optim_config import *
or explicitly import the names you need.

Batch-execution settings (FORCE_RANGE_*, NUM_PTS, …) are only used by
pose_optim_torch.py in batch_execution mode.
"""

from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent

# ============================================================
# Robot model
# ============================================================
URDF_PATH = SCRIPT_DIR / "g1_29dof_rev_1_0.urdf"

# ============================================================
# User inputs
# ============================================================
# Hand targets in base/pelvis frame [m].
POS_LEFT_ARM  = np.array([0.22959137,  0.10266018, 0.05003488])
POS_RIGHT_ARM = np.array([0.22959137, -0.10266018, 0.05003488])

# User-defined transform: bakes base-frame targets into fixed YAG-frame targets.
H_USER  = 0.76                  # nominal base height [m]
RP_USER = np.array([0., 0.])    # (roll, pitch) [rad]

# Horizontal forces applied ON the robot at each hand [N]  (Fz = 0).
F_LEFT  = np.array([-40.0, -30.0])   # (Fx, Fy)
F_RIGHT = np.array([-40.0,   0.0])   # (Fx, Fy)

# ============================================================
# Objective weights
# ============================================================
W1 = 2.5e-2         # ||tau||          (joint torques)
W2 = 1.0e-3         # ||F_feet||       (foot reaction forces)
W3 = 1.0e-3         # (h - H0)^2       (height regularization)
W4 = [1.0e2, 1.0e-2]   # ||rp||^2     (roll, pitch; keep pelvis upright)
W5 = 1.0e-2         # ||hip_yaw||^2    (left/right hip_yaw near 0)
W6 = 1.0e+1         # one-sided shoulder roll (left < 0, right > 0)

# Constraint penalty weights.
LAM_EQ    = 2.0e+3  # base static equilibrium  (tau_base == 0)
LAM_FOOTZ = 2.0e+3  # foot on floor (world z == 0)
LAM_ARM   = 5.0e+2  # hand FK == target         (YAG frame)
LAM_BOX   = 5.0e+2  # foot x,y inside the box
LAM_QLIM  = 1.0e+3  # joint position limits
LAM_TLIM  = 1.0e+3  # joint torque limits (URDF effort)
LAM_RP    = 1.0e+3  # pelvis roll/pitch bounds
LAM_H     = 1.0e+3  # pelvis height bounds
LAM_FRIC  = 1.0e+2  # Coulomb friction pyramid

# Regularisation
LAM_ANKLE   = 1.0e+2  # ankle z-axis upright in world frame
LAM_ARM_ORI = 5.0e+1  # hand orientation RPY in base frame
LAM_HEAD    = 1.0e+2  # head behind arms (head x < arm target x)
LAM_SYM     = 5.0e-2  # left/right foot symmetry

# ============================================================
# Geometry / physics
# ============================================================
H0 = 0.775          # nominal pelvis height above the feet [m]
Z_FEET_0 = 0.035    # foot origin z below the ankle frame origin [m]
H_MIN, H_MAX = 0.59, 0.775   # pelvis height bounds [m]

DEG = np.pi / 180.0
RPY_LOWER = np.array([-30.0, -5.0]) * DEG   # roll, pitch lower bounds [rad]
RPY_UPPER = np.array([+30.0, +40.0]) * DEG  # roll, pitch upper bounds [rad]

MU = 0.6            # friction coefficient
XL, XU = -0.15, 0.20    # foot x box in pelvis frame [m]
YL, YU =  0.05, 0.25    # foot |y| box (right foot uses the mirror) [m]

# Hand orientation bounds in base frame [rad], extrinsic XYZ (matches filter_arm_feasibility).
# Right hand; left uses mirrored roll/yaw: (-roll, pitch, -yaw) checked against these.
HAND_RPY_LO_R = np.array([-30.0, -80.0, -60.0]) * DEG   # roll, pitch, yaw
HAND_RPY_HI_R = np.array([+30.0, +80.0, +30.0]) * DEG

# ============================================================
# Single-run optimizer settings
# ============================================================
OPT         = "adam"   # "adam" | "SGD" | "LBFGS"
LR          = 5.0e-3
STEPS       = 10000
PRINT_EVERY = 100

# ============================================================
# Visualization (pose_optim.py / single_verification mode)
# ============================================================
VISUALIZE     = True
FORCE_SCALE   = 2.5e-2    # arrow length per Newton [m/N]
TARGET_RADIUS = 0.03

# ============================================================
# Batch execution settings (pose_optim_torch.py only)
# ============================================================

# Path to the filtered feasible list produced by filter_arm_feasibility.py.
FEASIBLE_LIST_PATH = SCRIPT_DIR / "data" / "feasible_filtered_list.hdf5"

# Output file for batch results.
OUTPUT_H5_PATH = SCRIPT_DIR / "data" / "optim_pose_data.hdf5"

# Path to the optim kinematics artifacts (built by build_kinematics_artifacts_optim.py).
_IK_CTRL_DIR = (
    SCRIPT_DIR.parent.parent.parent
    / "isaac_humanoid_suite"
    / "source"
    / "isaac_humanoid_suite"
    / "isaac_humanoid_suite"
    / "tasks"
    / "manager_based"
    / "humanoid_pushing"
    / "controllers"
    / "ik_controllers"
)
ARTIFACTS_DIR = _IK_CTRL_DIR / "artifacts" / "g1_29dof_optim"

# Batch optimizer.
OPT_BATCH   = "Adam"    # "Adam" | "SGD" (LBFGS does not support true batching)
LR_BATCH    = 5.0e-3
STEPS_BATCH = 800

# Sampling.
NUM_PTS              = 500    # how many arm configs to draw from feasible list
NUM_FORCE_PER_POINT  = 5      # independent force+pose samples per arm config
BATCH_SIZE           = 64     # optimisation batch size (samples solved in parallel)

# Force sampling ranges [N]: uniform in [lo, hi] per hand, per component.
# Format: (lo, hi) applied to each of the 2 hands independently.
FORCE_RANGE_X = (-60.0, -10.0)   # Fx range [N] (pushing forward → negative)
FORCE_RANGE_Y = (-40.0,  40.0)   # Fy range [N]

# Base pose sampling ranges.
H_RANGE  = (H_MIN, H_MAX)                          # height [m]
RP_RANGE = (RPY_LOWER, RPY_UPPER)                  # (lower (2,), upper (2,)) [rad]

# Discard solutions whose total penalty exceeds this threshold.
MAX_PENALTY_DISCARD = 1.0e3
