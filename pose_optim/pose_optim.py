"""Static pose optimization for the Unitree G1 (29 DoF) humanoid.

We solve, for a fixed-feet / fixed-hand-target static pose:

    min  w1||tau||^2 + w2||F_feet||^2 + w3(h-h0)^2 + w4||rpy||^2
    s.t. (soft penalties)
         FK_{l/r hand}(q)        == target_{l/r}        (pelvis frame)
         FK_{l/r foot}_z (world) == 0                    (feet on flat floor, h below pelvis)
         FK_foot x,y in box, left/right symmetric        (pelvis frame)
         q in joint limits
         |tau_joints| <= URDF effort limits
         rpy in [roll, pitch, yaw] bounds;  h in [H_MIN, H_MAX]
         Coulomb friction pyramid on each foot force     (world vertical normal)
         static equilibrium of the unactuated base:  tau_base == 0
    where tau = tG(q) - sum_c J_c^T F_c  (feet: decision vars, hands: fixed inputs).

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
# User inputs / configuration (edit values directly, no CLI)
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
URDF_PATH = SCRIPT_DIR / "g1_29dof_rev_1_0.urdf"

# Hand targets, expressed w.r.t. the pelvis (base) frame [m].
POS_LEFT_ARM = np.array([0.4, +0.15, 0.15])
POS_RIGHT_ARM = np.array([0.3, -0.15, 0.05])

# Horizontal forces applied ON the robot at each hand [N]  (Fz = 0).
F_LEFT = np.array([-20.0, -30.0])   # (Fx, Fy)
F_RIGHT = np.array([-20.0, 0.0])  # (Fx, Fy)

# Objective weights (scalars or broadcastable diagonals).
W1 = 1.0e-1   # ||tau||      (joint torques)
W2 = 1.0e-5   # ||F_feet||   (foot reaction forces)
W3 = 1.0e-5   # (h - H0)^2   (height regularization)
W4 = [1.0e2, 1.0e-5, 1.0e2]   # ||rpy||^2    (keep pelvis upright)

# Constraint penalty weights.
LAM_EQ = 5.0e+1     # base static equilibrium  (tau_base == 0)
LAM_ARM = 1.0e+3    # hand FK == target
LAM_FOOTZ = 1.0e+2  # foot on floor (world z == 0)
LAM_SYM = 1e-3    # left/right foot symmetry
LAM_BOX = 1.0e+1    # foot x,y inside the box
LAM_QLIM = 1.0e+2   # joint position limits
LAM_TLIM = 1.0e+2   # joint torque limits (URDF effort)
LAM_RPY = 1.0e+2    # pelvis roll/pitch/yaw bounds
LAM_H = 1.0e+2      # pelvis height bounds
LAM_FRIC = 1.0e+1   # Coulomb friction pyramid

# Geometry / physics.
H0 = 0.70           # nominal pelvis height above the feet [m]
H_MIN, H_MAX = 0.50, 0.75   # pelvis height bounds [m]
DEG = np.pi / 180.0
RPY_LOWER = np.array([-30.0, -5.0, -45.0]) * DEG   # roll, pitch, yaw [rad]
RPY_UPPER = np.array([+30.0, +40.0, +45.0]) * DEG
MU = 0.6            # friction coefficient
XL, XU = -0.15, 0.20    # foot x box in pelvis frame [m]
YL, YU = 0.05, 0.25     # foot |y| box (right foot uses the mirror) [m]

# Optimizer.
OPT = "adam"        # "adam" or "sgd"
LR = 5.0e-3
STEPS = 1500
PRINT_EVERY = 100

# Visualization.
VISUALIZE = True
FORCE_SCALE = 5.0e-2    # arrow length per Newton [m/N]
TARGET_RADIUS = 0.03

# ============================================================
# Model
# ============================================================
model = pin.buildModelFromUrdf(str(URDF_PATH), pin.JointModelFreeFlyer())
data = model.createData()

NV = model.nv          # 29
NJ = NV - 6            # 23 actuated joints
NQJ = model.nq - 7     # 23 joint configuration entries

# Contact frames: order is [left_foot, right_foot, left_hand, right_hand].
FRAME_NAMES = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_rubber_hand",
    "right_rubber_hand",
]
FRAME_IDS = [model.getFrameId(n) for n in FRAME_NAMES]
FOOT_IDS = FRAME_IDS[:2]

Q_LOWER = torch.as_tensor(model.lowerPositionLimit[7:])
Q_UPPER = torch.as_tensor(model.upperPositionLimit[7:])
TAU_LIMIT = torch.as_tensor(model.effortLimit[6:6 + NJ])  # URDF effort [N·m]
RPY_LO = torch.as_tensor(RPY_LOWER)
RPY_HI = torch.as_tensor(RPY_UPPER)
TOTAL_MASS = float(sum(model.inertias[i].mass for i in range(1, model.njoints)))
GRAVITY = float(abs(model.gravity.linear[2]))


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


def _hand_forces_world():
    return np.array([[F_LEFT[0], F_LEFT[1], 0.0],
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

    Inputs: rpy (3), q23 (23), foot_forces (2, 3) in world frame.
    Hand forces are fixed module inputs. Base position does not affect tau.

    Derivatives: d tau/d F = -J_foot^T is analytical and exact. The contact
    forces are constant in the *world* frame, whose exact d tau/d q needs the
    coordinate position Hessian (Pinocchio's static-torque / kinematic-Hessian
    helpers assume joint-local-constant forces, which differs). We therefore
    central-difference Pinocchio's analytical gravity+Jacobian tau function over
    (rpy, q23); in the future native-torch port this block is just autograd.
    """

    @staticmethod
    def forward(ctx, rpy, q23, foot_forces):
        rpy_n, q23_n, ff_n = _np(rpy), _np(q23), _np(foot_forces)
        tau, foot_J = _tau_eval(rpy_n, q23_n, ff_n)
        ctx.rpy, ctx.q23, ctx.ff, ctx.foot_J = rpy_n, q23_n, ff_n, foot_J
        return torch.as_tensor(tau)

    @staticmethod
    def backward(ctx, grad_out):
        g = _np(grad_out)  # (nv,) = dL/dtau

        def s(rpy, q23):
            return g @ _tau_eval(rpy, q23, ctx.ff)[0]

        eps = 1e-6
        grad_rpy = np.zeros(3)
        for k in range(3):
            d = np.zeros(3); d[k] = eps
            grad_rpy[k] = (s(ctx.rpy + d, ctx.q23) - s(ctx.rpy - d, ctx.q23)) / (2 * eps)
        grad_q23 = np.zeros(NQJ)
        for k in range(NQJ):
            d = np.zeros(NQJ); d[k] = eps
            grad_q23[k] = (s(ctx.rpy, ctx.q23 + d) - s(ctx.rpy, ctx.q23 - d)) / (2 * eps)

        grad_ff = np.stack([-Jk @ g for Jk in ctx.foot_J])  # d tau/dF = -J^T
        return (torch.as_tensor(grad_rpy),
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


COST_KEYS = ("tau", "F_feet", "height", "upright")
PENALTY_KEYS = (
    "equilibrium", "arm", "foot_z", "symmetry", "box",
    "qlim", "tlim", "rpy", "h", "friction",
)


def compute_terms(h, rpy, q23, foot_forces):
    """Return a dict of all (named) scalar loss terms for diagnostics + total."""
    rel = RelFramePos.apply(q23)              # (4, 3) pelvis frame
    tau = StaticTorque.apply(rpy, q23, foot_forces)  # (nv,)

    foot_rel = rel[:2]                        # (2, 3) pelvis frame
    hand_rel = rel[2:]                        # (2, 3) pelvis frame

    R = rpy_to_matrix_torch(rpy)
    foot_world_z = (foot_rel @ R.T)[:, 2] + h  # world z of each foot

    tgt = torch.as_tensor(np.stack([POS_LEFT_ARM, POS_RIGHT_ARM]))

    fx = foot_rel[:, 0]
    fy = foot_rel[:, 1]
    box = (torch.relu(XL - fx) + torch.relu(fx - XU)
           + torch.relu(YL - fy.abs()) + torch.relu(fy.abs() - YU))

    qviol = torch.relu(q23 - Q_UPPER) + torch.relu(Q_LOWER - q23)
    tau_j = tau[6:]
    tviol = torch.relu(tau_j.abs() - TAU_LIMIT)
    rpyviol = torch.relu(rpy - RPY_HI) + torch.relu(RPY_LO - rpy)
    hviol = torch.relu(h - H_MAX) + torch.relu(H_MIN - h)

    fz = foot_forces[:, 2]
    fric = (torch.relu(-fz)
            + torch.relu(foot_forces[:, 0].abs() - MU * fz)
            + torch.relu(foot_forces[:, 1].abs() - MU * fz))

    terms = {
        "tau": W1 * weighted_sq(tau[6:], 1.0),
        "F_feet": W2 * weighted_sq(foot_forces, 1.0),
        "height": W3 * (h - H0) ** 2,
        "upright":  weighted_sq(rpy, W4),
        "equilibrium": LAM_EQ * weighted_sq(tau[:6], 1.0),
        "arm": LAM_ARM * (hand_rel - tgt).pow(2).sum(),
        "foot_z": LAM_FOOTZ * foot_world_z.pow(2).sum(),
        "symmetry": LAM_SYM * ((fx[0] - fx[1]) ** 2 + (fy[0] + fy[1]) ** 2),
        "box": LAM_BOX * box.pow(2).sum(),
        "qlim": LAM_QLIM * qviol.pow(2).sum(),
        "tlim": LAM_TLIM * tviol.pow(2).sum(),
        "rpy": LAM_RPY * rpyviol.pow(2).sum(),
        "h": LAM_H * hviol.pow(2),
        "friction": LAM_FRIC * fric.pow(2).sum(),
    }
    cost_terms = {k: terms[k] for k in COST_KEYS}
    penalty_terms = {k: terms[k] for k in PENALTY_KEYS}
    cost = sum(cost_terms.values())
    penalty = sum(penalty_terms.values())
    total = cost + penalty
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
def initial_guess():
    h = torch.tensor(H0, requires_grad=True)
    rpy = torch.zeros(3, requires_grad=True)
    q23 = torch.zeros(NQJ, requires_grad=True)
    half_weight = TOTAL_MASS * GRAVITY / 2.0
    half_fx = -(F_LEFT[0]+F_RIGHT[0])/2
    half_fy = -(F_LEFT[1]+F_RIGHT[1])/2
    foot_forces = torch.tensor([[half_fx, half_fy, half_weight],
                                [half_fx, half_fy, half_weight]], requires_grad=True)
    return h, rpy, q23, foot_forces


def solve():
    h, rpy, q23, foot_forces = initial_guess()
    params = [h, rpy, q23, foot_forces]
    opt = (torch.optim.Adam(params, lr=LR) if OPT == "adam"
           else torch.optim.SGD(params, lr=LR))

    for step in range(STEPS + 1):
        opt.zero_grad()
        total, cost, penalty, cost_terms, penalty_terms = compute_terms(
            h, rpy, q23, foot_forces
        )
        if step < STEPS:
            total.backward()
            opt.step()
        if step % PRINT_EVERY == 0 or step == STEPS:
            print_loss(step, total, cost, penalty, cost_terms, penalty_terms)

    return h, rpy, q23, foot_forces


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


def visualize(h, rpy, q23, foot_forces):
    import meshcat.geometry as g
    from pinocchio.visualize import MeshcatVisualizer

    _, collision_model, visual_model = pin.buildModelsFromUrdf(
        str(URDF_PATH), str(SCRIPT_DIR), pin.JointModelFreeFlyer()
    )
    viz = MeshcatVisualizer(model, collision_model, visual_model)
    viz.initViewer(open=True)
    viz.loadViewerModel()

    h_n, rpy_n = float(h), _np(rpy)
    q = make_config(h_n, rpy_n, _np(q23))
    viz.display(q)

    R = pin.rpy.rpyToMatrix(rpy_n)
    base = np.array([0.0, 0.0, h_n])

    # Hand target spheres (pelvis -> world).
    for name, tgt, color in [("targets/left", POS_LEFT_ARM, 0xFF3333),
                             ("targets/right", POS_RIGHT_ARM, 0x3333FF)]:
        p_w = R @ tgt + base
        viz.viewer[name].set_object(
            g.Sphere(TARGET_RADIUS), g.MeshLambertMaterial(color=color, opacity=0.6)
        )
        T = np.eye(4)
        T[:3, 3] = p_w
        viz.viewer[name].set_transform(T)

    # Force arrows at the hands (fixed inputs) and feet (solution reactions).
    pin.framesForwardKinematics(model, data, q)
    hand_forces = [np.array([F_LEFT[0], F_LEFT[1], 0.0]),
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
    h, rpy, q23, foot_forces = solve()
    print("\nSolution:")
    print(f"  h   = {float(h):.4f} m")
    print(f"  rpy = {np.round(_np(rpy), 4)} rad")
    print(f"  q23 = {np.round(_np(q23), 3)}")
    print(f"  F_feet =\n{np.round(_np(foot_forces), 2)} N")
    if VISUALIZE:
        visualize(h, rpy, q23, foot_forces)
