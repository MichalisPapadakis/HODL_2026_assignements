# This examples shows how to load and move a robot in meshcat.
# Note: this feature requires Meshcat to be installed, this can be done using
# pip install --user meshcat

import os
import sys
from pathlib import Path

import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer

# Load the URDF model.
# mesh_dir must be the package root (parent of the "meshes/" folder referenced in URDF).
SCRIPT_DIR = Path(__file__).resolve().parent
urdf_path = SCRIPT_DIR / "g1_29dof_rev_1_0.urdf"
mesh_dir = SCRIPT_DIR

model, collision_model, visual_model = pin.buildModelsFromUrdf(
    str(urdf_path), str(mesh_dir), pin.JointModelFreeFlyer()
)

# Start a new MeshCat server and client.
# Note: the server can also be started separately using the "meshcat-server" command in
# a terminal:
# this enables the server to remain active after the current script ends.
#
# Option open=True pens the visualizer.
# Note: the visualizer can also be opened seperately by visiting the provided URL.
try:
    viz = MeshcatVisualizer(model, collision_model, visual_model)
    viz.initViewer(open=True)
except ImportError as err:
    print(
        "Error while initializing the viewer. "
        "It seems you should install Python meshcat"
    )
    print(err)
    sys.exit(0)

# Load the robot in the viewer.
viz.loadViewerModel()

# Display a robot configuration.
q0 = pin.neutral(model)
viz.display(q0)
viz.displayVisuals(True)

# Create a convex shape from solo main body
mesh = visual_model.geometryObjects[0].geometry
mesh.buildConvexRepresentation(True)
convex = mesh.convex

# Place the convex object on the scene and display it
if convex is not None:
    placement = pin.SE3.Identity()
    placement.translation[0] = 2.0
    geometry = pin.GeometryObject("convex", 0, placement, convex)
    geometry.meshColor = np.ones(4)
    # Add a PhongMaterial to the convex object
    geometry.overrideMaterial = True
    geometry.meshMaterial = pin.GeometryPhongMaterial()
    geometry.meshMaterial.meshEmissionColor = np.array([1.0, 0.1, 0.1, 1.0])
    geometry.meshMaterial.meshSpecularColor = np.array([0.1, 1.0, 0.1, 1.0])
    geometry.meshMaterial.meshShininess = 0.8
    visual_model.addGeometryObject(geometry)
    # After modifying the visual_model we must rebuild
    # associated data inside the visualizer
    viz.rebuildData()

# Simple motion demo using the loaded G1 model.
q1 = q0.copy()
if model.nq > 7:
    q1[7] = 0.3

data = viz.data
pin.forwardKinematics(model, data, q1)
viz.display(q1)

dt = 0.05
qs = [q1.copy()]
for _ in range(50):
    q = qs[-1].copy()
    q[7] += 0.02
    qs.append(q)

output_video = SCRIPT_DIR / "g1_motion.mp4"
with viz.create_video_ctx(str(output_video)):
    viz.play(qs, dt)
print(f"Saved video to: {output_video}")