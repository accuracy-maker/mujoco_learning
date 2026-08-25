"""
this file implements a 6DoF arm (UR5)'s IK solver given a target which can be moved in GUI

    1. target is movable in GUI
    2. differential IK
"""

import mujoco
import mujoco.viewer
import time
import numpy
from dataclasses import dataclass
import numpy as np

# simulation setup
@dataclass
class SimulationConfig:
    integration_dt: float = 1.0
    damping: float = 1e-4
    gravity_compensation: bool = True
    dt: float = 0.002
    max_angvel: float = 0.0 # disable



# load scene
xml_path = 'basic/scenes/universal_robots_ur5e/scene.xml'

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

print(f'load model {model} and data {data}')

model.opt.timestep = SimulationConfig.dt

print(f'model opt timestep is: {model.opt.timestep}')

# extract the end-effector information
site_id = model.site("attachment_site").id

print(f'site id is: {site_id}')

#  gravity compensation
body_names = [
    "shoulder_link",
    "upper_arm_link",
    "forearm_link",
    "wrist_1_link",
    "wrist_2_link",
    "wrist_3_link",
]
body_ids = [model.body(name).id for name in body_names]

for name, ids in zip(body_names, body_ids):
    print(f"name: {name} | id: {ids}")

if SimulationConfig.gravity_compensation:
        model.body_gravcomp[body_ids] = 1.0

# read dof and actuators
joint_names = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow",
        "wrist_1",
        "wrist_2",
        "wrist_3",
    ]
dof_ids = np.array([model.joint(name).id for name in joint_names])
# Note that actuator names are the same as joint names in this case.
actuator_ids = np.array([model.actuator(name).id for name in joint_names])

# read home configuration
key_id = model.key("home").id

# read mocap
mocap_id = model.body("target").mocapid[0]

# Pre-allocate numpy arrays.
jac = np.zeros((6, model.nv))
diag = SimulationConfig.damping * np.eye(6)
error = np.zeros(6)
error_pos = error[:3]
error_ori = error[3:]
site_quat = np.zeros(4)
site_quat_conj = np.zeros(4)
error_quat = np.zeros(4)

# Define a trajectory for the end-effector site to follow.
def circle(t: float, r: float, h: float, k: float, f: float) -> np.ndarray:
    """Return the (x, y) coordinates of a circle with radius r centered at (h, k)
    as a function of time t and frequency f."""
    x = r * np.cos(2 * np.pi * f * t) + h
    y = r * np.sin(2 * np.pi * f * t) + k
    return np.array([x, y])

# simulation viwer
with mujoco.viewer.launch_passive(
     model=model, data=data, show_left_ui=False, show_right_ui=False
) as viewer:
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mjv_defaultFreeCamera(model, viewer.cam)
    viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

    while viewer.is_running():
        step_start = time.time()
        # data.mocap_pos[mocap_id, 0:2] = circle(data.time, 0.1, 0.5, 0.0, 0.5) # position (x, y)

        # position error
        error_pos[:] = data.mocap_pos[mocap_id] - data.site(site_id).xpos

        # orientation error
        mujoco.mju_mat2Quat(site_quat, data.site(site_id).xmat)
        mujoco.mju_negQuat(site_quat_conj, site_quat)
        mujoco.mju_mulQuat(error_quat, data.mocap_quat[mocap_id], site_quat_conj)
        mujoco.mju_quat2Vel(error_ori, error_quat, 1.0)

        # jacobian matrix
        mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)

        # damped pseduoinverse
        dq = jac.T @ np.linalg.solve(jac @ jac.T + diag, error)

        # Scale down joint velocities if they exceed maximum.
        if SimulationConfig.max_angvel > 0:
            dq_abs_max = np.abs(dq).max()
            if dq_abs_max > SimulationConfig.max_angvel:
                dq *= SimulationConfig.max_angvel / dq_abs_max

        # integrate dq
        q = data.qpos.copy()
        mujoco.mj_integratePos(model, q, dq, SimulationConfig.integration_dt)

        # control signal
        np.clip(q, *model.jnt_range.T, out=q)
        data.ctrl[actuator_ids] = q[dof_ids]

        # step
        mujoco.mj_step(model, data)

        viewer.sync()
        time_until_next_step = SimulationConfig.dt - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
