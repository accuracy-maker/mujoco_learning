"""
Let the robotic arm reach the target pose but avoid the obstacle using nullspace project
"""
import mujoco
import mujoco.viewer

import time
import numpy as np
from dataclasses import dataclass


@dataclass
class SimulationConfig:
    integration_dt: float = 1.0
    damping: float = 1e-4
    gravity_compensation: bool = True
    dt: float = 0.002

    Kpos: float = 0.95
    Kori: float = 0.95

    Kn = np.asarray([10.0, 10.0, 10.0, 10.0, 5.0, 5.0, 5.0])
    max_angvel: float = 0.785

def main() -> None:
    
    cfg = SimulationConfig()
    # load scene
    xml_path = "basic/scenes/franka_emika_panda/scene_obs.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # enable gravity compensation
    model.body_gravcomp[:] = float(cfg.gravity_compensation)
    model.opt.timestep = cfg.dt # opt is short for options for physical setup

    # ee pose
    site_name = "attachment_site"
    site_id = model.site(site_name).id

    # joint and actuator ids
    joint_names = [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7",
    ]

    dof_ids = np.array([model.joint(name).id for name in joint_names])
    actuator_ids = np.array([model.actuator(name).id for name in joint_names])

    # Initial joint config
    key_name = "home"
    key_id = model.key(key_name).id
    q0 = model.key(key_name).qpos

    # mocap
    mocap_name = "target"
    mocap_id = model.body(mocap_name).mocapid[0]

    # obstacle id
    obs_geom_id = model.geom("obstacle_geom").id

    # collision bodies
    arm_bodies = {model.body(f"link{i}").id for i in range(1, 8)}
    link_geoms = np.array([
        g for g in range(model.ngeom)
        if model.geom_bodyid[g] in arm_bodies and model.geom_group[g] == 3
    ])

    d0      = 0.15   # activation distance (m)
    k_obs   = 0.5   # repulsion gain
    distmax = 0.4    # cap on the distance query

    jacp   = np.zeros((3, model.nv))
    fromto = np.zeros(6)


    # Pre-allocate numpy arrays.
    jac = np.zeros((6, model.nv)) # jacobian matrix
    diag = cfg.damping * np.eye(6) # damping
    eye = np.eye(model.nv)
    twist = np.zeros(6) # [w,v]
    site_quat = np.zeros(4)
    site_quat_conj = np.zeros(4)
    error_quat = np.zeros(4)

    with mujoco.viewer.launch_passive(
        model = model,
        data = data,
        show_left_ui = False,
        show_right_ui = False
    ) as viewer:

        # reset the simulation
        mujoco.mj_resetDataKeyframe(model, data, key_id)

        mujoco.mj_forward(model, data)

        # reset the free camera
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # enable the site frame visulization
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

        while viewer.is_running():
            step_start = time.time()

            # obs distance
            dq_obs = np.zeros(model.nv)
            for g in link_geoms:
                d = mujoco.mj_geomDistance(model, data, obs_geom_id, int(g), distmax, fromto)
                if d >= d0:
                    continue                      # fromto untouched when d == distmax

                n = fromto[3:] - fromto[:3]       # obstacle witness -> robot witness
                nn = np.linalg.norm(n)
                if nn < 1e-9:
                    continue
                n /= nn

                mujoco.mj_jac(model, data, jacp, None, fromto[3:], model.geom_bodyid[g])

                d_safe = max(d, 1e-3)             # keeps 1/d finite on penetration
                w = k_obs * (1.0 / d_safe - 1.0 / d0)
                dq_obs += w * (jacp[:, :].T @ n)

            # spatial velocity
            dx = data.mocap_pos[mocap_id] - data.site(site_id).xpos
            twist[:3] = cfg.Kpos * dx / cfg.integration_dt
            mujoco.mju_mat2Quat(site_quat, data.site(site_id).xmat)
            mujoco.mju_negQuat(site_quat_conj, site_quat)
            mujoco.mju_mulQuat(error_quat, data.mocap_quat[mocap_id], site_quat_conj)
            mujoco.mju_quat2Vel(twist[3:], error_quat, 1.0)
            twist[3:] *= cfg.Kori / cfg.integration_dt

            # jacobian
            mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)

            # damped pseudo-inverse, reused for both the task and the projector
            J_pinv = jac.T @ np.linalg.inv(jac @ jac.T + diag)
            N = eye - J_pinv @ jac

            dq_null = cfg.Kn * (q0 - data.qpos[dof_ids]) + dq_obs[dof_ids]
            dq = J_pinv @ twist + N @ dq_null

            # Clamp maximum joint velocity.
            dq_abs_max = np.abs(dq).max()
            if dq_abs_max > cfg.max_angvel:
                dq *= cfg.max_angvel / dq_abs_max

            # Integrate joint velocities to obtain joint positions.
            q = data.qpos.copy()  # Note the copy here is important.
            mujoco.mj_integratePos(model, q, dq, cfg.integration_dt)
            np.clip(q, *model.jnt_range.T, out=q)

            # Set the control signal and step the simulation.
            data.ctrl[actuator_ids] = q[dof_ids]
            mujoco.mj_step(model, data)

            viewer.sync()
            time_until_next_step = cfg.dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()