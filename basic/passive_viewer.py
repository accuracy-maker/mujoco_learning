import time
from dataclasses import dataclass

import mujoco
import mujoco.viewer
import numpy as np

@dataclass
class UiState:
    # lightweight modification by keyboard
    paused: bool = False
    reset_requested: bool = False
    show_contact: bool = False
    kp_scale: float = 1.0
    step_once: bool = False

def make_key_callback(ui: UiState):
    # key is encoded as ASCII or chr
    def key_callback(keycode: int) -> None:
        try:
            key = chr(keycode).lower()
        except ValueError:
            return

        if key == " ":
            ui.paused = not ui.paused
        elif key == "r":
            ui.reset_requested = True
        elif key == "d":
            ui.show_contact = not ui.show_contact
        elif key == "]":
            ui.kp_scale *= 1.1
        elif key == "[":
            ui.kp_scale /= 1.1
        elif key == ".":
            ui.step_once = True

    return key_callback

def actuated_joint_state(model: mujoco.MjModel,
                         data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:

    """
    type          DoF          qpos entries     meaning
    mjJNT_FREE     6          7(3pos + 4quat)    free-floating body
    mjJNT_BALL     3                4 quat       ball/spherical joint
    mjJNT_SLIDE    1                 1               prismatic
    mjJNT_HINGE    1                 1               revolute
    """

    q = np.zeros(model.nu)
    v = np.zeros(model.nu)

    for i in range(model.nu):
        joint_id = int(model.actuator_trnid[i,0])
        if joint_id < 0:
            raise ValueError(f"actuator {i} is not attached to a joint")

        joint_type = model.jnt_type[joint_id]

        if joint_type not in (mujoco.mjtJoint.mjJNT_HINGE,
                              mujoco.mjtJoint.mjJNT_SLIDE):
            raise ValueError("this minimal PD example only support hinge/slide joint actuators")

        # adr means "address"
        qadr = model.jnt_qposadr[joint_id]
        dadr = model.jnt_dofadr[joint_id]
        q[i] = data.qpos[qadr]
        v[i] = data.qvel[dadr]

    return q, v

def joint_pd(model: mujoco.MjModel,
             data: mujoco.MjData,
             q_des: np.ndarray,
             kp: float,
             kd: float) -> np.ndarray:
    """minimal PD, output ctrl corresponding to actuator"""

    q, v = actuated_joint_state(model, data)
    tau = kp*(q_des - q) - kd * v
    return np.clip(tau, -20.0, 20.0)

def run_interactive_pd(xml_path: str) -> None:
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    ui = UiState()
    q_home, _ = actuated_joint_state(model, data)

    key_callback = make_key_callback(ui)

    with mujoco.viewer.launch_passive(
        model,
        data,
        key_callback=key_callback,
        show_left_ui=True,
        show_right_ui=True,
    ) as viewer:

        while viewer.is_running():
            tic = time.perf_counter()

            if ui.reset_requested:
                mujoco.mj_resetData(model, data)
                ui.reset_requested = False

            should_step = (not ui.paused) or ui.step_once

            if should_step:
                kp = 30.0 * ui.kp_scale
                kd = 2.0 * np.sqrt(kp)
                data_ctrl[:] = joint_pd(model, data, q_home, kp, kd)
                mujoco.mj_step(model, data)
                ui.step_once = False

            with viewer.lock():
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(ui.show_contact)

            viewer.sync()

            sleep_time = model.opt.timestep - (time.perf_counter() - tic)

            if sleep_time > 0.0:
                time.sleep(sleep_time)
            

