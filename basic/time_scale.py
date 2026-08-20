"""
In a simulator, there are three types of time:
    1. integration time step: \delta t_{sim}, usually 0.5-2ms, model.opt.timestep
    2. control time step: \delta t_{ctrl}, usually 1-10ms, update data,ctrl
    3. rendering time step: delta t_{render} usually 16-33ms rendering simulation: (30-60Hz)

Following code: 1kHz simulation, 500Hz PD, 100Hz update, 60Hz rendering
"""

import time
from dataclasses import dataclass

import mujoco
import mujoco.viewer
import numpy as np

from passive_viewer import actuated_joint_state

@dataclass
class RateDriver:

    # sim_dt should be good for being divided by other freqs
    sim_dt: float
    ctrl_dt: float
    slow_dt: float
    render_dt: float
    step_count: int = 0
    last_render_time: float = 0.0

    @property
    def ctrl_div(self) -> int:
        return max(1, round(self.ctrl_dt / self.sim_dt))

    @property
    def slow_div(self) -> int:
        return max(1, round(self.slow_dt / self.sim_dt))

    def do_ctrl(self) -> bool:
        return self.step_count % self.ctrl_div == 0

    def do_slow(self) -> bool:
        return self.step_count % self.slow_div == 0

    def do_render(self, now: float) -> bool:
        return now - self.last_render_time >= self.render_dt

def run_multirate_loop(xml_path: str) -> None:

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    sim_dt = model.opt.timestep

    # ctrl: 500Hz slow: 100Hz render: 60Hz
    rate = RateDriver(sim_dt=sim_dt, ctrl_dt=0.002, slow_dt=0.010, render_dt=1.0/60.0)

    q_target, _ = actuated_joint_state(model, data)
    tau_cmd = np.zeros(model.nu)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        wall_start = time.perf_counter()
        sim_start = data.time

        while viewer.is_running():
            loop_start = time.perf_counter()

            if rate.do_slow():
                # slow level: update model, MPC, trajactor generation
                t = data.time
                if model.nu > 0:
                    q_target, _ = actuated_joint_state(model, data)
                    q_target[0] += 0.2 * np.sin(2.0 * np.pi * 0.2 * t)

            if rate.do_ctrl():
                # fast level: update force, torque
                q, v = actuated_joint_state(model, data)
                kp = 40.0
                kd = 2.0 * np.sqrt(kp)
                tau_cmd = kp * (q_target - q) - kd * v
                tau_cmd = np.clip(tau_cmd, -30.0, 30.0)

            data.ctrl[:] = tau_cmd
            mujoco.mj_step(model,data)
            rate.step_count += 1

            now = time.perf_counter()
            if rate.do_render(now):
                viewer.sync()
                rate.last_render_time = now

            # match sim_t with wall_t avoiding system drift
            sim_elapsed = data.time - sim_start
            wall_elapsed = time.perf_counter - wall_start
            sleep_time = sim_elapsed - wall_elapsed
            if sleep_time > 0.0:
                time.sleep(sleep_time)

