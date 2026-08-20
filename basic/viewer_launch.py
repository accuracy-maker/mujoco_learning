import mujoco
import numpy as np
import mujoco.viewer

# load model/scene from xml file
scene_path = "/home/z5506409/mujoco_menagerie/franka_emika_panda/scene.xml"
m = mujoco.MjModel.from_xml_path(scene_path)
d = mujoco.MjData(m)

print(m)
print(d)

qpos_view = d.qpos
print(qpos_view)

# view lauch
mujoco.viewer.launch(m, d)