import mujoco
import numpy as np

path = "../data/assets/mjcf/stompypro/robot.xml"
m = mujoco.MjModel.from_xml_path(path)
data = mujoco.MjData(m)
quat_local = m.body_quat[1:,:].copy()
quat_local = quat_local[:,[1,2,3,0]]
# data.qpos = m.key_qpos # assign with key_qpos
mujoco.mj_kinematics(m, data)
# mujoco.mj_step(m, data)
new_data = data.xquat[1:,[1,2,3,0]]
# breakpoint()
key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, 'default')
mujoco.mj_resetDataKeyframe(m, data, key)
mujoco.mj_kinematics(m, data)

new_rotation = data.xquat[1:,[1,2,3,0]]
print('The result transformation', new_rotation)
breakpoint()
