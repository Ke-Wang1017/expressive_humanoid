import mujoco
import numpy as np

path = "../data/assets/mjcf/stompypro/robot.xml"
m = mujoco.MjModel.from_xml_path(path)
# m.opt.solver = mujoco.mjtSolver.mjSOL_CG
# m.opt.iterations = 6
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
# print('The result transformation', new_quat)
breakpoint()

# print("model body pos", data.xpos)
# breakpoint()
# print("model body orientation", m.body_quat)
# new_data = m.body_quat[:,[1,2,3,0]]

# breakpoint()
# mujoco.mju_copy(np.array(m.key_qpos.T), np.expand_dims(data.qpos, axis=1))
# mujoco.mju_copy(m.key_qvel.T, np.expand_dims(data.qvel, axis=1))
# # mujoco.mju_copy(m.key_act.T, data.act, )
# # mujoco.mju_copy(m.key_mpos.T, data.mocap_pos.T)
# # mujoco.mju_copy(m.key_mquat.T, data.mocap_quat.T)
# # breakpoint()
# mujoco.mju_copy(m.key_ctrl.T, np.expand_dims(data.ctrl, axis=1))
# mujoco.mj_step(m, data)
# print("model body pos", m.body_pos)
# breakpoint()
# mujoco.mj_saveLastXML("new_model.xml", m)