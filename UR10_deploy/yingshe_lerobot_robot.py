import tf
import sys
import rospy
import numpy as np
import pandas as pd
import time
sys.path.append("/home/k202/lerobot/src")
sys.path.append("/home/k202/lerobot")
from UR10e_deploy.robot_control import RobotOperation
from UR10e_deploy.transform_utils import convert_pose_quat2mat, convert_pose_euler2quat, convert_pose_euler2mat, \
    convert_pose_mat2quat


df = pd.read_parquet('/home/k202/0604_dklerobotdataset/Insert the plug into the power strip/data/chunk-000/file-000.parquet', engine='pyarrow')
# 统计总轨迹数 (Episode 数量)
total_episodes = df['episode_index'].nunique()
print(f"实际存储的总轨迹数: {total_episodes}")

grouped_data = df.groupby('episode_index')

# 示例：获取第一个 Episode 的 state 和 action
first_ep_index = 20
first_ep_states = grouped_data.get_group(first_ep_index)['observation.state'].to_numpy()
first_ep_actions = grouped_data.get_group(first_ep_index)['action'].to_numpy()
print(f"Episode {first_ep_index} 的 State 数据维度: {first_ep_states.shape}")
print(f"Episode {first_ep_index} 的 Action 数据维度: {first_ep_actions.shape}")

states_matrix = np.vstack(first_ep_states)
actions_matrix = np.vstack(first_ep_actions)


rospy.init_node("UR10_Robot_Gripper_Publisher")
tf_listener = tf.TransformListener()
rospy.sleep(1)
(trans, rot) = tf_listener.lookupTransform('/tool0_controller', '/tool0', rospy.Time(0))
Ttool2tcp = np.array([trans[0], trans[1], trans[2], rot[0], rot[1], rot[2], rot[3]])
Ttool2tcp = convert_pose_quat2mat(Ttool2tcp)
robotoperation = RobotOperation(Ttool2tcp)

check_type = "state"
robotoperation.close_gripper_num(100)
if check_type == "state":
    for i in range(first_ep_states.shape[0]):
        robotoperation.UR10_moveto_angle_rtde(first_ep_states[i][:6])
        print(first_ep_states[i][:6])


# if check_type == "action":
#     robotoperation.UR10_moveto_angle_rtde([first_ep_states[0][:6]])
#     # rospy.sleep(0.5)
#     current_pose = convert_pose_quat2mat(first_ep_states[0][:6])    # T_b_t1
#     for i in range(first_ep_actions.shape[0]):
#         # 实现方式1
#         current_pose = robotoperation.get_ee_pose_rtde(return_quat = False)  # [4 4] T_base_current
#         T_current_next = convert_pose_quat2mat(first_ep_actions[i][:7])
#         T_base_next = current_pose @ T_current_next
#         robotoperation.UR10_moveto_pose_rtde([convert_pose_mat2quat(T_base_next)])


if check_type == "action":
    robotoperation.UR10_moveto_angle_rtde(first_ep_states[0][:6])
    current_pose = robotoperation.get_joint_angle_rtde()    # T_b_t1
    for i in range(first_ep_actions.shape[0]):
        if i == 267:
            break
        # 判断机械臂是否执行完毕
        while not robotoperation.rtde_c.isSteady():
            time.sleep(0.002)
        current_pose = robotoperation.get_joint_angle_rtde()  # [4 4] T_base_current
        angle_move = first_ep_actions[i][:6]
        angle_next = current_pose + angle_move
        robotoperation.UR10_moveto_angle_rtde(angle_next)
        time.sleep(0.02)


if check_type == "action_all":
    current_pose = first_ep_states[0][:6]
    zero = np.array([0., 0., 0., 0., 0., 0.])
    for i in range(266):
        zero += first_ep_actions[i][:6]
    final_delta_angle = zero
    angle_next = current_pose + final_delta_angle
    robotoperation.UR10_moveto_angle_rtde(angle_next)

robotoperation.UR10_moveto_angle_rtde(first_ep_states[266][:6])
