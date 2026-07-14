from robot_control import RobotOperation
import rospy
import tf
import numpy as np
from transform_utils import convert_pose_quat2mat, convert_pose_quat2euler, \
    convert_pose_mat2quat, convert_pose_quat2euler, convert_pose_euler2quat
import time


rospy.init_node("UR10_Robot_Gripper_Publisher")
tf_listener = tf.TransformListener()
rospy.sleep(1)
(trans, rot) = tf_listener.lookupTransform('/tool0_controller', '/tool0', rospy.Time(0))
Ttool2tcp = np.array([trans[0], trans[1], trans[2], rot[0], rot[1], rot[2], rot[3]])
Ttool2tcp = convert_pose_quat2mat(Ttool2tcp)
robotoperation = RobotOperation(Ttool2tcp)

robotoperation.UR10_moveto_pose_rtde([[-0.3241732,   0.77846563,  0.52747114, -0.71660192, -0.009239,    0.00897958, 0.69736339]])

type = "angle"
if type == "pose":
    # robotoperation.UR10_moveto_pose_rtde([[-0.3241732,   0.77846563,  0.52747114, -0.71660192, -0.009239,    0.00897958, 0.69736339]])
    # robotoperation.UR10_moveto_pose_rtde([[-0.33765788,  0.73102404,  0.13788645, -0.97499137, -0.03540557,  0.03019274, 0.21731697]])
    all_pose = np.loadtxt("/home/k202/0616_zhucong/000001/ur10_ee_pose.txt")[::5, 1:]
    start_time = time.time()
    for index, single_pose in enumerate(all_pose):
        print(index)
        robotoperation.UR10_moveto_pose_rtde([single_pose])

if type == "angle":
    robotoperation.UR10_moveto_pose_rtde([[-0.3241732,   0.77846563,  0.52747114, -0.71660192, -0.009239,    0.00897958, 0.69736339]])
    # robotoperation.UR10_moveto_pose_rtde([[-0.33765788,  0.73102404,  0.13788645, -0.97499137, -0.03540557,  0.03019274, 0.21731697]])
    all_pose = np.loadtxt("/home/k202/0616_zhucong/000001/ur10_angle.txt")[::5, 1:]
    start_time = time.time()
    for index, single_pose in enumerate(all_pose):
        robotoperation.UR10_moveto_angle_rtde(single_pose)
