"""UR10 real-robot deployment for an OpenPI pi0.5 checkpoint.

The OpenPI policy owns the same preprocessing pipeline used during training:
PlugInputs -> Normalize -> model transforms -> model -> Unnormalize -> PlugOutputs.
This script only needs to provide raw observations with the same keys as the
training config expects.
"""

import argparse
import dataclasses
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import rospy
import torch
import tf

OPENPI_ROOT = Path(__file__).resolve().parents[1]
OPENPI_SRC = OPENPI_ROOT / "src"
if str(OPENPI_SRC) not in sys.path:
    sys.path.insert(0, str(OPENPI_SRC))
if str(OPENPI_ROOT) not in sys.path:
    sys.path.insert(0, str(OPENPI_ROOT))

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

from UR10_deploy.gopro_tac_reader import GoproManager, RealsenseRosManager, ForceTorqueManager
from UR10_deploy.robot_control import RobotOperation
from UR10_deploy.transform_utils import convert_pose_quat2mat


@dataclasses.dataclass
class DeployConfig:
    # 在/home/k202/openpi/openpi/src/openpi/training/config.py中指定配置文件路径
    config_name: str = "pi05_lerobot_plug_finetune"
    checkpoint_dir: str = "/home/8TDisk/0704model_pi05_openpi/pi05_lerobot_plug_finetune/plug_pi05_pytorch"
    device: str = "cuda"
    task: str = "Insert the plug into the power strip"
    max_steps: int = 10000          # 执行10000步后停止
    action_chunk_size: int = 1      # 每次执行几步
    num_inference_steps: int = 16
    compile_model: bool = False
    gopro_device_id: int = 6
    force_log_path: str = "/home/k202/openpi/openpi/UR10_deploy/multimodal_records/force_test.txt"
    use_force: bool = False


def parse_args() -> DeployConfig:
    parser = argparse.ArgumentParser(description="Run an OpenPI checkpoint on the UR10.")
    parser.add_argument("--config-name", default=DeployConfig.config_name)
    parser.add_argument("--checkpoint-dir", default=DeployConfig.checkpoint_dir)
    parser.add_argument("--device", default=DeployConfig.device)
    parser.add_argument("--task", default=DeployConfig.task)
    parser.add_argument("--max-steps", type=int, default=DeployConfig.max_steps)
    parser.add_argument("--action-chunk-size", type=int, default=DeployConfig.action_chunk_size)
    parser.add_argument("--num-inference-steps", type=int, default=DeployConfig.num_inference_steps)
    parser.add_argument("--compile-model", action="store_true", default=DeployConfig.compile_model)
    parser.add_argument("--gopro-device-id", type=int, default=DeployConfig.gopro_device_id)
    parser.add_argument("--force-log-path", default=DeployConfig.force_log_path)
    parser.add_argument("--use-force", default=DeployConfig.use_force)
    return DeployConfig(**vars(parser.parse_args()))


def resolve_checkpoint_dir(checkpoint_path: str | Path) -> Path:
    path = Path(checkpoint_path).expanduser().resolve()
    if path.is_file():
        path = path.parent
    if (path / "model.safetensors").exists():
        return path

    candidates = [
        p
        for p in path.rglob("*")
        if p.is_dir() and p.name.isdigit() and (p / "model.safetensors").exists()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No OpenPI checkpoint with model.safetensors found under: {path}"
        )
    return max(candidates, key=lambda p: int(p.name))


class UR10ePolicyRunner:
    def __init__(self, cfg: DeployConfig):
        self.cfg = cfg
        self.max_steps = cfg.max_steps
        self.all_force = []
        self.action_queue = deque()
        self.use_force = cfg.use_force

        print("Initializing UR10e Policy Runner...")
        if cfg.device.startswith("cuda") and not torch.cuda.is_available():
            print("CUDA is not available, falling back to CPU.")
            self.device = "cpu"
        else:
            self.device = cfg.device

        checkpoint_dir = resolve_checkpoint_dir(cfg.checkpoint_dir)
        print(f"OpenPI checkpoint: {checkpoint_dir}")

        # 配置文件
        train_config = _config.get_config(cfg.config_name)
        if not cfg.compile_model and hasattr(train_config.model, "pytorch_compile_mode"):
            train_config = dataclasses.replace(
                train_config,
                model=dataclasses.replace(train_config.model, pytorch_compile_mode=None),
            )

        # 根据readme的案例定义推理模型，这里也会定义归一化方式和反归一化方式
        self.policy = _policy_config.create_trained_policy(
            train_config,
            checkpoint_dir,
            sample_kwargs={"num_steps": cfg.num_inference_steps},
            default_prompt=cfg.task,
            pytorch_device=self.device,
        )
        print("OpenPI Policy Model Ready!")

        # 初始化硬件，包括GOPRO相机，REALSENSE相机，六维力传感器等
        print("Initializing Camera...")
        self.gopro_manager = GoproManager(device_id=cfg.gopro_device_id, width=224, height=224, fps=30)
        self.realsense = RealsenseRosManager()
        # 用执行记得在零点归零六维力数据
        self.force_manager = ForceTorqueManager(topic_name="/landian_wrench", save_dir="", median_window=310)
        self.force_deque_filter = deque(maxlen = 31)
        self.median_window = 31
        self.mean_window = 9

        print("Initializing Robot...")
        tf_listener = tf.TransformListener()
        rospy.sleep(1)
        try:
            (trans, rot) = tf_listener.lookupTransform('/tool0_controller', '/tool0', rospy.Time(0))
        except:
            print("ERROR: TF lookup failed, using identity.")
            exit()
            trans, rot = [0,0,0], [0,0,0,1]
        Ttool2tcp = np.array([trans[0], trans[1], trans[2], rot[0], rot[1], rot[2], rot[3]])
        Ttool2tcp = convert_pose_quat2mat(Ttool2tcp)
        self.robot = RobotOperation(Ttool2tcp)


    def get_observation(self, task_intru):
        angle_state = self.robot.get_joint_angle_rtde()
        if self.use_force:
            print("正在请求六维力数据...")
            force_state = self.force_manager.get_filtered_wrench()  # 当前的实时观测数据
            print("六维力数据：",force_state)
            self.force_deque_filter.append(force_state)
            # 六维力滤波
            raw_buffer = np.array(self.force_deque_filter)
            if raw_buffer.shape[0] != self.median_window:
                extend_num = self.median_window - raw_buffer.shape[0]
                concat_tensor = np.broadcast_to(raw_buffer[0][None], (extend_num, 6))
                raw_buffer = np.concatenate((concat_tensor, raw_buffer), axis=0)
            raw_buffer_filter = self.force_manager.apply_edge_preserving_filter(raw_buffer, self.median_window, self.mean_window)
            force_state = raw_buffer_filter[-1][None]  # [6] -> [1 6]
            self.all_force.append(force_state[0])
            force_log_path = Path(self.cfg.force_log_path)
            force_log_path.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(force_log_path, np.array(self.all_force))

        gopro_image, _ = self.gopro_manager.get_latest_frame()
        realsense_image , _ =self.realsense.get_latest_frame()
        if gopro_image is None or realsense_image is None:
            raise RuntimeError("Camera frame is None; cannot run OpenPI inference.")

        if gopro_image is not None :
            cv2.imshow("GoPro Camera", gopro_image)
            cv2.waitKey(100)
        if realsense_image is not None :
            cv2.imshow("Realsense Camera", realsense_image)
            cv2.waitKey(100)

        gripper_state = self.robot.close_num
        gripper_state = 1 if gripper_state / 100 > 0.25 else 0
        # 拼接成7维向量
        state_np = np.append(angle_state, gripper_state).astype(np.float32)
        gopro_image = cv2.cvtColor(gopro_image, cv2.COLOR_BGR2RGB).astype(np.uint8)
        realsense_image = cv2.cvtColor(realsense_image, cv2.COLOR_BGR2RGB).astype(np.uint8)

        return {
            "observation/images/gopro": gopro_image,
            "observation/images/realsense": realsense_image,
            "observation/state": state_np,
            # "observation/force": force_state.astype(np.float32),
            "prompt": task_intru,
        }

    def run(self, task_intru):
        print(f"Starting policy execution loop for task: {task_intru}")

        step = 0
        while step < self.max_steps:
            if len(self.action_queue) == 0:
                # 获取观测 and 预处理
                raw_obs = self.get_observation(task_intru)
                print(f'raw_obs.keys:{raw_obs.keys()}')

                # 撞到的分支
                if self.use_force:
                    raw_force_coll = raw_obs["observation/force"]
                    if raw_force_coll[0][2] <= 0:
                        print("========================导致抬升的六维力======================")
                        print(raw_force_coll[0][2])
                        bad_pose = self.robot.get_ee_pose(return_quat=True).tolist()
                        self.robot.UR10_moveto_pose_rtde([[bad_pose[0], bad_pose[1], bad_pose[2]+0.03, bad_pose[3], bad_pose[4], bad_pose[5], bad_pose[6]]])
                        self.robot.UR10_moveto_pose_rtde([[bad_pose[0] - 0.03, bad_pose[1], bad_pose[2], bad_pose[3], bad_pose[4], bad_pose[5], bad_pose[6]]])
                        rospy.sleep(0.5)
                        self.action_queue.clear()   # 清空动作列表
                        self.force_deque_filter.clear()     # 清空六维力列表
                        continue

                # OpenPI infer 会执行 Normalize 和 Unnormalize，返回动作已经是原始 action 空间。
                outputs = self.policy.infer(raw_obs)
                actions = np.asarray(outputs["actions"], dtype=np.float32)
                if actions.ndim == 1:
                    actions = actions[None, :]
                if actions.shape[-1] < 7:
                    raise ValueError(f"OpenPI action dim should be at least 7, got {actions.shape}")

                for action in actions[: self.cfg.action_chunk_size]:
                    self.action_queue.append(action[:7])

            action_numpy = np.asarray(self.action_queue.popleft(), dtype=np.float32)
            print("预测关节角", action_numpy)
            pose_angle_numpy = action_numpy[:6]

            raw_gripper_val = float(action_numpy[6])
            print(raw_gripper_val)
            target_gripper_val = raw_gripper_val * 100.0
            gripper_val = max(0.0, min(100.0, target_gripper_val))

            curr_angle = self.robot.get_joint_angle_rtde()  # [6]
            action_angle = curr_angle + pose_angle_numpy    # [6]
            print(f"目标夹爪闭合百分比: {gripper_val:.2f}")
            print("运动到的关节角：", action_angle)

            self.robot.close_gripper_num(gripper_val)
            while not self.robot.rtde_c.isSteady():
                time.sleep(0.002)
            self.robot.UR10_moveto_angle_rtde(action_angle)
            time.sleep(0.02)
            step += 1



if __name__ == "__main__":
    UR10_runner = None
    try:
        rospy.init_node("UR10_PI05")
        cfg = parse_args()
        UR10_runner = UR10ePolicyRunner(cfg)
        UR10_runner.run(task_intru=cfg.task)


    except KeyboardInterrupt:
        print("\n检测到键盘中断 (Ctrl+C)。准备强制终止当前任务...")

    except Exception as e:
        print(f"\n运行时发生异常: {e}")

    finally:
        print("\n开始执行安全退出流程并释放硬件资源...")
        if UR10_runner is not None:
            if hasattr(UR10_runner, 'gopro_manager') and UR10_runner.gopro_manager is not None:
                try:
                    UR10_runner.gopro_manager.release()
                    print("GoPro 相机资源已释放。")
                except Exception as e:
                    print(f"释放 GoPro 时发生异常: {e}")

            if hasattr(UR10_runner, 'tactile_manager') and UR10_runner.tactile_manager is not None:
                try:
                    UR10_runner.tactile_manager.release()
                    print("触觉传感器资源已释放。")
                except Exception as e:
                    print(f"释放触觉传感器时发生异常: {e}")

        cv2.destroyAllWindows()
        print("所有硬件及系统资源清理完毕，进程安全退出。")

#查看端口v4l2-ctl --list-devices
