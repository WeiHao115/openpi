'''
该文件用于转换数据格式
    __call__函数是主要函数, 其余函数均为其服务
    __call__函数中定义的数据格式以及env_preprocessor函数中将机器人状态转换为pi0.5中输入的state变量[XYZ, 欧拉角, 夹爪参数]
    参考了libero, 若是真实场景采集的数据的话，可能需要进行修改
    __call__函数输入变量包括
        cfg: 配置文件, 通过命令行中指定的parse生成EvalPipelineConfig类配置文件
        task_intru: list[str], 文本指令
        imgs: list, 相机采集的图像(原始图像, 维度为[H W 3]), normal_imgs函数会进行处理(参考libero)
        eef_mat: 末端位姿齐次矩阵
        eef_pos: 末端位置XYZ
        eef_quat: 末端四元数
        gripper_qpos: 夹爪参数，我使用的是闭合程度((0 ~ 255) / 255)
        joints_pose: 关节角
'''

import torch
import cv2
import numpy as np
import einops
from dataclasses import dataclass, field
from collections.abc import Callable
from typing import TypeVar, cast, Generic
from collections.abc import Callable, Sequence

import sys
sys.path.append("/home/ywl/lerobot/src")
from lerobot.processor.core import EnvTransition
from lerobot.processor.converters import batch_to_transition, transition_to_batch
from lerobot.processor.pipeline import ProcessorStep
from lerobot.utils.hub import HubMixin
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.configs import parser

TInput = TypeVar("TInput")
TOutput = TypeVar("TOutput")


# ========输出：observation: **dict_keys(['observation.images.image', 'observation.images.image2', 'observation.state', 'observation.robot_state'])**
#   observation['observation.images.image'].shape     torch.Size([1, 3, 480, 640])
#   observation['observation.images.image2'].shape    torch.Size([1, 3, 480, 640])
#   observation['observation.state'].shape  torch.Size([1, 14])
#       或者observation['observation.robot_state']
#               dict_keys(['eef', 'gripper', 'joints'])
#                   observation['observation.robot_state']['eef']
#                       dict_keys(['mat', 'pos', 'quat']) torch.Size([1, 3, 3]) torch.Size([1, 3]) torch.Size([1, 4])
#                   observation['observation.robot_state']['gripper']
#                       {'qpos': tensor([[ 0.0387, -0.0387]], dtype=torch.float64), 'qvel': tensor([[ 0.0032, -0.0032]], dtype=torch.float64)}
#                   observation['observation.robot_state']['joints']
#                       dict_keys(['pos', 'vel']) torch.Size([1, 7]) torch.Size([1, 7])
# !!!!!! imgs是相机采集的图像，维度为[B H W 3]
@dataclass
class PI05_Data_Trans(HubMixin, Generic[TInput, TOutput]):
    steps: Sequence[ProcessorStep] = field(init=False)
    to_transition: Callable[[TInput], EnvTransition] = field(init=False, repr=False)
    to_output: Callable[[EnvTransition], TOutput] = field(init=False, repr=False)
    before_step_hooks: list[Callable[[int, EnvTransition], None]] = field(init=False, repr=False)
    after_step_hooks: list[Callable[[int, EnvTransition], None]] = field(init=False, repr=False)

    def __init__(self):
        # Copy from /home/ywl/lerobot/src/lerobot/processor/pipeline.py -> class DataProcessorPipeline
        steps: Sequence[ProcessorStep] = []
        self.steps = list(steps)
        self.to_transition = cast(Callable[[TInput], EnvTransition], batch_to_transition)
        self.to_output = cast(Callable[[EnvTransition], TOutput], transition_to_batch)
        self.before_step_hooks = []
        self.after_step_hooks = []

    # Copy from /home/ywl/lerobot/src/lerobot/envs/utils.py -> preprocess_observation函数
    # 对图像进行归一化处理 [B H W C]
    def normal_imgs(self, img):
        if isinstance(img, np.ndarray):
            img_tensor = torch.from_numpy(img)
        else:
            img_tensor = img
        if img_tensor.ndim == 3:
            img_tensor = img_tensor.unsqueeze(0)
        _, h, w, c = img_tensor.shape
        assert c < h and c < w, f"expect channel last images, but instead got {img_tensor.shape=}"

        if img_tensor.dtype != torch.uint8:
            img_tensor = img_tensor.clamp(0, 255).to(torch.uint8)
        # sanity check that images are uint8
        assert img_tensor.dtype == torch.uint8, f"expect torch.uint8, but instead {img_tensor.dtype=}"

        # convert to channel first of type float32 in range [0,1]
        img_tensor = einops.rearrange(img_tensor, "b h w c -> b c h w").contiguous()
        img_tensor = img_tensor.type(torch.float32)
        img_tensor /= 255
        return img_tensor

    def get_preprocessor(self):
        preprocessor_overrides = {
            "device_processor": {"device": str(self.cfg.policy.device)},
            "rename_observations_processor": {"rename_map": self.cfg.rename_map},
        }
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg = self.cfg.policy,
            pretrained_path = self.cfg.policy.pretrained_path,
            preprocessor_overrides = preprocessor_overrides,
        )
        # 定义数据处理流程
        self.steps = preprocessor.steps


    # Copy from /home/ywl/lerobot/src/lerobot/scripts/lerobot_eval.py -> LiberoProcessorStep
    def _quat2axisangle(self, quat: torch.Tensor) -> torch.Tensor:
        """
        Convert batched quaternions to axis-angle format.
        Only accepts torch tensors of shape (B, 4).
        Args:
            quat (Tensor): (B, 4) tensor of quaternions in (x, y, z, w) format
        Returns:
            Tensor: (B, 3) axis-angle vectors
        """
        if not isinstance(quat, torch.Tensor):
            raise TypeError(f"_quat2axisangle expected a torch.Tensor, got {type(quat)}")

        if quat.ndim != 2 or quat.shape[1] != 4:
            raise ValueError(f"_quat2axisangle expected shape (B, 4), got {tuple(quat.shape)}")
        quat = quat.to(dtype=torch.float32)
        device = quat.device
        batch_size = quat.shape[0]
        w = quat[:, 3].clamp(-1.0, 1.0)
        den = torch.sqrt(torch.clamp(1.0 - w * w, min=0.0))
        result = torch.zeros((batch_size, 3), device=device)
        mask = den > 1e-10
        if mask.any():
            angle = 2.0 * torch.acos(w[mask])  # (M,)
            axis = quat[mask, :3] / den[mask].unsqueeze(1)
            result[mask] = axis * angle.unsqueeze(1)
        return result

    # /home/ywl/lerobot/src/lerobot/processor/env_processor.py -> LiberoProcessorStep
    # 根据机器人状态定义输入状态
    def env_preprocessor(self):
        if "observation.robot_state" in self.obs_features:
            robot_state = self.obs_features.pop("observation.robot_state")
            eef_pos = robot_state["eef"]["pos"]  # (B, 3,)
            eef_quat = robot_state["eef"]["quat"]  # (B, 4,)
            gripper_qpos = robot_state["gripper"]["qpos"]  # (B, 2,)
            # Convert quaternion to axis-angle
            eef_axisangle = self._quat2axisangle(eef_quat)  # (B, 3)
            state = torch.cat((eef_pos, eef_axisangle, gripper_qpos), dim=-1)

            # ensure float32
            state = state.float()
            if state.dim() == 1:
                state = state.unsqueeze(0)
            self.obs_features["observation.state"] = state


            # robot_state = self.obs_features.pop("observation.robot_state")
            # eef_pos = robot_state["eef"]["pos"]  # (B, 3,)
            # eef_quat = robot_state["eef"]["quat"]  # (B, 4,)
            # gripper_qpos = robot_state["gripper"]["qpos"]  # (B, 2,)
            # state = torch.cat((eef_pos, eef_quat, gripper_qpos[:, [0]]), dim=-1)

            # # ensure float32
            # state = state.float()
            # if state.dim() == 1:
            #     state = state.unsqueeze(0)
            # self.obs_features["observation.state"] = state


            # robot_state = self.obs_features.pop("observation.robot_state")
            # joint_angle = robot_state["joints"]["pos"]
            # state = torch.from_numpy(joint_angle)
            # # gripper_qpos = robot_state["gripper"]["qpos"][:, [0]]
            # # state = torch.cat((torch.from_numpy(joint_angle[None]), gripper_qpos), dim=-1)
            # state = state.float()
            # if state.dim() == 1:
            #     state = state.unsqueeze(0)
            # self.obs_features["observation.state"] = state



    # Copy from /home/ywl/lerobot/src/lerobot/processor/pipeline.py -> class DataProcessorPipeline
    def _forward(self, transition: EnvTransition) -> EnvTransition:
        """Executes all processing steps and hooks in sequence.
        Args:
            transition: The initial `EnvTransition` object.
        Returns:
            The final `EnvTransition` after all steps have been applied.
        """
        for idx, processor_step in enumerate(self.steps):
            # Execute pre-hooks
            for hook in self.before_step_hooks:
                hook(idx, transition)

            transition = processor_step(transition)

            # Execute post-hooks
            for hook in self.after_step_hooks:
                hook(idx, transition)
        return transition

    # Copy from /home/ywl/lerobot/src/lerobot/processor/pipeline.py -> class DataProcessorPipeline
    # def __call__(self, cfg: EvalPipelineConfig, task_intru: list[str], imgs: list, eef_mat = None,
    #              eef_pos = None, eef_quat = None, gripper_qpos = None,
    #              joints_pose = None):
    #     self.cfg = cfg
    #     self.obs_features = {
    #         "observation.image": self.normal_imgs(imgs[0]),
    #         "observation.images.image2": self.normal_imgs(imgs[1]),
    #         "observation.robot_state":{
    #             "eef":{
    #                 "mat": eef_mat,
    #                 "pos": eef_pos,
    #                 "quat": eef_quat
    #             },
    #             "gripper":{
    #                 "qpos": gripper_qpos
    #             },
    #             "joints":{
    #                 "pos": joints_pose
    #             }
    #         },
    #         "task": task_intru
    #     }
    #     # 将机械臂的状态observation.robot_state转为pi0的输入量observation.state
    #     self.env_preprocessor()
    #     self.get_preprocessor() # 定义self.steps

    #     # 用来生成observation.language.tokens和observation.language.attention_mask
    #     transition = self.to_transition(self.obs_features)
    #     transformed_transition = self._forward(transition)
    #     return self.to_output(transformed_transition)


    def __call__(self, cfg: EvalPipelineConfig, task_intru: list[str], imgs: list, eef_mat = None,
                 eef_pos = None, eef_quat = None, gripper_qpos = None,
                 joints_pose = None):
        self.cfg = cfg


        # 不再使用自定义的 observation.images.image2，而是使用 pi0 的标准命名
        # imgs[0] -> observation.images.top (外部)
        # imgs[1] -> observation.images.wrist (腕部)
        self.obs_features = {
            "observation.images.image": self.normal_imgs(imgs[0]),
            "observation.images.image2": self.normal_imgs(imgs[1]),
            "observation.robot_state":{
                "eef":{
                    "mat": eef_mat,
                    "pos": eef_pos,
                    "quat": eef_quat
                },
                "gripper":{
                    "qpos": gripper_qpos
                },
                "joints":{
                    "pos": joints_pose
                }
            },
            "task": task_intru
        }


    # cv2.imwrite("../1_ours.png", (np.array(self.obs_features['observation.images.image']) * 255).astype(np.int)[0].transpose(1,2,0))

        # 将机械臂的状态observation.robot_state转为pi0的输入量observation.state
        self.env_preprocessor()
        # 确保初始化 preprocessor (只执行一次)
        self.get_preprocessor()

        # 用来生成observation.language.tokens和observation.language.attention_mask
        transition = self.to_transition(self.obs_features)
        transformed_transition = self._forward(transition)
        return self.to_output(transformed_transition)

if __name__ == "__main__":
    # @parser.wrap()将命令行中定义的变量转换到cfg中
    @parser.wrap()
    def eval_main(cfg: EvalPipelineConfig):
        return cfg
    cfg = eval_main()

    task_intru = ["pick up the alphabet soup and place it in the basket"]
    imgs = [(torch.rand(1, 480, 640, 3) * 255).int(), (torch.rand(1, 480, 640, 3) * 255).int()]
    eef_mat = torch.rand(1, 3, 3)
    eef_pos = torch.rand(1, 3)
    eef_quat = torch.rand(1, 4)
    gripper_qpos = torch.rand(1, 2)
    gripper_qvel = torch.rand(1, 2)
    joints_pose = torch.rand(1, 7)
    joints_vel = torch.rand(1, 7)
    data_trans = PI05_Data_Trans()
    res = data_trans(cfg, task_intru, imgs, eef_mat, eef_pos, eef_quat,
                    gripper_qpos, joints_pose)
    import pdb; pdb.set_trace()
