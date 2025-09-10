"""
DiffusionPipeline class for inference.

Main loop goes something like:

env.launch()
while True:
    obs = env.get_observation()
    obs_deque.append(obs)
    action = pipeline(obs_deque)
    env.step(action)
"""

import copy
from typing import Deque

import numpy as np
import torch
import torch.nn as nn

from diffusion_policy.utils import deque2tensor, get_condition


# class ConditionalDiffusionPolicy(DiffusionPipeline):
class ConditionalDiffusionPolicy:
    def __init__(self, config, nets: nn.ModuleDict, scheduler) -> None:
        super().__init__()
        self.config = config
        self.nets = nets
        self.nets.eval()
        self.scheduler = scheduler
        self.device = torch.device("cpu")

    def to(self, device):
        self.nets = self.nets.to(device)
        self.device = device

    @torch.no_grad()
    def __call__(
        self,
        act_deque: Deque[np.ndarray],
        user_inputs: Deque[np.ndarray],
        proprio_deque: Deque[np.ndarray],
        img_deque: Deque[np.ndarray],
        gamma: float,
        inpainting: str = "total",
    ):
        """
        Args:
            act_deque (Deque[np.ndarray]): Robot control signals from user
            proprio_deque (Deque[np.ndarray]): Deque of past obs_horizon frames of proprioceptive info from RoboEnv
            img_deque (Deque[np.ndarray]): Deque of past obs_horizon frames of camera info from RoboEnv
            gamma (float in [0, 1]): diffusion strength, 0 = no diffusion and 1 = diffuse all the way.
                Setting gamma = 1 means fully-autonomous control, gamma = 0 means full user control.

        Returns:
            np.ndarray: generated shared-autonomy control signal.
        """
        print("spacemouse:", user_inputs)
        # 1. process inputs into torch tensors and move to device
        action: torch.Tensor = deque2tensor(act_deque).float().to(self.device)  # torch.Size([1,12,7]) -- batch size, timesteps, action dim?
        proprio: torch.Tensor = deque2tensor(proprio_deque).float().to(self.device)  # torch.Size([1,3,7])
        images: torch.Tensor = deque2tensor(img_deque).float().to(self.device)  # torch.Size([1,3,2,3,224,224])
        user_inputs: torch.Tensor = deque2tensor(user_inputs).float().to(self.device)  # # torch.Size([1, 3, 7])
        # torch.Size([1,3,7]) --> torch.Size([1,12,7]) by repeating the last row
        future_user_inputs = [user_inputs[:, -1, :] for _ in range(9)]
        future_user_inputs = torch.unsqueeze(torch.vstack(future_user_inputs), dim=0)
        user_inputs = torch.cat([user_inputs, future_user_inputs], dim=1)

        # seeds actions with past and anticipated user actions / intent
        action = copy.deepcopy(user_inputs)

        # 2. get conditioning features
        condition: torch.Tensor = get_condition(proprio, images, self.nets)

        # 3 (A): run full forward diffusion on future actions
        # 3 (B): run partial forward diffusion on past user actions
        num_timesteps: int = self.scheduler.config.num_train_timesteps  # total number of diffusion steps T
        k_sw: int = max(int(gamma * num_timesteps) - 1, 0)  # number of diffusion steps for shared autonomy
        timesteps: torch.Tensor = torch.LongTensor(self.scheduler.timesteps[-k_sw])  # actual timesteps for shared autonomy
        noise: torch.Tensor = torch.randn_like(action)  # noise used for forward process
        a_t: torch.Tensor = self.scheduler.add_noise(action, noise, timesteps)  # generated noise for forward process for shared autonomy

        # 4 Run reverse diffusion on future actions
        # case 4(A): partial inpainting
        if inpainting == "partial":
            a_t[:, :3, :] = user_inputs[:, :3, :]  # seed a_{t-2}, a_{t-1}, a_{t} with user intentions u_{t-2}, u_{t-1}, u_{t}
            # loop over all diffusion steps
            for t in self.scheduler.timesteps:
                residual: torch.Tensor = self.nets["noise_pred_net"](a_t, t, global_cond=condition)
                a_t = self.scheduler.step(residual, t, a_t).prev_sample
                # in-paint user inentions for the first k_sw steps of reverse diffusion
                if t not in self.scheduler.timesteps[:-k_sw:]:
                    # we need to inpaint begining of trajectory
                    a_t[:, :3, :] = user_inputs[:, :3, :]  # overrides a_{t-2}, a_{t-1}, a_{t} with user intentions u_{t-2}, u_{t-1}, u_{t}

        # case 4(B): complete inpainting
        if inpainting == "total":
            a_t[:, :3, :] = user_inputs[:, :3, :]  # overrides a_{t-2}, a_{t-1}, a_{t} with user intentions u_{t-2}, u_{t-1}, u_{t}
            for t in self.scheduler.timesteps[-k_sw:]:
                residual: torch.Tensor = self.nets["noise_pred_net"](a_t, t, global_cond=condition)
                a_t = self.scheduler.step(residual, t, a_t).prev_sample
                a_t[:, :3, :] = user_inputs[:, :3, :]  # overrides a_{t-2}, a_{t-1}, a_{t} with user intentions u_{t-2}, u_{t-1}, u_{t}

        # case 4(C): seed in-painting
        if inpainting == "seed":
            a_t[:, :3, :] = user_inputs[:, :3, :]  # overrides a_{t-2}, a_{t-1}, a_{t} with user intentions u_{t-2}, u_{t-1}, u_{t}
            for t in self.scheduler.timesteps[-k_sw:]:
                """performs a single step of reverse diffusion"""
                # timestep = t.unsqueeze(0).to(self.device)
                residual: torch.Tensor = self.nets["noise_pred_net"](a_t, t, global_cond=condition)
                a_t = self.scheduler.step(residual, t, a_t).prev_sample

        # case 4(D): No in-painting
        if inpainting == "none" or inpainting == "autonomous":
            for t in self.scheduler.timesteps[-k_sw:]:
                """performs a single step of reverse diffusion"""
                # timestep = t.unsqueeze(0).to(self.device)
                residual: torch.Tensor = self.nets["noise_pred_net"](a_t, t, global_cond=condition)
                a_t = self.scheduler.step(residual, t, a_t).prev_sample

        # 5. convert to numpy array
        expert_action: np.ndarray = a_t.cpu().numpy()

        return expert_action
