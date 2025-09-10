import time
from collections import deque
from typing import Deque, Dict, List

import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.utils import get_condition
from diffusion_policy.utils.config import ZarrDataConfig


class BaselineDiffusionPolicy:
    """
    A baseline diffusion policy that runs a full diffusion pass every frame,
    waits for it to finish, and then executes the next action.
    """

    def __init__(
        self,
        config: ZarrDataConfig,
        nets: nn.ModuleDict,
        scheduler: DDPMScheduler,
        seed_mode: str = "hybrid",
        gamma: float = 0.2,
        rho: float = 0.0,
    ) -> None:
        # horizons
        self.config = config
        self.views = config.views
        self.O: int = config.obs_horizon
        self.D: int = getattr(config, "inference_delay", config.pred_horizon - 2 * self.O)
        self.R: int = self.O + self.D
        self.P: int = 2 * self.O + self.D
        self.t: int = 0
        self.start_time = time.time()

        # model & diffusion setup
        self.nets = nets
        self.scheduler = scheduler
        assert seed_mode in {"auto", "hybrid", "user"}
        self.seed_mode = seed_mode
        self.gamma = gamma
        self.rho = rho

        # compute number of diffusion steps
        self.ksw = max(int(self.scheduler.config.num_train_timesteps * self.gamma) - 1, 0)
        self.kip = int(self.rho * self.ksw)
        ind = int(len(self.scheduler.timesteps) * self.gamma)
        self.inference_steps = self.scheduler.timesteps[-ind:]

        # history buffers
        self.proprio_buffer: Deque[torch.Tensor] = deque(maxlen=self.O)
        self.image_buffer: Deque[torch.Tensor] = deque(maxlen=self.O)
        self.user_history: Deque[torch.Tensor] = deque(maxlen=self.O)

        self.device = torch.device("cpu")
        self.warmed_up = False

    def warmup(self) -> None:
        """
        Run the model once to warm up GPU (otherwise first call takes 3x longer than rest)
        """
        print("\nWarming up BaselineDiffusionPolicy...")
        fake_prop: torch.Tensor = torch.randn(1, self.O, self.config.action_dim, dtype=torch.float, device=self.device)
        fake_imgs: torch.Tensor = torch.randn(1, self.O, len(self.views), 3, 224, 224, dtype=torch.float, device=self.device)
        fake_cond: torch.Tensor = get_condition(fake_prop, fake_imgs, self.nets)
        fake_samp: torch.Tensor = torch.randn(1, self.P, self.config.action_dim, dtype=torch.float, device=self.device)
        _ = self.nets["noise_pred_net"](fake_samp, 0, fake_cond)
        self.warmed_up: bool = True
        self.start_time = time.time()
        print("BaselineDiffusionPolicy warmed up and ready to go!\n")

    def to(self, device: torch.device) -> None:
        """Move model and buffers to specified device."""
        self.nets.to(device)
        self.device = device
        print(f"Moved BaselineDiffusionPolicy to {device}.")

    def construct_future_seed(self) -> List[torch.Tensor]:
        """Build future-action seed list of length R."""
        last = self.user_history[-1]
        if self.seed_mode in {"auto", "user"}:
            return [last] * self.R

        # hybrid
        if len(self.user_history) >= self.R:
            tail = list(self.user_history)[-self.R :]
        else:
            tail = list(self.user_history) + [last] * (self.R - len(self.user_history))
        return tail

    def forward_diffusion(self, seq: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(seq)
        timesteps = torch.LongTensor([self.ksw]).to(self.device)
        return self.scheduler.add_noise(seq, noise, timesteps)

    @torch.no_grad()
    def reverse_diffusion(self, noisy: torch.Tensor, cond: torch.Tensor, user_action_tensor: torch.Tensor) -> torch.Tensor:
        sample: torch.Tensor = noisy.unsqueeze(0).float()
        for t in self.inference_steps:
            residual: torch.Tensor = self.nets["noise_pred_net"](sample, t, cond.float())
            # if t > self.kip:
            #     inpaint: torch.Tensor = user_action_tensor
            #     sample[:, : self.O] = self.scheduler.add_noise(inpaint, torch.randn_like(inpaint), t)
            sample = self.scheduler.step(residual, t, sample).prev_sample
        return sample

    def __call__(self, observation: Dict[str, np.ndarray], user_action: np.ndarray) -> np.ndarray:
        """
        Run a synchronous diffusion pass every frame and return the next action,
        using only the current observation and action (no history).
        """
        if not self.warmed_up:
            self.warmup()

        print(f"\n\n{120 * '='}\n")
        print(f"Frame {self.t}, time {time.time() - self.start_time:.2f}, begin diffusion...")
        start_time: float = time.time()

        # Convert current user action to tensor
        action_tensor = torch.from_numpy(user_action).to(self.device)

        # 1) Build the input sequence: repeat the current action P times
        #    (so the policy sees no history, only the latest command)
        seq = action_tensor.unsqueeze(0).repeat(self.P, 1)  # shape (P, action_dim)

        # 2) Build the proprioceptive batch: repeat the current state O times
        proprio = torch.from_numpy(observation["robot_state"]).to(self.device)
        prop_batch = proprio.unsqueeze(0).unsqueeze(0).repeat(1, self.O, 1)
        #    shape (1, O, state_dim)

        # 3) Build the image batch: repeat the current camera views O times
        views = []
        for cam in self.views:
            img = torch.from_numpy(observation[f"cam_{cam}"].transpose(2, 0, 1)).float().to(self.device)
            views.append(img)
        last_img = torch.stack(views, dim=0)  # shape (n_views, C, H, W)
        img_batch = last_img.unsqueeze(0).unsqueeze(0).repeat(1, self.O, 1, 1, 1, 1)
        #    shape (1, O, n_views, C, H, W)

        # 4) Compute conditioning and run diffusion
        condition = get_condition(prop_batch, img_batch, self.nets)
        noisy = self.forward_diffusion(seq)
        pred = self.reverse_diffusion(noisy, condition, action_tensor).squeeze()  # shape (P, action_dim)

        # 5) Return the very next action (index O in the unfolded plan)
        next_action = pred[self.O].detach().cpu().numpy()

        fmt = lambda arr: "[" + "  ".join(f"{x:6.3f}" for x in arr) + "]"
        print(f"\n\t{'User action:':<20}{fmt(user_action)}")
        print(f"\t{'Generated action:':<20}{fmt(next_action)}\n")
        print(f"{self.t = } Planning took {time.time() - start_time:.3f} secs")

        self.t += 1
        return next_action
