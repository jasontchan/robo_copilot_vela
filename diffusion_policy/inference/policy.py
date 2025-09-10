import threading
import time
from collections import deque
from typing import Deque, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.utils import get_condition
from diffusion_policy.utils.config import DiffusionPolicyConfig


class DiffusionPolicy:
    def __init__(
        self,
        config: DiffusionPolicyConfig,
        nets: nn.ModuleDict,
        scheduler: DDPMScheduler,
        seed_mode: str = "user",
        gamma: float = 0.2,
        rho: float = 0.0,
        use_threading: bool = True,
    ) -> None:
        """
        Initialize the DiffusionPolicy used for asynchronous shared autonomy.

        This policy implements a receding horizon diffusion-based planning strategy that runs
        the heavy diffusion inference asynchronously. It maintains buffers for proprioceptive data,
        camera images, and user inputs, and uses a diffusion process to predict a sequence of control
        actions based on recent user inputs and a seeded future action sequence.

        Args:
            config (ZarrDataConfig): Configuration object containing parameters such as obs_horizon, views,
                                     and optionally inference_delay.
            nets (nn.ModuleDict): Module dictionary containing neural networks (e.g., vision encoder, noise predictor).
            scheduler (DDPMScheduler): The diffusion scheduler used for guiding the reverse diffusion process.
            seed_mode (str, optional): Seeding strategy for future actions ("auto", "hybrid", or "user").
                                       Defaults to "user".
            gamma (float, optional): Noise parameter (0.0 <= gamma <= 1.0) determining the fraction of training timesteps
                                     used during the diffusion process. Defaults to 0.2.
        Raises:
            ValueError: If an unsupported seeding strategy is provided.
        """
        self.config = config
        # Extract planning horizons from config:
        # O: the observation/history horizon
        self.O: int = config.obs_horizon  # number of past frames provided for conditioning
        # D: inference delay. (If not provided in config, default to O - 1 frames.)
        self.D: int = getattr(config, "inference_delay", config.pred_horizon - 2 * self.O)
        # R: re-planning interval = O + D, and total prediction horizon P = 2O + D.
        self.R: int = self.O + self.D
        self.P: int = 2 * self.O + self.D

        self.views: Tuple[int, ...] = config.views
        self.nets: nn.ModuleDict = nets
        self.scheduler: DDPMScheduler = scheduler
        assert seed_mode in {"auto", "hybrid", "user"}, f"Unsupported seeding strategy: {seed_mode}"
        self.seed_mode: str = seed_mode
        self.gamma: float = gamma
        self.rho: float = rho

        self.ksw: int = max(int(self.scheduler.config.num_train_timesteps * self.gamma) - 1, 0)
        self.kip: int = max(int(self.rho * self.ksw), 0)
        ind: int = int(len(self.scheduler.timesteps) * self.gamma)
        self.inference_steps: torch.Tensor = self.scheduler.timesteps[-ind:]

        # Buffers for proprioceptive, visual, and control data.
        self.proprio_buffer: Deque[torch.Tensor] = deque(maxlen=self.O)
        self.image_buffer: Deque[torch.Tensor] = deque(maxlen=self.O)
        self.user_history: Deque[torch.Tensor] = deque(maxlen=self.O)
        self.action_buffer: Deque[torch.Tensor] = deque(maxlen=self.O)
        self.exec_buffer: Deque[torch.Tensor] = deque(maxlen=self.O)

        # Threading attributes for asynchronous planning.
        self.planning_in_progress: bool = False
        self.planning_lock: threading.Lock = threading.Lock()
        self.buffer_lock: threading.Lock = threading.Lock()

        self.t: int = 0  # Frame counter
        self.device: torch.device = torch.device("cpu")

        self.use_threading: bool = use_threading
        self.warmed_up: bool = False
        self.start_time: float = time.time()

    def warmup(self) -> None:
        """
        Run the model once to warm up GPU (otherwise first call takes 3x longer than rest)
        """
        print("\nWarming up DiffusionPolicy...")
        fake_prop: torch.Tensor = torch.randn(1, self.O, self.config.action_dim, dtype=torch.float, device=self.device)
        fake_imgs: torch.Tensor = torch.randn(1, self.O, len(self.views), 3, 224, 224, dtype=torch.float, device=self.device)
        fake_cond: torch.Tensor = get_condition(fake_prop, fake_imgs, self.nets)
        fake_samp: torch.Tensor = torch.randn(1, self.P, self.config.action_dim, dtype=torch.float, device=self.device)
        _ = self.nets["noise_pred_net"](fake_samp, 0, fake_cond)
        self.warmed_up: bool = True
        self.start_time: float = time.time()
        print("Policy warmed up and ready to go!\n")

    def to(self, device: torch.device) -> None:
        """
        Move the network components and policy to the specified device.

        Args:
            device (torch.device): The target device, e.g., "cuda" or "cpu".
        """
        self.nets.to(device)
        self.device = device
        print(f"Moved DiffusionPolicy to {device}.")

    def update_history(self, observation: Dict[str, np.ndarray], user_action: np.ndarray) -> None:
        """
        Update the history buffers with the current observation and user action.

        Processes the observation (robot state and multiple camera images) and converts them
        into torch tensors that are stored in respective deques. The user action is also converted
        to a tensor and appended to the user history.

        Args:
            observation (Dict[str, np.ndarray]): Dictionary containing sensor data, e.g., "robot_state"
                                                 and camera images ("cam_{n}").
            user_action (np.ndarray): The current control input provided by the user.
        """
        proprio: torch.Tensor = torch.from_numpy(observation["robot_state"]).to(self.device)
        views: List[torch.Tensor] = []
        for cam in self.views:
            img: torch.Tensor = torch.from_numpy(observation[f"cam_{cam}"].transpose(2, 0, 1)).float().to(self.device)
            views.append(img)
        imgs: torch.Tensor = torch.stack(views, 0).to(self.device)
        action: torch.Tensor = torch.from_numpy(user_action).to(self.device)
        self.proprio_buffer.append(proprio)
        self.image_buffer.append(imgs)
        self.user_history.append(action)

    def buffers_are_equal(self) -> bool:
        """
        Helper to check if self.action_buffer and self.exec_buffer contain equal elements.
        """
        if len(self.action_buffer) != len(self.exec_buffer):
            return False
        return all(torch.equal(t1, t2) for t1, t2 in zip(self.action_buffer, self.exec_buffer))

    def construct_future_seed(self) -> List[torch.Tensor]:
        """
        Construct the seed for future action frames based on the selected seeding strategy.

        For "auto" and "user" modes, the seed is a repetition of the last user action.
        In "hybrid" mode, the seed is constructed from the tail of the previously buffered actions
        (if available) and padded as needed. This seed is used to condition the forward diffusion.

        Returns:
            List[torch.Tensor]: A list of future seed actions of length equal to the re-planning interval (R).

        Raises:
            ValueError: If an unsupported seeding strategy is provided.
        """
        last_user: torch.Tensor = self.user_history[-1]

        if self.seed_mode in {"auto", "user"}:
            seed: List[torch.Tensor] = [last_user] * self.R
        elif self.seed_mode == "hybrid":
            if self.use_threading:
                if len(self.action_buffer) > 0:
                    last_pred: torch.Tensor = self.action_buffer[-1]
                    seed = list(self.action_buffer)[-self.D :] + [last_pred] * self.O
                else:
                    seed = [last_user] * self.R
            else:
                # If action_buffer has elements, use the tail and pad with the last prediction.
                if len(self.action_buffer) > 0:
                    last_pred: torch.Tensor = self.action_buffer[-1]
                    seed = list(self.action_buffer)[-self.D :] + [last_pred] * self.O
                else:
                    seed = [last_user] * self.R

        else:
            raise ValueError(f"Unsupported seeding strategy: {self.seed_mode}")

        return seed

    def forward_diffusion(self, seeded_input: torch.Tensor) -> torch.Tensor:
        """
        Apply forward diffusion to the seeded input to generate a noisy version.

        Adds Gaussian noise to the input based on a fraction of the total training timesteps,
        as determined by the gamma parameter.

        Args:
            seeded_input (torch.Tensor): The input tensor to be noised.

        Returns:
            torch.Tensor: The noised tensor.
        """
        noise: torch.Tensor = torch.randn_like(seeded_input)
        timesteps: torch.Tensor = torch.LongTensor([self.ksw])
        return self.scheduler.add_noise(seeded_input, noise, timesteps)

    @torch.no_grad()
    def reverse_diffusion(self, noisy_input: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Perform reverse diffusion to generate a prediction from the noisy input.

        Iteratively denoises the input tensor by predicting the noise residual using the noise
        prediction network, updating the sample via the scheduler's step function until the sample
        is sufficiently denoised.

        Args:
            noisy_input (torch.Tensor): The input tensor after forward diffusion.
            condition (torch.Tensor): The conditioning tensor computed from sensor data.

        Returns:
            torch.Tensor: The denoised tensor representing the predicted action sequence.
        """
        sample: torch.Tensor = noisy_input.unsqueeze(0).float()
        for t in self.inference_steps:
            # Predict noise added at current timestep
            residual: torch.Tensor = self.nets["noise_pred_net"](sample, t, condition.float())
            # Inpaint first O frames with partially diffused user actions for first k_ip steps
            if t > self.kip:
                inpaint: torch.Tensor = torch.stack(list(self.user_history), dim=0).to(self.device)
                sample[:, : self.O] = self.scheduler.add_noise(inpaint, torch.randn_like(inpaint), t)
            # Take reverse diffusion step
            sample = self.scheduler.step(residual, t, sample).prev_sample
        return sample

    def plan(self) -> None:
        """
        Execute the diffusion-based planning step to generate future control actions.

        Constructs the conditioning sequence by concatenating the user history with future seed actions,
        then applies forward and reverse diffusion to predict a sequence of control actions.
        The last O predicted actions are stored in the action_buffer in a thread-safe manner.
        """
        print(f"{54 * ' '}Before planning: act_buf = {len(self.action_buffer)}, exe_buf = {len(self.exec_buffer)}")
        S_list: List[torch.Tensor] = list(self.user_history) + self.construct_future_seed()
        S: torch.Tensor = torch.stack(S_list, dim=0)

        proprio_batch: torch.Tensor = torch.stack(list(self.proprio_buffer), dim=0).unsqueeze(0)
        image_batch: torch.Tensor = torch.stack(list(self.image_buffer), dim=0).unsqueeze(0)
        condition: torch.Tensor = get_condition(proprio_batch, image_batch, self.nets)

        noisy_S: torch.Tensor = self.forward_diffusion(S)
        S_pred: torch.Tensor = self.reverse_diffusion(noisy_S, condition).squeeze()

        self.action_buffer.clear()
        for action in S_pred[-self.O :]:
            self.action_buffer.append(action)
            self.exec_buffer.append(action)

        print(f"{54 * ' '}After planning: {len(self.action_buffer) = }, {len(self.exec_buffer) = }")

        if not self.buffers_are_equal():
            print(f"\n***Buffers are not equal in length! {len(self.action_buffer) = } while {len(self.exec_buffer) = }***\n")

    def _async_plan(self) -> None:
        """
        Private method to run the planning step asynchronously.

        This method invokes the synchronous plan() function and ensures that the planning_in_progress flag
        is reset when the planning is complete.
        """
        try:
            start_t: int = self.t
            start_time: float = time.time()
            print(f"{50 * ' '}Frame {start_t}, time {start_time - self.start_time:.2f}: Begin async planning...")
            self.plan()
            print(
                f"{50 * ' '}Frame {self.t}, time {time.time() - self.start_time:.2f}: Finished async planning, took {self.t - start_t} frames or {time.time() - start_time:.2f} secs"
            )
        finally:
            with self.planning_lock:
                self.planning_in_progress = False

    def __call__(self, observation: Dict[str, np.ndarray], user_action: np.ndarray) -> np.ndarray:
        """
        Compute and return the next action based on the current observation and user action.

        Updates internal histories, and—if appropriate—launches a background planning thread.
        If buffered predicted actions are available, returns the next action from the buffer.
        Otherwise, falls back on the current user action.

        Args:
            observation (Dict[str, np.ndarray]): Dictionary of current sensor data (robot state and images).
            user_action (np.ndarray): The current control input from the user.

        Returns:
            np.ndarray: The control action to execute.
        """
        if not self.warmed_up:
            print("\n***Running DiffusionPolicy without calling policy.warmup() might lead to unexpected behavior!***\n")

        print(f"\n\n{120 * '='}\n{15 * ' '}Control Thread:{30 * ' '}Policy Thread:")
        print(f"Frame {self.t}, time {time.time() - self.start_time:.2f}: act_buf: {len(self.action_buffer)}, exe_buf: {len(self.exec_buffer)}")

        self.update_history(observation, user_action)

        # If it's time to replan and no planning is already in progress, launch an asynchronous planning thread.
        if (self.t + 1) % self.O == 0:
            if self.use_threading:
                with self.planning_lock:
                    if not self.planning_in_progress:
                        self.planning_in_progress = True
                        threading.Thread(target=self._async_plan, daemon=True).start()
            else:
                print(f"{self.t = } No actions left in buffer! time to diffuse!")
                start_time: float = time.time()
                start_t: int = self.t
                self.plan()
                print(f"{self.t = } Planning took {self.t - start_t} frames or {time.time() - start_time:.3f} secs")

        if len(self.exec_buffer) > 0 and self.t >= self.O + self.D:
            action: np.ndarray = self.exec_buffer.popleft().squeeze(0).detach().cpu().numpy()
        else:
            print("\n***No policy action available, falling back to user action***\n")
            action = user_action

        self.t += 1  # Increment frame counter.
        return action
