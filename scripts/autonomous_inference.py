import argparse
import importlib.util
from collections import deque
from pathlib import Path

import numpy as np
import torch
from deoxys.utils.io_devices import SpaceMouse
from diffusers.schedulers.scheduling_ddpm import DDIMScheduler, DDPMScheduler
from shutdown import stop_event

from diffusion_policy.inference import ConditionalDiffusionPolicy
from diffusion_policy.utils import ZarrDataConfig, load_pretrained_nets
from roboenv import RealWorldEnvironment


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Inference with full autonomy")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="path/to/config.py",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="path/to/model.ckpt",
    )
    return parser.parse_args()


def load_config(config_path: str) -> ZarrDataConfig:
    """
    Load a configuration file given its file path.

    Args:
        config_path (str): File path to the config file.

    Returns:
        ZarrDataConfig: The configuration object defined in the config file.
    """
    config_path_obj: Path = Path(config_path)
    spec = importlib.util.spec_from_file_location(config_path_obj.stem, str(config_path_obj))
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module.config


def main(config: ZarrDataConfig, ckpt_path: str):
    interface_cfg = "/home/robomaster/git/robo_copilot/roboenv/configs/charmander.yml"
    controller_cfg = "/home/robomaster/git/robo_copilot/roboenv/configs/osc-position-controller.yml"
    controller_type = "OSC_POSE"
    # Configuration parameters: adjust robot IP and camera IDs as needed.
    camera_ids = [0, 1]  # list of ZED camera IDs

    # Initialize controller
    controller = SpaceMouse()
    controller.start_control()

    capture_rate = 10  # hz

    # Initialize environment
    env = RealWorldEnvironment(
        controller=controller,
        camera_ids=camera_ids,
        interface_cfg=interface_cfg,
        controller_cfg=controller_cfg,
        controller_type=controller_type,
        save_dir=None,
        fps=capture_rate,
    )

    nets = load_pretrained_nets(ckpt_path, config)
    scheduler = DDPMScheduler(num_train_timesteps=config.num_diffusion_iters, beta_schedule="squaredcos_cap_v2", clip_sample=True, prediction_type="epsilon")
    scheduler = DDIMScheduler.from_config(scheduler.config)
    scheduler.set_timesteps(num_inference_steps=10)
    policy = ConditionalDiffusionPolicy(config=config, nets=nets, scheduler=scheduler)
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    policy.to(device)

    proprio_deque = deque(maxlen=config.obs_horizon)
    img_deque = deque(maxlen=config.obs_horizon)
    act_deque = deque(maxlen=config.pred_horizon)
    exe_deque = deque(maxlen=config.pred_horizon - config.obs_horizon + 1)
    user_inputs = deque(maxlen=3)  # seed horizon
    env.launch()
    try:
        while not stop_event.is_set():
            # Get synchronized observation from robot and cameras
            obs = env.get_observation()
            # Append obs to double-ended queue
            proprio_deque.append(obs["robot_state"])
            views = []
            for cam in camera_ids:
                img = obs[f"cam_{cam}"]
                views.append(img.transpose(2, 0, 1))
            img_deque.append(np.stack(views, 0))
            # Get sequence of policy preds if not enough are left in exe_deque
            # this should hold the actions played by the user over the past 3 timesteps
            user_action = env.get_controller_action()
            while len(user_inputs) < user_inputs.maxlen:
                user_inputs.append(user_action)
            if len(exe_deque) <= config.obs_horizon:
                if len(proprio_deque) < config.obs_horizon:
                    continue
                # Get action signal from SpaceMouse

                print(f"{user_action = }")
                # Generate noise for action
                # user_action = np.array([0., 0., 0., 0., 0., 0., -1.])

                # Append user_action to act_deque
                while len(act_deque) < act_deque.maxlen:
                    act_deque.append(user_action)
                # diffusion!
                expert_actions = policy(
                    act_deque=act_deque, user_inputs=user_inputs, proprio_deque=proprio_deque, img_deque=img_deque, gamma=0.2, inpainting="total"
                )
                # Fill act_deque with preds
                for i, act in enumerate(expert_actions.squeeze(axis=0)):  # remove batch dim
                    act_deque.append(act)
                    if i >= config.obs_horizon - 1:
                        exe_deque.append(act)
                for _ in range(config.obs_horizon - 1):
                    act_deque.popleft()

            # Get current expert action
            expert_action = exe_deque.popleft()
            user_inputs.popleft()
            act_deque.popleft()
            # Execute the action on the FRANKA arm and sleep for 1/capture_rate seconds
            env.step(expert_action)
            act_deque.append(expert_action)
    except KeyboardInterrupt:
        stop_event.set()

    env.shutdown()


if __name__ == "__main__":
    args = parse_args()
    config: ZarrDataConfig = load_config(args.config)
    main(config, args.ckpt)
