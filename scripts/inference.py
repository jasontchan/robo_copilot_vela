import argparse
import importlib.util
from pathlib import Path
from typing import Any, Optional
import time

import torch
from deoxys.utils.io_devices import SpaceMouse
from diffusers import DDIMScheduler, DDPMScheduler
from shutdown import stop_event

from diffusion_policy.inference import DiffusionPolicy, BaselineDiffusionPolicy
from diffusion_policy.utils import ZarrDataConfig, load_pretrained_nets
from roboenv import RealWorldEnvironment


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the inference script.

    Returns:
        argparse.Namespace: A namespace containing the parsed command-line arguments, which include:
            - seed (str): Seeding strategy; should be one of 'autonomous', 'partial', or 'total'.
            - gamma (float): Diffusion strength (a value between 0.0 and 1.0).
            - config (str): Path to the configuration file (config.py).
            - ckpt (str): Path to the pretrained model checkpoint.
    """
    parser = argparse.ArgumentParser(description="Inference! Autonomous or shared!")
    parser.add_argument(
        "--seed",
        type=str,
        required=True,
        help="Choose from 'autonomous', 'partial', 'total'.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        required=True,
        help="0.0 <= Diffusion strength <= 1.0.",
    )
    parser.add_argument(
        "--rho",
        type=float,
        required=True,
        help="0.0 <= In-painting ratio <= 1.0.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        required=True,
        help="a = alpha * u + (1 - alpha) * e",
    )
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
    parser.add_argument(
        "--ddim",
        type=bool,
        default=False,
        help="Set to True for ddim sampling",
    )
    parser.add_argument(
        "--baseline",
        type=bool,
        default=False,
        help="Set to True for baseline policy",
    )
    parser.add_argument(
        "--time_limit",
        type=float,
        default=300.0,
        help="Set a time limit in seconds",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="path/to/sace_dir",
    )
    parser.add_argument("--corrupt", type=bool, default=False, help="Set to True for fun times")
    return parser.parse_args()


def load_config(config_path: str) -> ZarrDataConfig:
    """
    Load a configuration file from the given file path.

    This function dynamically imports the configuration module using its file path and extracts the
    `config` object defined within it.

    Args:
        config_path (str): The file path to the configuration file (typically a Python file).

    Returns:
        ZarrDataConfig: The configuration object defined in the config file.
    """
    config_path_obj: Path = Path(config_path)
    spec = importlib.util.spec_from_file_location(config_path_obj.stem, str(config_path_obj))
    config_module: Any = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module.config


def main(config: ZarrDataConfig, ckpt_path: str, seed: str, gamma: float, rho: float, alpha: float, corrupt: bool, ddim: bool, baseline: bool, time_limit: float, save_dir: Optional[str]=None) -> None:
    """
    Set up and run the inference loop for the shared autonomy system.

    This function performs the following:
      1. Initializes the robotic environment and controller.
      2. Loads the pretrained neural network modules and diffusion scheduler.
      3. Instantiates the DiffusionPolicy with the provided configuration, networks, scheduler, seeding strategy, and gamma.
      4. Launches the environment and continuously queries the policy to compute actions based on sensor observations
         and user inputs, which are then executed by the robot.
      5. Handles graceful shutdown on keyboard interrupt.

    Args:
        config (ZarrDataConfig): The configuration object containing relevant system parameters.
        ckpt_path (str): Path to the pretrained model checkpoint.
        seed (str): Seeding strategy to use for future action generation ('autonomous', 'partial', or 'total').
        gamma (float): Diffusion strength; typically a value between 0.0 and 1.0.
    """
    # Define paths for the interface and controller configuration files.
    interface_cfg: str = "/home/robomaster/git/robo_copilot/roboenv/configs/charmander.yml"
    controller_cfg: str = "/home/robomaster/git/robo_copilot/roboenv/configs/osc-position-controller.yml"
    controller_type: str = "OSC_POSE"
    camera_ids: list[int] = [0, 1]  # List of ZED camera IDs

    # Initialize the space mouse controller.
    print(f"\n{50 * '='}\nLaunching SpaceMouse...\n")
    controller: SpaceMouse = SpaceMouse(pos_sensitivity=0.7, rot_sensitivity=0.5)
    controller.start_control()

    if corrupt:
        from scripts.corrupt_spacemouse import corruptor
    print(f"SpaceMouse launched!\n{50 * '='}\n")

    capture_rate: int = 10  # Capture rate in Hz

    # Initialize the real-world robotic environment.
    env: RealWorldEnvironment = RealWorldEnvironment(
        controller=controller,
        camera_ids=camera_ids,
        interface_cfg=interface_cfg,
        controller_cfg=controller_cfg,
        controller_type=controller_type,
        save_dir=save_dir,
        fps=capture_rate,
    )

    print(f"\n\n{50 * '='}\nLoading Policy...")
    # Load pretrained network modules.
    nets = load_pretrained_nets(ckpt_path, config)

    # Create and configure the diffusion scheduler.
    # Here we initialize the DDPMScheduler and then obtain a DDIMScheduler from its configuration.
    ddpm_scheduler = DDPMScheduler(num_train_timesteps=config.num_diffusion_iters, beta_schedule="squaredcos_cap_v2", clip_sample=True, prediction_type="epsilon")
    if ddim:
        ddim_scheduler = DDIMScheduler.from_config(ddpm_scheduler.config)
        ddim_scheduler.set_timesteps(num_inference_steps=30)
        scheduler = ddim_scheduler
    else:
        scheduler = ddpm_scheduler

    # Instantiate the diffusion-based shared autonomy policy.
    if not baseline:
        print(f"{100*'='}\nRECEDING HORIZON\n{100*'='}")
        policy = DiffusionPolicy(config=config, nets=nets, scheduler=scheduler, seed_mode=seed, gamma=gamma, rho=rho)
    else:
        policy = BaselineDiffusionPolicy(config=config, nets=nets, scheduler=scheduler, seed_mode=seed, gamma=gamma, rho=rho)

    # Set device (use CUDA if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device)
    print(f"Policy loaded!\n{50 * '='}\n\n")

    # Launch the environment and begin inference.
    env.launch()
    policy.warmup()
    try:
        while not stop_event.is_set() and time.time() - policy.start_time < time_limit:
            # Get synchronized observation from the robot sensors and cameras.
            obs = env.get_observation()
            # Obtain current user action from the controller.
            user_action = env.get_controller_action()
            if corrupt:
                user_action = corruptor.corrupt(user_action)
            # Compute the expert (diffusion-predicted) action using the policy.
            expert_action = policy(obs, user_action)
            expert_action = (1.0 - alpha) * expert_action + alpha * user_action
            expert_action[-1] = user_action[-1]
            env.record(obs, expert_action)
            # Execute the computed action in the environment.
            env.step(expert_action)
        stop_event.set()
    except KeyboardInterrupt:
        # Set the stop event for a graceful shutdown on keyboard interrupt.
        stop_event.set()

    # Clean up the environment.
    env.write_to_disk()
    env.shutdown()


if __name__ == "__main__":
    args: argparse.Namespace = parse_args()
    config: ZarrDataConfig = load_config(args.config)
    main(config=config,
         ckpt_path=args.ckpt,
         seed=args.seed,
         gamma=args.gamma,
         rho=args.rho,
         alpha=args.alpha,
         corrupt=args.corrupt,
         ddim=args.ddim,
         baseline=args.baseline,
         time_limit=args.time_limit,
         save_dir=args.save_dir)
