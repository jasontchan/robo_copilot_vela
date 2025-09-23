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

from policy.openpi_policy import OpenPIPolicyClient


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the inference script.

    """
    parser = argparse.ArgumentParser(description="Inference via PI_0.5")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="path/to/config.py",
    )
    parser.add_argument(
        "--host_ip",
        type=str,
        required=True,
        help="0.0.0.0",
    )
    parser.add_argument(
        "--port",
        type=int,
        required=True,
        help="00",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help='input prompt for VLA here'
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


def main(config: ZarrDataConfig, host_ip: str, port: int, prompt: str, time_limit: float, save_dir: Optional[str]=None) -> None:
    """
    Set up and run the inference loop for the shared autonomy system.

    This function performs the following:
      1. Initializes the robotic environment and controller.
      2. Connects to openpi client.
        2.5 Openpi instantiates the checkpoint vla policy
      3. Launches the environment and continuously queries the policy to compute actions based on sensor observations
         and user emg, which are then executed by the robot.
      4. Handles graceful shutdown on keyboard interrupt.

    Args:
        config (ZarrDataConfig): The configuration object containing relevant system parameters.
        ckpt_path (str): Path to the pretrained model checkpoint.
        seed (str): Seeding strategy to use for future action generation ('autonomous', 'partial', or 'total').
        gamma (float): Diffusion strength; typically a value between 0.0 and 1.0.
    """

    # Define paths for the interface and controller configuration files.
    interface_cfg: str = "/home/chopper/robo_copilot_vela/configs/charmander.yml"
    controller_cfg: str = "/home/chopper/robo_copilot_vela/configs/joint-position-controller.yml"
    controller_type: str = "JOINT_POSITION"
    camera_ids: list[int] = [0, 1]  # List of ZED camera IDs

    # Initialize the space mouse controller.
    print(f"\n{50 * '='}\nLaunching SpaceMouse...\n")
    controller: SpaceMouse = SpaceMouse(pos_sensitivity=0.7, rot_sensitivity=0.5)
    controller.start_control()

    print(f"SpaceMouse launched!\n{50 * '='}\n")

    capture_rate: int = 10  # Capture rate in Hz

    # Initialize the real-world robotic environment.
    env: RealWorldEnvironment = RealWorldEnvironment(
        controller=controller,
        camera_ids=camera_ids,
        emg_mac_tty=None,
        interface_cfg=interface_cfg,
        controller_cfg=controller_cfg,
        controller_type=controller_type,
        save_dir=save_dir,
        fps=capture_rate,
    )

    print(f"\n\n{50 * '='}\nLoading Policy...")
    # Load pretrained network modules.
    # Connect to openpi policy
    client = OpenPIPolicyClient(host_ip=host_ip, port=port)
    print(f"Instantiated OpenPI Client!\n{50 * '='}\n\n")
    # Set device (use CUDA if available, otherwise CPU)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # policy.to(device)
    print(f"Policy loaded!\n{50 * '='}\n\n")
    start_time = time.time()
    # Launch the environment and begin inference.
    env.launch()
    try:
        while not stop_event.is_set() and time.time() - start_time < time_limit:
            # Get synchronized observation from the robot sensors and cameras.
            obs = env.get_observation()

            # Compute the expert action using the policy.
            expert_action = client.infer_droid(ext_img=obs['cam_0'], wrist_img=obs['cam_1'], state=obs['robot_state'][:-1], gripper_state=obs['robot_state'][-1], text=prompt)
           
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
         host_ip=args.host_ip,
         port=args.port,
         prompt=args.prompt,
         time_limit=args.time_limit,
         save_dir=args.save_dir)
