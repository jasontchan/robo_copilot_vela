# main.py
"""
collects data with SpaceMouse
"""

import argparse
from typing import Union
import time

from deoxys.utils.io_devices import SpaceMouse
from shutdown import stop_event

from roboenv import RealWorldEnvironment


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Collect a demonstration with SpaceMouse.")
    parser.add_argument(
        "--save_dir",
        help="path/to/save_dir",
    )
    parser.add_argument("--corrupt", help="set to True for a fun time", type=bool, default=False)
    parser.add_argument(
        "--time_limit",
        type=float,
        default=300.0,
        help="Set a time limit in seconds",
    )
    return parser.parse_args()


def main(save_dir: Union[str, None], corrupt: bool, time_limit: float):
    interface_cfg = "/home/robomaster/git/robo_copilot/roboenv/configs/charmander.yml"
    controller_cfg = "/home/robomaster/git/robo_copilot/roboenv/configs/osc-position-controller.yml"
    controller_type = "OSC_POSE"
    # Configuration parameters: adjust robot IP and camera IDs as needed.
    camera_ids = [0]  # list of ZED camera IDs

    # Initialize controller
    print(f"\n{50 * '='}\nLaunching SpaceMouse...\n")
    controller = SpaceMouse(pos_sensitivity=0.7, rot_sensitivity=0.5)
    controller.start_control()

    if corrupt:
        from scripts.corrupt_spacemouse import corruptor
    print(f"SpaceMouse launched!\n{50 * '='}\n")

    capture_rate = 10  # hz

    # Initialize environment
    env = RealWorldEnvironment(
        controller=controller,
        camera_ids=camera_ids,
        interface_cfg=interface_cfg,
        controller_cfg=controller_cfg,
        controller_type=controller_type,
        save_dir=save_dir,
        fps=capture_rate,
    )

    # Initialize data recorder to log synchronized data
    # recorder = DataRecorder(data_save_path)

    try:
        env.launch()
        start_time = time.time()
        curr_time = time.time() - start_time
        frame = 0
        while not stop_event.is_set() and curr_time < time_limit:
            curr_time = time.time() - start_time
            print(f"\n{120 * '='}\nFrame: {frame}, Time: {curr_time:.2f}")
            # Get synchronized observation from robot and cameras
            observation = env.get_observation()
            # Get action signal from SpaceMouse
            action = env.get_controller_action()
            print(f"action: {action}")
            # corrupt signal if desired
            if corrupt:
                action = corruptor.corrupt(action)
            # Record the observation-action pair for training data
            env.record(observation, action)
            # Execute the action on the FRANKA arm and sleep for 1/capture_rate seconds
            env.step(action)
            frame += 1
        stop_event.set()
    except KeyboardInterrupt:
        stop_event.set()

    env.write_to_disk()
    env.shutdown()
    if save_dir:
        print(f"Data written to {save_dir}.")


if __name__ == "__main__":
    args = parse_args()
    main(args.save_dir, args.corrupt, args.time_limit)
