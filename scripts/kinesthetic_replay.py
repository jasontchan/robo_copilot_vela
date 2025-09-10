# main.py
"""
Replay the trajectory collected by kinesthetic teaching and record the image 
"""
import argparse
import json
import h5py
import os
from easydict import EasyDict
import time

from roboenv import RealWorldEnvironment
from deoxys.utils.config_utils import robot_config_parse_args
from deoxys import config_root
from deoxys.franka_interface import FrankaInterface
from deoxys.utils.io_devices import SpaceMouse
from deoxys.utils.log_utils import get_deoxys_example_logger

from shutdown import stop_event

logger = get_deoxys_example_logger()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vendor_id",
        type=int,
        default=9583,
    )
    parser.add_argument(
        "--product_id",
        type=int,
        default=50734,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="recorded_trajecotry.hdf5",
    )
    robot_config_parse_args(parser)
    return parser.parse_args()

def main():
    args = parse_args()

    # Load recorded demonstration trajectories
    with open(args.dataset, "r") as f:
        demo_file = h5py.File(args.dataset)

        config = json.loads(demo_file["data"].attrs["config"])

        joint_sequence = demo_file["data/joint_states"]
        eef_sequence = demo_file["data/ee_states_all"] # absolute value of [x, y, z, p, r, y, g]
    interface_cfg = "/home/robomaster/git/robo_copilot/roboenv/configs/charmander.yml"
    controller_cfg = "/home/robomaster/git/robo_copilot/roboenv/configs/osc-position-controller.yml"
    controller_type = "OSC_POSE"

    # Configuration parameters: adjust robot IP and camera IDs as needed.
    camera_ids = [0, 1]         # list of ZED camera IDs
    data_save_path = "/home/robomaster/data/cereal_making_formal_replay"

    # Initialize controller
    device = SpaceMouse()
    device.start_control()

    capture_rate = 100 #hz

    # Initialize environment
    env = RealWorldEnvironment(device, camera_ids, interface_cfg, controller_cfg, controller_type, data_save_path, capture_rate)

    # Initialize data recorder to log synchronized data
    #recorder = DataRecorder(data_save_path)

    # reset joint to initial states
    env.reset_to_initial_joints(joint_sequence[0])

    try:
        env.launch()
        for eef_state in eef_sequence:
            if stop_event.is_set():
                break
            # Get synchronized observation from robot and cameras
            observation = env.get_observation()
            # compute real action from eef_sequence
            action = env.compute_action(eef_state=eef_state)
            # Record the observation-action pair for training data
            env.record(observation, action)
            # Execute the action on the FRANKA arm and sleep for 1/capture_rate seconds
            env.step(action)
    except KeyboardInterrupt:
        print('kinesthetic_replay.py Keyboard Interrupt. Threads may still be running.')
    finally:
        #env.shutdown()
        env.write_to_disk()

if __name__ == "__main__":
    main()
