"""Moving robot joint positions to initial pose for starting new experiments."""
import numpy as np
import argparse
from pathlib import Path

from deoxys.utils import YamlConfig
from deoxys.franka_interface import FrankaInterface
from deoxys.experimental.motion_utils import reset_gripper_move_to


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface-cfg", type=str, default="charmander.yml")
    parser.add_argument(
        "--controller-cfg", type=str, default="joint-position-controller.yml"
    )
    parser.add_argument(
        "--folder", type=Path, default="data_collection_example/example_data"
    )
    parser.add_argument(
        "--random", default=False, action='store_true'
    )

    args = parser.parse_args()
    return args

def sample_random_pos():
    x_range = (0.3, 0.9)
    y_range = (-0.3, 0.3)
    z_range = (0.05, 0.25)

    x = np.random.uniform(low=x_range[0], high=x_range[1])
    y = np.random.uniform(low=y_range[0], high=y_range[1])
    z = np.random.uniform(low=z_range[0], high=z_range[1])

    return np.array([[x],[y],[z]])

def sample_random_joint():
    j1_range = (-2, 2)      # real: (-2.89, 2.89)
    j2_range = (0.8, 3)     # real: (0.3, 3.75)
    j3_range = (-2, 2)      # real: (-2.89, 2.89)

    j1 = np.random.uniform(low=j1_range[0], high=j1_range[1])
    j2 = np.random.uniform(low=j2_range[0], high=j2_range[1])
    j3 = np.random.uniform(low=j3_range[0], high=j3_range[1])

    return [j1, j2, j3]


def main():
    robot_interface = FrankaInterface("/home/robomaster/git/robo_copilot/roboenv/configs/charmander.yml")

    args = parse_args()

    target_pos = sample_random_pos()
    target_rot = sample_random_joint()
    fixed = (not args.random)
    reset_gripper_move_to(robot_interface, target_pos, target_rot, fixed=fixed, rotate=False)

    robot_interface.close()


if __name__ == "__main__":
    main()
