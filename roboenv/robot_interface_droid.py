# robot_interface.py
import time
import numpy as np
from shutdown import stop_event
from deoxys.utils.transform_utils import mat2euler_via_quat


class RobotInterfaceDroid():
    def __init__(self, franka_interface, fps=30):
        self.interface = franka_interface
        self.state = None
        self.sleep = 1 / fps

    def update_state(self):
        while not stop_event.is_set():
            joint_pos = self.interface.last_q
            gripper_width = self.interface.last_gripper_q

            self.state = np.concatenate([joint_pos, gripper_width], axis=None)
            time.sleep(self.sleep)

    def control(self, action, controller_type, controller_cfg):
        self.interface.control(
            controller_type=controller_type,
            action=action,
            controller_cfg=controller_cfg
        )

