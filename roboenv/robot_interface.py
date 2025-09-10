# robot_interface.py
import time
import numpy as np
from shutdown import stop_event
from deoxys.utils.transform_utils import mat2euler_via_quat


class RobotInterface():
    def __init__(self, franka_interface, fps=30):
        self.interface = franka_interface
        self.state = None
        self.sleep = 1 / fps

    def update_state(self):
        while not stop_event.is_set():
            ee_rot_matrix, ee_pos = self.interface.last_eef_rot_and_pos
            gripper_width = self.interface.last_gripper_q

            # transfer ee_rot to 3_dim (pitch, roll, yaw)
            # rot_tmp = np.array(mat2euler(ee_rot))
            rot_tmp = mat2euler_via_quat(ee_rot_matrix)
            ee_rot = rot_tmp[[1, 0, 2]]

            self.state = np.concatenate([ee_pos, ee_rot, gripper_width], axis=None) # (x, y, z, p, r, y, g)
            time.sleep(self.sleep)

    def control(self, action, controller_type, controller_cfg):
        self.interface.control(
            controller_type=controller_type,
            action=action,
            controller_cfg=controller_cfg
        )

    # def read_state(self):
    #     # get cureent arm ee_pos on z axis
    #     ee_rot, ee_pos = self.interface.last_eef_rot_and_pos
    #     ee_z_pos = ee_pos[2]
    #     return ee_z_pos
