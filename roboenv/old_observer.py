# synchronizer.py
import datetime
import time
import cv2
import pandas as pd
from pathlib import Path


class Observer:
    def __init__(self, robot_interface, camera_interfaces, save_dir=None):
        self.robot = robot_interface
        self.cameras = camera_interfaces
        self.start_time = time.time()
        
        if save_dir is not None:
            now = datetime.datetime.now()
            self.save_dir = f"{save_dir}/{now.year}_{now.month:02d}_{now.day:02d}_{now.hour:02d}_{now.minute:02d}_{now.second:02d}"
            self.proprioceptives = {
                'timestamp': [],
                'x_pos': [], 'y_pos': [], 'z_pos': [],
                'roll': [], 'pitch': [], 'yaw': [],
                'gripper_width': []
            }
            self.actions = {
                'timestamp': [],
                'x_vel': [], 'y_vel': [], 'z_vel': [],
                'roll': [], 'pitch': [], 'yaw': [],
                'gripper': []
            }
            self.image_paths = {
                'timestamp': [],
            }
            for cam in self.cameras:
                self.image_paths[f"cam_{cam.id}"] = []
        else:
            self.save_dir = None


    def get_synchronized_observation(self):
        """
        Gathers the latest robot state and camera frames into one observation dict.
        Each sensor output includes a timestamp, so you can perform further alignment if needed.

        obs = {
            'timestamp': int(milliseconds since start),
            'robot_state': np.array([x_pos, y_pos, z_pos, roll, pitch, yaw, gripper_width]),
            'cam_0': np.array([img_dims]),
            ...,
        }
        """
        obs = {}
        obs['timestamp'] = int((time.time() - self.start_time) * 1000)
        obs['robot_state'] = self.robot.state # (x, y, z, r, p, y, g)
        
        # Save each frame
        for cam in self.cameras:
            obs[f'cam_{cam.id}'] = cam.frame['image']
            # save img to disk only if save_dir is provided
            if self.save_dir is not None:
                cam_folder = Path(f"{self.save_dir}/cam_{cam.id}")
                cam_folder.mkdir(parents=True, exist_ok=True)
                save_name = cam_folder / f"{obs['timestamp']}.jpg"
                cv2.imwrite(save_name, cam.frame['image'])
    
        return obs


    def record(self, obs, action):
        if self.save_dir is None:
            raise ValueError(f"save_dir not specified!")

        self.proprioceptives['timestamp'].append(obs['timestamp'])
        self.proprioceptives['x_pos'].append(obs['robot_state'][0])
        self.proprioceptives['y_pos'].append(obs['robot_state'][1])
        self.proprioceptives['z_pos'].append(obs['robot_state'][2])
        self.proprioceptives['roll'].append(obs['robot_state'][3])
        self.proprioceptives['pitch'].append(obs['robot_state'][4])
        self.proprioceptives['yaw'].append(obs['robot_state'][5])
        self.proprioceptives['gripper_width'].append(obs['robot_state'][6])

        self.actions['timestamp'].append(obs['timestamp'])
        self.actions['x_vel'].append(action[0])
        self.actions['y_vel'].append(action[1])
        self.actions['z_vel'].append(action[2])
        self.actions['roll'].append(action[3])
        self.actions['pitch'].append(action[4])
        self.actions['yaw'].append(action[5])
        self.actions['gripper'].append(action[6])

        self.image_paths['timestamp'].append(obs['timestamp'])
        for cam in self.cameras:
            img_location = f"{self.save_dir}/cam_{cam.id}/{obs['timestamp']}.jpg"
            self.image_paths[f'cam_{cam.id}'].append(img_location)


    def write_to_disk(self):
        if self.save_dir is None:
            raise ValueError("save_dir not specified!")
        pd.DataFrame.from_dict(self.proprioceptives).to_csv(f"{self.save_dir}/proprioceptives.csv")
        pd.DataFrame.from_dict(self.actions).to_csv(f"{self.save_dir}/actions.csv")
        pd.DataFrame.from_dict(self.image_paths).to_csv(f"{self.save_dir}/image_paths.csv")
