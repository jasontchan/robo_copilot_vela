# environment.py
import threading
import signal
from shutdown import stop_event
import time

import numpy as np
from deoxys.experimental.motion_utils import reset_joints_to
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import YamlConfig
from deoxys.utils.input_utils import input2action

from observer import Observer
from observer_droid import ObserverDroid
from robot_interface import RobotInterface
from robot_interface_droid import RobotInterfaceDroid
from vision_interface import VisionInterface
from emg_interface import EMGInterface

### THOUGHT: use OSC for controller type but then grab robot states as JOINT_POSITION
class RealWorldEnvironment:
    def __init__(
        self,
        controller,
        camera_ids,
        emg_mac_tty, # (mac, tty)
        interface_cfg,
        controller_cfg,
        controller_type,
        save_dir=None,
        fps=10,
    ):
        # Instantiate the controller
        self.controller = controller
        self.controller_cfg = YamlConfig(controller_cfg).as_easydict()
        self.controller_type = controller_type

        # Instantiate the FRANKA interface
        franka_interface = FrankaInterface(interface_cfg)
        # self.robot_interface = RobotInterface(franka_interface)
        self.robot_interface = RobotInterfaceDroid(franka_interface)


        # Instantiate one ZED camera interface per provided camera id
        self.camera_interfaces = [VisionInterface(camera_id=id) for id in camera_ids]
        self.emg_interface = EMGInterface(mac=emg_mac_tty[0], tty=emg_mac_tty[1], window_length=100) if emg_mac_tty is not None else None

        self.sleep_durr = 1 / fps
        self.save_dir = save_dir

        # initilazie the target position
        self.target_pos_z = None

    def launch(self):
        def signal_handler(sig, frame):
            print("\n[Main] Ctrl+C detected. Stopping threads...")
            stop_event.set()

        # Register SIGINT, SIGTERM handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        print(f"\n{100 * '_'}\n{100 * '='}\nLaunching RoboEnv...\n")

        # Launch asynchronous threads for robot state update and camera streaming
        print("\nLaunching thread for Franka Arm...")
        self.robot_thread = threading.Thread(target=self.robot_interface.update_state)
        self.robot_thread.daemon = True
        self.robot_thread.start()
        print("Franka Arm launched!\n")

        print("\nLaunching threads for ZED Cameras...")
        self.camera_threads = []
        for cam in self.camera_interfaces:
            cam.open()
            t = threading.Thread(target=cam.update_frame)
            t.daemon = True
            t.start()
            self.camera_threads.append(t)
        print("Zed Cameras launched!\n")

        print("Launching thread for EMG band...")
        if self.emg_interface is not None:
            self.emg_interface.connect()
            while not self.emg_interface.is_connected():
                time.sleep(0.1)


        print("\nInitiating observer...")
        # The observer will create disk stores only if save_dir is provided.
        # if self.save_dir is not None:
        #     self.observer = Observer(self.robot_interface, self.camera_interfaces, self.emg_interface, self.save_dir)
        # else:
        #     self.observer = Observer(self.robot_interface, self.camera_interfaces, self.emg_interface)
        # print("Observer launched!\n")
        if self.save_dir is not None:
            self.observer = ObserverDroid(self.robot_interface, self.camera_interfaces, self.emg_interface, self.save_dir)
        else:
            self.observer = ObserverDroid(self.robot_interface, self.camera_interfaces, self.emg_interface)
        print("Observer launched!\n")


        print(f"\n{13 * ' '}RoboEnv ready in\n{(13 + 8) * ' '}3... ")
        for i in range(2, 0, -1):
            print(f"{13 * ' '}{8 * ' '}{i}... ")
            time.sleep(1)
        print(f"RoboEnv is ready to go! GLHF!\n{100 * '_'}\n{100 * '='}\n")

    def get_observation(self):
        """
        Return a synchronized observation dictionary.
        """
        return self.observer.get_synchronized_observation()

    def get_controller_action(self):
        action, grasp = input2action(
            device=self.controller,
            controller_type=self.controller_type,
        )
        if action is None:
            stop_event.set()
            print("Controller is off!")
            return np.array([0., 0., 0., 0., 0., 0., -1.])
            # raise ValueError("Controller is off! Please check!")
        else:
            # # to solve the head dropping down problem
            # _, last_eef_pos = self.robot_interface.interface.last_eef_rot_and_pos
            # last_eef_pos = np.array(last_eef_pos.flatten())
            # if self.target_pos_z is None:
            #     self.target_pos_z = last_eef_pos[2]
            # else:
            #     action[2] *= 0.5
            #     real_action = action[2] * self.controller_cfg.action_scale.translation
            #     self.target_pos_z += real_action
            #     new_action = (self.target_pos_z - last_eef_pos[2]) / self.controller_cfg.action_scale.translation
            #     # # avoid the accumulation makes the action too large
            #     # if new_action >= 0.7:
            #     #     new_action = 0.7
            #     action[2] = new_action
            return action

    def reset_to_initial_joints(self, joint_init):
        # Reset to the same initial joint configuration
        reset_joints_to(self.robot_interface.interface, joint_init, gripper_open=True)

    def compute_action(self, eef_state):
        try:
            dpos = eef_state[:3] - self.robot_interface.state[:3]
            dpos /= self.controller_cfg.action_scale.translation

            drot = eef_state[3:6] - self.robot_interface.state[3:6]
            drot = (drot + np.pi) % (2 * np.pi) - np.pi
            drot = drot[[1, 0, 2]]
            drot /= self.controller_cfg.action_scale.rotation

            action = np.hstack((dpos, drot, eef_state[-1]))
            return action
        except Exception as e:
            print(f"Error in computing action: {e}")
            return None

    def step(self, action):
        # to solve the head dropping down problem
        _, last_eef_pos = self.robot_interface.interface.last_eef_rot_and_pos
        last_eef_pos = np.array(last_eef_pos.flatten())
        if self.target_pos_z is None:
            self.target_pos_z = last_eef_pos[2]
        else:
            action[2] *= 0.5
            real_action = action[2] * self.controller_cfg.action_scale.translation
            self.target_pos_z += real_action
            new_action = (self.target_pos_z - last_eef_pos[2]) / self.controller_cfg.action_scale.translation
            # # avoid the accumulation makes the action too large
            # if new_action >= 0.7:
            #     new_action = 0.7
            action[2] = new_action
        self.robot_interface.control(action, self.controller_type, self.controller_cfg)
        time.sleep(self.sleep_durr)

    def record(self, obs, action):
        self.observer.record(obs, action)

    def write_to_disk(self):
        print("\n")  # so ^C is on its own line
        if self.save_dir is None:
            print("No save_dir specified! Exiting...")
            return

        print(f"Writing data to {self.save_dir}...")
        self.observer.write_to_disk()
        print("Done!")

    def shutdown(self):
        print("Interrupted! Shutting down ZED Cameras...")
        for cam, thread in zip(self.camera_interfaces, self.camera_threads):
            cam.close()
            thread.join()
        self.robot_thread.join()
        if self.emg_interface is not None:
            self.emg_interface.disconnect()
        print("Shutdown successful!")
