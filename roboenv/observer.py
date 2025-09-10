# observer.py
import datetime
import io
import time
from pathlib import Path

import numcodecs
import numpy as np
import zarr
from PIL import Image


class Observer:
    def __init__(self, robot_interface, camera_interfaces, save_dir=None, chunk_size=6):
        self.robot = robot_interface
        self.cameras = camera_interfaces
        self.start_time = time.time()

        # Demo collection mode: create unified Zarr store and datasets.
        if save_dir is not None:
            self.chunk_size = chunk_size
            self.obs_counter = 0
            # Buffers to hold data until flush.
            self.image_buffer = []  # Each element is a list of JPEG bytes for all cameras.
            self.proprio_buffer = []  # Each element is a tuple with proprioceptive data.
            self.actions_buffer = []  # Each element is a tuple with action data.

            # Create a unique save directory.
            self.save_dir = save_dir
            Path(self.save_dir).mkdir(parents=True, exist_ok=True)

            # Nest the Zarr store directory within the save_dir.
            self.zarr_store_dir = (
                f"{self.save_dir}/{datetime.datetime.now():%Y_%m_%d_%H_%M_%S}.zarr"
            )
            self.zarr_store = zarr.DirectoryStore(self.zarr_store_dir)
            self.zarr_root = zarr.group(store=self.zarr_store, overwrite=True)
            n_cameras = len(self.cameras)

            # Create images dataset.
            self.images_ds = self.zarr_root.create_dataset(
                "images",
                shape=(0, n_cameras),
                maxshape=(None, n_cameras),
                chunks=(chunk_size, n_cameras),
                dtype=object,
                object_codec=numcodecs.Pickle(),  # For variable-length JPEG bytes.
                compressor=None,
                overwrite=True,
            )

            # Create proprioceptive data dataset.
            # Here we use a structured dtype with timestamp and seven sensor readings.
            proprio_dtype = np.dtype(
                [
                    ("timestamp", "int64"),
                    ("x_pos", "float32"),
                    ("y_pos", "float32"),
                    ("z_pos", "float32"),
                    ("roll", "float32"),
                    ("pitch", "float32"),
                    ("yaw", "float32"),
                    ("gripper_width", "float32"),
                ]
            )
            self.proprio_ds = self.zarr_root.create_dataset(
                "proprio",
                shape=(0,),
                maxshape=(None,),
                chunks=(chunk_size,),
                dtype=proprio_dtype,
                compressor=None,
                overwrite=True,
            )

            # Create action data dataset.
            actions_dtype = np.dtype(
                [
                    ("timestamp", "int64"),
                    ("x_vel", "float32"),
                    ("y_vel", "float32"),
                    ("z_vel", "float32"),
                    ("roll", "float32"),
                    ("pitch", "float32"),
                    ("yaw", "float32"),
                    ("gripper", "float32"),
                ]
            )
            self.actions_ds = self.zarr_root.create_dataset(
                "actions",
                shape=(0,),
                maxshape=(None,),
                chunks=(chunk_size,),
                dtype=actions_dtype,
                compressor=None,
                overwrite=True,
            )
        else:
            # In inference mode, no disk stores are created.
            self.save_dir = None
            self.zarr_store_dir = None
            self.zarr_store = None
            self.zarr_root = None
            self.images_ds = None
            self.proprio_ds = None
            self.actions_ds = None
            self.image_buffer = None
            self.proprio_buffer = None
            self.actions_buffer = None
            self.chunk_size = None
            self.obs_counter = None

    def get_synchronized_observation(self):
        """
        Collects the latest observation.
        For each camera, returns the raw RGB numpy array (for inference).
        Additionally, if in demo collection mode, also encodes the image as JPEG using PIL.
        """
        obs = {}
        obs["timestamp"] = int((time.time() - self.start_time) * 1000)
        obs["robot_state"] = (
            self.robot.state
        )  # Expected to be something like [x, y, z, roll, pitch, yaw, gripper_width]

        encoded_images = []  # JPEG bytes (only if demo mode)
        for cam in self.cameras:
            frame = cam.frame["image"]  # Expected to be a numpy array in RGB
            obs[f"cam_{cam.id}"] = frame  # Raw image for downstream inference.
            if self.save_dir is not None:
                # Encode using PIL.
                pil_img = Image.fromarray(frame)
                buf = io.BytesIO()
                pil_img.save(buf, format="JPEG", quality=75)
                jpeg_bytes = buf.getvalue()
                encoded_images.append(jpeg_bytes)

        if self.save_dir is not None:
            self.current_encoded_images = encoded_images

        return obs

    def record(self, obs, action):
        """
        Records an observation and its corresponding action.
        In demo collection mode, the JPEG-encoded images, proprioceptive data, and action data
        are buffered in RAM until a full chunk is accumulated and then flushed to the Zarr store.
        """
        if self.save_dir is None:
            return  # In inference mode, do not record anything.

        ts = obs["timestamp"]
        # Buffer JPEG-encoded images.
        self.image_buffer.append(self.current_encoded_images)

        # Buffer proprioceptive data.
        # Expecting robot_state to have 7 elements: [x_pos, y_pos, z_pos, roll, pitch, yaw, gripper_width].
        proprio_entry = (
            ts,
            float(obs["robot_state"][0]),
            float(obs["robot_state"][1]),
            float(obs["robot_state"][2]),
            float(obs["robot_state"][3]),
            float(obs["robot_state"][4]),
            float(obs["robot_state"][5]),
            float(obs["robot_state"][6]),
        )
        self.proprio_buffer.append(proprio_entry)

        # Buffer action data.
        # Expecting action to be an array/list of 7 elements: [x_vel, y_vel, z_vel, roll, pitch, yaw, gripper].
        action_entry = (
            ts,
            float(action[0]),
            float(action[1]),
            float(action[2]),
            float(action[3]),
            float(action[4]),
            float(action[5]),
            float(action[6]),
        )
        self.actions_buffer.append(action_entry)

        self.obs_counter += 1
        if self.obs_counter % self.chunk_size == 0:
            self.flush_buffers()

    def flush_buffers(self):
        """
        Appends the buffered images, proprioceptive data, and action data to their respective datasets,
        then clears the buffers.
        """
        if self.save_dir is None:
            return

        images_chunk = np.array(self.image_buffer, dtype=object)
        self.images_ds.append(images_chunk)

        proprio_chunk = np.array(self.proprio_buffer, dtype=self.proprio_ds.dtype)
        self.proprio_ds.append(proprio_chunk)

        actions_chunk = np.array(self.actions_buffer, dtype=self.actions_ds.dtype)
        self.actions_ds.append(actions_chunk)

        # Clear buffers.
        self.image_buffer = []
        self.proprio_buffer = []
        self.actions_buffer = []

    def write_to_disk(self):
        """
        Flush any remaining buffered data to the Zarr store.
        """
        if self.save_dir is None:
            return

        if self.image_buffer or self.proprio_buffer or self.actions_buffer:
            self.flush_buffers()
