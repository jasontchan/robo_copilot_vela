import time
from shutdown import stop_event

import pyzed.sl as sl
import numpy as np
from PIL import Image

# TODO: can someone correct this
ZED_SERIAL_NUMBER_MAP: dict = {
    # 0: 23474280,
    0: 23474280,
    1: 14920248,
    # 2: 24680
}


class VisionInterface:
    def __init__(self, camera_id, fps=30, size=(224, 224)):
        self.id = camera_id
        self.serial_number = ZED_SERIAL_NUMBER_MAP[camera_id]
        self.frame = {"timestamp": None, "image": None}
        self.sleep_durr = 1 / fps
        self.size = size

        # Instantiate a ZED camera
        self.camera = sl.Camera()
        self.init_params = sl.InitParameters()
        self.init_params.set_from_serial_number(self.serial_number)
        self.init_params.camera_resolution = sl.RESOLUTION.HD720
        self.init_params.camera_fps = fps
        self.closed = True

    def open(self):
        # Open the camera (serial number is set in init_params)
        err = self.camera.open(self.init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            raise Exception("error opening camera")
        self.closed = False

    def close(self):
        self.closed = True
        self.camera.close()
        print(f"Camera {self.id} closed")

    def prep_img(self, img):
        H, W, C = img.shape

        # Convert NumPy array to PIL Image
        image_pil = Image.fromarray(img)

        # Resize
        image_resized_pil = image_pil.resize(self.size)

        # Back to NumPy array
        resized_image = np.array(image_resized_pil)

        return resized_image

    def update_frame(self):
        image = sl.Mat()
        while not self.closed and not stop_event.is_set() and self.camera.grab() == sl.ERROR_CODE.SUCCESS:
            # A new image is available if grab() returns SUCCESS
            self.camera.retrieve_image(image, sl.VIEW.LEFT)
            image_data = image.get_data()[:,:,:3]  # remove alpha layer
            image_data = image_data[:,:,::-1] # BGR -> RGB
            # image_data is now np.array(720, 1080, (r, g, b))

            image_data = self.prep_img(image_data)

            # Get the timestamp at the time the image was captured
            timestamp = self.camera.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_milliseconds()

            # Update latest frame
            self.frame.update({"timestamp": timestamp, "image": image_data})

            time.sleep(self.sleep_durr)
