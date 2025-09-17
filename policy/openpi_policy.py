'''Wrapper to connect to openpi client'''
import numpy as np
from openpi_client import image_tools, websocket_client_policy

class OpenPIPolicyClient:

    def __init__(self, host_ip, port):
        self._client = websocket_client_policy.WebsocketClientPolicy(host=host_ip, port=port)

    def infer_droid(
            self,
            ext_img,
            wrist_img,
            state,
            gripper_state,
            text,
            resize=224
    ) -> np.ndarray:
        obs = {
            "observation/image": image_tools.convert_to_uint8(image_tools.resize_with_pad(ext_img, resize, resize)),
            "observation/wrist_image": image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, resize, resize)),
            "observation/joint_position": np.asarray(state, dtype=np.float32),
            "observation/gripper_position": np.asarray([gripper_state], dtype=np.float32),
            "prompt": text,
        }
        out = self._client.infer(obs)
        return out['actions'] 
