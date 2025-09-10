######################
# Utility Functions  #
######################

import io
from typing import Dict, List, Tuple

import numpy as np
import torch
import torchvision.transforms as T
import zarr
from PIL import Image


def decode_and_process_image(jpeg_bytes: bytes, size: Tuple[int, int] = (224, 224)) -> torch.Tensor:
    """
    Given JPEG-encoded bytes, decode the image using PIL, apply a resize, and return a torch.Tensor.
    The output tensor has shape (C, H, W) and pixel values in [0,1].
    """
    transform: T.Compose = T.Compose(
        [
            T.Resize(size),
            T.ToTensor(),  # converts to float [0,1]
        ]
    )
    with io.BytesIO(jpeg_bytes) as buf:
        pil_img = Image.open(buf).convert("RGB")
        tensor = transform(pil_img)
    return tensor


def create_sample_indices_from_trial(
    trial_length: int,
    sequence_length: int,
    pad_before: int,
    pad_after: int,
) -> np.ndarray:
    """
    For a single trial of length trial_length, compute sample indices.
    Returns an array of shape (N, 4) where each row is
    [buffer_start, buffer_end, sample_start, sample_end].

    buffer indices are in the global trial data (0-indexed) and sample indices are
    where that slice should be placed in a final sequence window of fixed length.
    """
    indices: List[List[int]] = []
    # For each trial, the valid global indices go from 0 to trial_length.
    # We allow negative start indices up to -pad_before and indices beyond trial_length up to pad_after.
    min_start = -pad_before
    max_start = trial_length - sequence_length + pad_after
    for idx in range(min_start, max_start + 1):
        buffer_start = max(idx, 0)
        buffer_end = min(idx + sequence_length, trial_length)
        # How many frames are missing before and after?
        left_pad = buffer_start - idx  # (idx if idx >= 0 else 0)
        right_pad = (idx + sequence_length) - buffer_end
        sample_start = left_pad
        sample_end = sequence_length - right_pad
        indices.append(
            [
                buffer_start,
                buffer_end,
                sample_start,
                sample_end,
            ]
        )
    return np.array(indices)


def sample_sequence_from_trial(
    trial_data: Dict[str, zarr.Array],
    sequence_length: int,
    buffer_start: int,
    buffer_end: int,
    sample_start: int,
    sample_end: int,
    image_size: int = 224,
) -> Dict[str, torch.Tensor]:
    """
    Given a trial’s zarr datasets and a set of indices, sample a fixed-length sequence.
    For non-image data, we convert the slice into a torch.Tensor and pad by repeating the
    first or last frame if needed. For images, we decode JPEG bytes and process them into
    a 4D tensor (View, C, H, W) per time step; then these are stacked into a tensor of shape
    (T, View, C, H, W).

    Returns a dictionary with keys 'agent_pos', 'action', and 'image'.
    """
    result = {}

    # Helper: pad a 2D tensor along the time dimension.
    def pad_sequence(
        seq: torch.Tensor,
        sample_start: int,
        sample_end: int,
        seq_len: int,
    ) -> torch.Tensor:
        # seq is of shape (n, ...), n = buffer_end - buffer_start
        if seq.shape[0] == seq_len:
            return seq
        padded = torch.zeros((seq_len,) + seq.shape[1:], dtype=seq.dtype)
        # Left pad: fill missing entries with first frame.
        if sample_start > 0:
            padded[:sample_start] = seq[0:1].expand(sample_start, *seq.shape[1:])
        # Right pad: fill missing entries with last frame.
        if sample_end < seq_len:
            padded[sample_end:] = seq[-1:].expand(seq_len - sample_end, *seq.shape[1:])
        # Place the valid slice.
        padded[sample_start:sample_end] = seq
        return padded

    # Process non-image modalities.
    for key in ["proprio", "actions"]:
        # Read slice from zarr; convert to numpy then torch.
        # Assume the zarr arrays are structured so that numeric conversion is possible.
        data_slice = trial_data[key][buffer_start:buffer_end]

        # Turn into a standard np array instead of a structured array
        # timestamp field removed because different data type
        # TODO: consider making this less hard coded? lmao
        if key == "proprio":
            fields_to_extract = [
                "x_pos",
                "y_pos",
                "z_pos",
                "roll",
                "pitch",
                "yaw",
                "gripper_width",
            ]
        else:
            fields_to_extract = [
                "x_vel",
                "y_vel",
                "z_vel",
                "roll",
                "pitch",
                "yaw",
                "gripper",
            ]

        data_slice = np.stack(
            [data_slice[field] for field in fields_to_extract],
            axis=-1,
        )

        tensor = torch.from_numpy(np.array(data_slice))

        tensor = pad_sequence(
            tensor,
            sample_start,
            sample_end,
            sequence_length,
        )
        # Rename keys to match expected model inputs. For example, 'proprio' -> 'agent_pos'
        if key == "proprio":
            result["agent_pos"] = tensor
        else:
            result["action"] = tensor

    # Process images.
    # For images, trial_data["images"] has shape (N, num_views); each element is JPEG bytes.
    images_slice = trial_data["images"][buffer_start:buffer_end]  # shape: (n, num_views)
    # For each time step in the slice, decode all camera views.
    decoded_frames = []
    for frame in images_slice:
        # frame is a sequence (or array) of JPEG bytes for each view.
        views = []
        for jpeg_bytes in frame:
            img_tensor = decode_and_process_image(jpeg_bytes, size=image_size)
            views.append(img_tensor)
        # Stack views along a new dimension -> shape (num_views, C, H, W)
        views_tensor = torch.stack(views, dim=0)
        decoded_frames.append(views_tensor)
    # Now stack along the time dimension: shape (n, num_views, C, H, W)
    images_tensor = torch.stack(decoded_frames, dim=0)
    # Pad images along time if needed.
    if images_tensor.shape[0] != sequence_length:
        # For images, pad by repeating first/last frame.
        pad_left = sample_start
        pad_right = sequence_length - sample_end
        if pad_left > 0:
            left_pad = images_tensor[0:1].expand(pad_left, -1, -1, -1, -1)
        else:
            left_pad = torch.empty(0, *images_tensor.shape[1:])
        if pad_right > 0:
            right_pad = images_tensor[-1:].expand(pad_right, -1, -1, -1, -1)
        else:
            right_pad = torch.empty(0, *images_tensor.shape[1:])
        images_tensor = torch.cat([left_pad, images_tensor, right_pad], dim=0)
    result["image"] = images_tensor  # shape: (sequence_length, num_views, C, H, W)

    return result
