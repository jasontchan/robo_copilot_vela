"""
Multicam implementation of Diffusion Policy

Main difference:
Images are now 5D: (Batch, View, C, H, W)
"""

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from torchvision.io import read_image


def prep_imgs(episode: str, views: tuple = ("cam0", "cam1"), size: int = 224):
    """
    Prepare image data from multiple camera views and stack them into a tensor.

    This function reads images from specified view folders within an episode
    directory, sorts them numerically based on the digits in the filename, removes
    any extra channels (keeping only the first 3 channels), resizes the images,
    and then stacks them into a tensor of shape (Batch, View, C, H, W).

    Args:
        episode (str): Path to the episode folder.
        views (tuple): Tuple of folder names corresponding to each camera view.
        size (int): Desired width and height to resize images (size x size).

    Returns:
        torch.Tensor: A tensor containing the processed images with shape
                      (Batch, View, C, H, W).
    """
    img_index = pd.read_csv(f"{episode}/image_paths.csv")
    resize_transform = T.Resize((size, size))
    view_batches = []  # Will hold the image tensors for each view

    # Process images for each view folder
    for view in views:
        img_paths = img_index[view]
        images = []  # Collect images for the current view
        for img_path in img_paths:
            # Read image and remove any alpha channel by taking only the first 3 channels
            img = read_image(str(img_path))[:3]
            # Resize the image
            img = resize_transform(img)
            images.append(img.float())

        # Stack images for this view to form a tensor of shape (Batch, C, H, W)
        view_tensor = torch.stack(images, dim=0)
        view_batches.append(view_tensor)

    # Stack all view tensors to get shape (View, Batch, C, H, W)
    stacked_views = torch.stack(view_batches, dim=0)
    # Transpose to have the batch dimension first: (Batch, View, C, H, W)
    final_tensor = stacked_views.transpose(0, 1)

    return final_tensor


def prepare_train_data(root_dir: str, views: tuple) -> tuple:
    """
    Prepares training data by reading proprioceptive, action, and image data from multiple episodes.

    For each episode (i.e., subdirectory in `root_dir`), the function reads:
        - Proprioceptive data from 'proprioceptive.csv'
        - Action data from 'action.csv'
        - Image data using the `prep_imgs` function, which returns a 5D tensor
          of shape (Batch, View, C, H, W).

    The CSV files are expected to have a 'time' column that is dropped before converting
    to a tensor. The function asserts that the number of time steps (rows in the CSVs and
    the batch dimension of the image tensor) are equal for all modalities.

    After processing each episode, the data is concatenated along the time dimension, and
    cumulative episode lengths are computed, which can later be used to segment the data.

    Args:
        root_dir (str): The root directory containing episode subdirectories.

    Returns:
        tuple: A tuple containing:
            - episode_ends (np.ndarray): 1D array with the cumulative lengths of episodes.
            - train_data (dict): A dictionary with keys:
                'agent_pos': torch.Tensor of proprioceptive data,
                'action': torch.Tensor of action data,
                'image': torch.Tensor of image data with shape (Total_Batch, View, C, H, W).
    """
    # Convert the root directory to a Path object
    root = Path(root_dir)
    episode_lens = []  # List to store the number of time steps per episode.
    pos_list, act_list, img_list = [], [], []

    # Iterate over each subdirectory in the root directory.
    for episode_dir in root.iterdir():
        # Skip non-directories and hidden/system directories (e.g., '.DS_Store')
        if not episode_dir.is_dir() or episode_dir.name.startswith("."):
            continue

        # Read proprioceptive data from 'proprioceptive.csv'
        proprioceptive_path = episode_dir / "proprioceptives.csv"
        proprioceptive_df = pd.read_csv(proprioceptive_path).astype("float32").drop("time", axis=1)
        proprioceptive = torch.from_numpy(proprioceptive_df.values)

        # Read action data from 'action.csv'
        action_path = episode_dir / "actions.csv"
        action_df = pd.read_csv(action_path).astype("float32").drop("time", axis=1)
        action = torch.from_numpy(action_df.values)

        # Process image data using the prep_imgs function.
        # Note: Convert the Path object to string if prep_imgs expects a string path.
        images = prep_imgs(str(episode_dir), views)

        # Verify that all modalities have the same number of time steps.
        batch_len = len(proprioceptive)
        assert batch_len == len(action) == len(images), (
            f"Data length mismatch in episode {episode_dir.name}: proprioceptive({len(proprioceptive)}), action({len(action)}), images({len(images)})"
        )

        # Record the length and append the data for this episode.
        episode_lens.append(batch_len)
        pos_list.append(proprioceptive)
        act_list.append(action)
        img_list.append(images)

    # Compute cumulative episode lengths; useful for later segmenting the data.
    episode_ends = np.cumsum(episode_lens)

    # Concatenate data from all episodes along the first dimension (time steps).
    train_data = {
        "agent_pos": torch.cat(pos_list),
        "action": torch.cat(act_list),
        "image": torch.cat(img_list),  # Expected shape: (Total_Batch, View, C, H, W)
    }

    return episode_ends, train_data


def create_sample_indices(
    episode_ends: np.ndarray,  # cumulative end indices of each episode
    sequence_length: int,  # window length of each subsequence
    pad_before: int = 0,  # num frames of allowed padding before start
    pad_after: int = 0,  # num frames of allowed padding after end
) -> np.ndarray:  # [buffer_start, buffer_end, sample_start, sample_end]
    """
    Generate indices for slicing subsequences from a dataset with episodic boundaries.

    This function enumerates all valid (and partially valid) windows of length
    `sequence_length` across multiple episodes. Negative or beyond-episode starts
    are allowed by specifying `pad_before` and `pad_after`, which enable sampling
    frames prior to episode start or after episode end (to be handled later via padding).

    Parameters
    ----------
    episode_ends : np.ndarray
        1D array of shape `(num_episodes,)` specifying the *cumulative* end index of
        each episode. For example, if you have two episodes of length 5 and 6, then
        `episode_ends = [5, 11]`.
    sequence_length : int
        The length of each subsequence to be sampled.
    pad_before : int, optional
        Number of frames to allow sampling *before* the nominal start of an episode
        (defaults to 0).
    pad_after : int, optional
        Number of frames to allow sampling *beyond* the nominal end of an episode
        (defaults to 0).

    Returns
    -------
    np.ndarray
        A 2D array of shape `(N, 4)`, where each row is `[buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx]`.

        - `buffer_start_idx, buffer_end_idx` define the slice of the actual dataset
          (in the range `[start_idx, end_idx)` ).
        - `sample_start_idx, sample_end_idx` define where that slice should be placed
          in a final subsequence array of length `sequence_length` (in `[0, sequence_length)` ).

    Notes
    -----
    The four indices returned in each row are used by `sample_sequence` to slice
    and pad data properly. `N` is the total number of valid (or partially valid)
    subsequences across all episodes.
    """
    indices: list = []
    # for each episode:
    for i in range(len(episode_ends)):
        # determine the start_idx and end_idx in the global dataset
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i - 1]  # end of previous episode
        end_idx = episode_ends[i]
        # num frames in current episode
        episode_length = end_idx - start_idx
        # figure out all possible starting offsets
        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after
        # for each possible starting offset:
        for idx in range(min_start, max_start + 1):
            # these idxs define slice of actual dataset array
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
            # how much padding offset do we need?
            start_offset = buffer_start_idx - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
            # where this slice belongs to in the final subsequence window
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            # collect outputs
            indices.append([buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx])
    # return as np.ndarray
    return np.array(indices)


def sample_sequence(
    train_data: Dict[str, torch.Tensor],
    sequence_length: int,
    buffer_start_idx: int,
    buffer_end_idx: int,
    sample_start_idx: int,
    sample_end_idx: int,
) -> Dict[str, torch.Tensor]:
    """
    Extract a padded subsequence of fixed length from the dataset using the provided slice indices.

    This function slices each tensor in `train_data` between `buffer_start_idx` and
    `buffer_end_idx`, then places that slice into a new tensor of length `sequence_length`.
    If the slice extends before the start or beyond the end (as indicated by
    `sample_start_idx` or `sample_end_idx`), the missing frames are filled by
    repeating the first or last valid frame.

    Parameters
    ----------
    train_data : Dict[str, torch.Tensor]
        A dictionary where each key maps to a PyTorch tensor of shape `(N, ...)`,
        representing a time series or frame-based data. For example:
          - `train_data["agent_pos"]` might be of shape `(N, 2)`,
          - `train_data["image"]` might be of shape `(N, 3, 96, 96)`,
          - etc.
    sequence_length : int
        The fixed length of the subsequence to return.
    buffer_start_idx : int
        Start index of the slice within the dataset tensors.
    buffer_end_idx : int
        End index (exclusive) of the slice within the dataset tensors.
    sample_start_idx : int
        The offset in the final subsequence tensor where the slice should begin.
    sample_end_idx : int
        The offset in the final subsequence tensor where the slice should end
        (exclusive).

    Returns
    -------
    Dict[str, torch.Tensor]
        A dictionary with the same keys as `train_data`. Each value is now of shape
        `(sequence_length, ...)`, containing the requested slice plus any necessary
        padding at the start or end.

    Notes
    -----
    - Frames falling outside the valid dataset range (due to negative start or
      beyond-episode end) are filled by repeating the first or last real frame.
    - This function is typically used in conjunction with `create_sample_indices`
      to ensure the correct slicing and padding for each subsequence.
    """
    result = {}

    for key, input_tensor in train_data.items():
        # 1) Slice the data (PyTorch slicing)
        sample = input_tensor[buffer_start_idx:buffer_end_idx]
        data = sample

        # 2) Check if we need padding
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            # 2a) Create a zero-filled tensor of the final desired size
            data = torch.zeros(
                (sequence_length,) + input_tensor.shape[1:],
                dtype=input_tensor.dtype,
                device=input_tensor.device,
            )

            # 2b) Left padding
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]

            # 2c) Right padding
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]

            # 2d) Place the real sample in the correct slice
            data[sample_start_idx:sample_end_idx] = sample

        result[key] = data

    return result


def get_data_stats(data: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Compute per-dimension min and max of a PyTorch tensor, ignoring all but
    the last dimension in the final stat computation.

    Specifically, it flattens the tensor to shape `(-1, data.shape[-1])`,
    then computes the min and max along dimension 0.

    Parameters
    ----------
    data : torch.Tensor
        A tensor of shape `(N, ..., D)` where `D` is the dimensionality
        of interest (e.g. positions). The function will flatten all but
        the last dimension.

    Returns
    -------
    Dict[str, torch.Tensor]
        A dictionary with two keys:
        - `'min'`: shape `(D,)`, the per-dimension minimum values.
        - `'max'`: shape `(D,)`, the per-dimension maximum values.
    """
    # Flatten everything except the last dimension
    data_reshaped = data.reshape(-1, data.shape[-1])

    # Compute min/max along the first dimension of the reshaped tensor
    min_vals = data_reshaped.min(dim=0).values
    max_vals = data_reshaped.max(dim=0).values

    stats = {"min": min_vals, "max": max_vals}
    return stats


def normalize_data(data: torch.Tensor, stats: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Normalize a PyTorch tensor to the range `[-1, 1]` using provided min/max stats.

    The steps are:
      1. Shift data so that 'min' maps to 0.
      2. Scale so that 'max' maps to 1.
      3. Convert range [0, 1] to [-1, 1].

    Parameters
    ----------
    data : torch.Tensor
        The original data of shape `(N, ..., D)`.
    stats : Dict[str, torch.Tensor]
        A dictionary with keys 'min' and 'max' (each of shape `(D,)`),
        typically obtained from `get_data_stats`.

    Returns
    -------
    torch.Tensor
        The normalized data, same shape as `data`, but with values
        in the range `[-1, 1]`.

    Notes
    -----
    - When normalizing to [0, 1], features with constant values (e.g. gripper_state)
      will encounter division-by-zero (since min = max). Clamping used as suboptimal fix.
    """
    # ndata in [0, 1]
    range_val = stats["max"] - stats["min"]
    # clamp to avoid division by zero
    range_val = torch.clamp(range_val, min=1e-7)
    ndata = (data - stats["min"]) / range_val
    # now scale to [-1, 1]
    ndata = ndata * 2.0 - 1.0
    return ndata


def unnormalize_data(ndata: torch.Tensor, stats: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Undo the `[-1, 1]` normalization to recover the original data range.

    The steps are:
      1. Shift `[-1, 1]` range back to `[0, 1]`.
      2. Rescale `[0, 1]` to [min, max].

    Parameters
    ----------
    ndata : torch.Tensor
        The normalized data, shape `(N, ..., D)`, in the range `[-1, 1]`.
    stats : Dict[str, torch.Tensor]
        A dictionary with keys 'min' and 'max' (each of shape `(D,)`),
        typically obtained from `get_data_stats`.

    Returns
    -------
    torch.Tensor
        The unnormalized data, same shape as `ndata`, back in the original
        value range described by `stats['min']` and `stats['max']`.
    """
    # Shift [-1, 1] back to [0, 1]
    ndata_01 = (ndata + 1.0) / 2.0
    # Rescale [0, 1] to [min, max]
    data = ndata_01 * (stats["max"] - stats["min"]) + stats["min"]
    return data


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def test_prep_imgs_with_plot(
        example_episode="/Users/andywang/Projects/KaoLab/data/2024_11_14_17_50_31",
    ):
        """
        Test the prep_imgs function on an example episode directory and plot a few images.

        Replace 'path/to/example_episode' with the path to your episode folder.
        This function prints the shape of the processed tensor (should be 5D: Batch, View, C, H, W)
        and displays a grid of images where each row corresponds to a time step (batch index)
        and each column corresponds to a camera view.
        """
        # Process the images from the example episode
        processed_tensor = prep_imgs(example_episode)
        print("Processed tensor shape (Batch, View, C, H, W):", processed_tensor.shape)

        # Sanity check: ensure that the output is a 5D tensor
        assert processed_tensor.ndim == 5, "Output tensor must be 5-dimensional."

        batch_dim, view_dim, _, _, _ = processed_tensor.shape
        num_to_plot = min(3, batch_dim)  # Plot at most 3 time steps (batch elements)

        # Create a grid of subplots: rows = time steps, columns = views
        fig, axes = plt.subplots(num_to_plot, view_dim, figsize=(4 * view_dim, 4 * num_to_plot))

        # If there's only one row, ensure axes is iterable as a list
        if num_to_plot == 1:
            axes = [axes]

        for b in range(num_to_plot):
            for v in range(view_dim):
                # Extract the image tensor for the given batch index and view.
                # The tensor shape is (C, H, W), so we transpose to (H, W, C) for plotting.
                img_tensor = processed_tensor[b, v]
                img_np = img_tensor.permute(1, 2, 0).cpu().numpy()

                # If the image is in float but in the range 0-255, convert to uint8 for proper display.
                if img_np.dtype != "uint8":
                    img_np = img_np.astype("uint8")

                # Plot the image.
                ax = axes[b][v] if num_to_plot > 1 else axes[v]
                ax.imshow(img_np)
                ax.set_title(f"Batch {b} - View {v}")
                ax.axis("off")

        plt.tight_layout()
        plt.show()

    test_prep_imgs_with_plot()
