"""
Data loading pipeline for unified Zarr stores.

Each trial is stored under a directory (e.g. trial_1.zarr, trial_2.zarr, ...), and each trial
contains 3 datasets: 'images', 'proprio', and 'actions'. This pipeline builds sample indices
for sequences of fixed length (pred_horizon) given padding parameters (obs_horizon, action_horizon), and it loads only the required chunks from disk. """

from pathlib import Path
from typing import List, Tuple

import zarr
from torch.utils.data import Dataset

from diffusion_policy.utils import create_sample_indices_from_trial, sample_sequence_from_trial

##############################
# Dataset Class for Zarr Data #
##############################


class ZarrTrialDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        pred_horizon: int,
        obs_horizon: int,
        action_horizon: int,
        image_size: int = 224,
    ):
        """
        Args:
            root_dir (str): Path to the root folder containing trial directories (each ending in .zarr).
            pred_horizon (int): Total sequence length (must be an integer multiple of the chunk size).
            obs_horizon (int): Number of frames used for observation (at the start of the sequence).
            action_horizon (int): Number of action steps executed.
            image_size (int): Resize images to this size.
        """
        self.root_dir = Path(root_dir)
        # List all trial directories (assume directories ending with .zarr)
        self.trial_paths = sorted([p for p in self.root_dir.iterdir() if p.suffix == ".zarr"])
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.image_size = image_size

        # For each trial, open the zarr store and compute sample indices.
        # self.trials: List[Dict[str, zarr.Array]] = []  # each element: dict with keys "images", "proprio", "actions"
        self.sample_index: List[Tuple[int, int, int, int, int]] = []
        # Each entry in sample_index: (trial_idx, buffer_start, buffer_end, sample_start, sample_end)

        for trial_idx, trial_path in enumerate(self.trial_paths):
            # Open the zarr group in read-only mode.
            store = zarr.DirectoryStore(str(trial_path))
            group = zarr.open_group(store, mode="r")
            # trial_data = {
            #     "images": group["images"],
            #     "proprio": group["proprio"],
            #     "actions": group["actions"],
            # }
            # self.trials.append(trial_data)
            # Determine trial length from one of the datasets (they are all the same length).
            trial_length = group["actions"].shape[0]
            # print(f"{group['proprio'].shape[0] = }")
            # print(f"{group['actions'].shape[0] = }")
            # For each trial, create sample indices.
            # Here, we allow padding such that the observation window (obs_horizon) is at the start.
            indices = create_sample_indices_from_trial(
                trial_length=trial_length,
                sequence_length=pred_horizon,
                pad_before=obs_horizon - 1,
                pad_after=action_horizon - 1,
            )
            # if trial_idx == 25:
                # print(f"-25-, {trial_length = } {indices = }")
            # For each sample in this trial, store trial index and the indices.
            for row in indices:
                buffer_start, buffer_end, sample_start, sample_end = row
                self.sample_index.append((trial_idx, buffer_start, buffer_end, sample_start, sample_end))

    def __len__(self):
        return len(self.sample_index)

    def __getitem__(self, idx):
        trial_idx, buffer_start, buffer_end, sample_start, sample_end = self.sample_index[idx]
        # trial_data = self.trials[trial_idx]
        store = zarr.DirectoryStore(self.trial_paths[trial_idx])
        group = zarr.open_group(store, mode="r")
        trial_data = {
            "images": group["images"],
            "proprio": group["proprio"],
            "actions": group["actions"],
        }

        # local_rank = int(os.environ['LOCAL_RANK'])
        # print(f"{idx = }, {trial_data['actions'].shape = }, {buffer_start = }, {buffer_end = }")
        # print(f"{trial_idx = }")
        sample = sample_sequence_from_trial(
            trial_data,
            sequence_length=self.pred_horizon,
            buffer_start=buffer_start,
            buffer_end=buffer_end,
            sample_start=sample_start,
            sample_end=sample_end,
            image_size=self.image_size,
        )
        # The model expects a sequence of frames for observation.
        # For images and agent_pos, we take only the first obs_horizon frames.
        sample["image"] = sample["image"][: self.obs_horizon]
        sample["agent_pos"] = sample["agent_pos"][: self.obs_horizon]
        return sample

