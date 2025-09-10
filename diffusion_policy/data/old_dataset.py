'''
Multicam implementation of Diffusion Policy

Main difference:
Images are now 5D: (Batch, View, C, H, W)
'''
from diffusion_policy.utils import (
    prepare_train_data,
    create_sample_indices,
    sample_sequence,
    get_data_stats,
    normalize_data,
    unnormalize_data
)
from torch.utils.data import Dataset


class BlockPushDataset(Dataset):
    def __init__(self, 
                 dataset_path: str, views: tuple,
                 pred_horizon: int, obs_horizon: int, action_horizon: int):
        episode_ends, train_data = prepare_train_data(dataset_path, views)
        # compute start and end of each state-action sequence
        # also handles padding
        indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=pred_horizon,
            pad_before=obs_horizon-1,
            pad_after=action_horizon-1)

        # compute statistics and normalized data to [-1,1]
        stats = {}
        normalized_train_data = {}
        for key, data in train_data.items():
            stats[key] = get_data_stats(data)
            normalized_train_data[key] = normalize_data(data, stats[key])

        # images are already normalized
        #normalized_train_data['image'] = train_data['image']

        self.indices = indices
        self.stats = stats
        self.normalized_train_data = normalized_train_data
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # get the start/end indices for this datapoint
        buffer_start_idx, buffer_end_idx, \
            sample_start_idx, sample_end_idx = self.indices[idx]

        # get nomralized data using these indices
        nsample = sample_sequence(
            train_data=self.normalized_train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx
        )

        # discard unused observations
        nsample['image'] = nsample['image'][:self.obs_horizon,:]
        nsample['agent_pos'] = nsample['agent_pos'][:self.obs_horizon,:]
        return nsample
