from diffusion_policy.utils import ZarrDataConfig

config = ZarrDataConfig(
    data_dir="/Users/andywang/Projects/KaoLab/data/zarr_test",
    save_dir="/Users/andywang/Projects/KaoLab/runs",
    # parameters
    views=(0, 1),
    pred_horizon=24,
    obs_horizon=12,
    action_horizon=12,
    # |o|o|                             observations: 2,
    # | |a|a|a|a|a|a|a|a|               actions executed: 8,
    # |p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16,
    batch_size=4,
    # ResNet18 has output dim of 512,
    vision_feature_dim=512,
    # (x, y, z, r, p, y, g)
    lowdim_obs_dim=7,
    # observation feature has 514 dims in total per step,
    # obs_dim = vision_feature_dim * len(views) + lowdim_obs_dim,
    action_dim=7,
    num_diffusion_iters=100,
    num_epochs=2000,
)
