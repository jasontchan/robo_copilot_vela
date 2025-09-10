from diffusion_policy.utils import BlockPushConfig


config = BlockPushConfig(
    data_dir = "/home/andy/data/random_init/train",
    save_dir = "/home/andy/robo_copilot/runs",
    views = ("cam0_left", "cam1_left", "mini_left"),
    # parameters
    pred_horizon = 16,
    obs_horizon = 2,
    action_horizon = 8,
    #|o|o|                             observations: 2,
    #| |a|a|a|a|a|a|a|a|               actions executed: 8,
    #|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16,
    batch_size = 64,
    # ResNet18 has output dim of 512,
    vision_feature_dim = 512,
    # agent_pos is 2 dimensional,
    lowdim_obs_dim = 4,
    # observation feature has 514 dims in total per step,
    #obs_dim = vision_feature_dim * len(views) + lowdim_obs_dim,
    action_dim = 4,
    num_diffusion_iters = 100,
    num_epochs = 2000,
)
