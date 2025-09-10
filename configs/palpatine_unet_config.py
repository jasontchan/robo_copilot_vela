from diffusion_policy.utils import DiffusionPolicyConfig

config = DiffusionPolicyConfig(
    # Training
    data_dir="/home/palpatine/data/panda/choose_block_all",
    save_dir="/home/palpatine/git/robo_copilot/runs",
    batch_size=16,
    num_epochs=1000,
    # Policy Parameters,
    model_type="unet",
    num_diffusion_iters=100,
    inference_delay=4,
    obs_horizon=6,
)
