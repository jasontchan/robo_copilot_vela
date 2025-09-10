from diffusion_policy.utils import DiffusionPolicyConfig

config = DiffusionPolicyConfig(
    # Training
    data_dir="/home/andy/data/choose_block_all",
    save_dir="/home/andy/robo_copilot/runs",
    batch_size=16,
    num_epochs=1000,
    # Policy Parameters,
    model_type="transformer",
    num_diffusion_iters=100,
    inference_delay=4,
    obs_horizon=6,
)
