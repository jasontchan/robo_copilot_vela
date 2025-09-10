from diffusion_policy.utils import DiffusionPolicyConfig

config = DiffusionPolicyConfig(
    # Training
    data_dir="/home/andy/data/all_drawers",
    save_dir="/home/andy/robo_copilot/runs",
    batch_size=16,
    num_epochs=500,
    # Policy Parameters,
    model_type="unet",
    num_diffusion_iters=100,
    inference_delay=2,
    obs_horizon=1,
)
