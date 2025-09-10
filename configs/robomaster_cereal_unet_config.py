from diffusion_policy.utils import DiffusionPolicyConfig

config = DiffusionPolicyConfig(
    # Training
    data_dir="/home/robomaster/data/cereal_making_formal_replay",
    save_dir="/home/robomaster/git/robo_copilot/runs",
    batch_size=16,
    num_epochs=100,
    # Policy Parameters,
    model_type="unet",
    num_diffusion_iters=100,
    inference_delay=4,
    obs_horizon=6,
)
