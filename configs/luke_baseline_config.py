
from diffusion_policy.utils import DiffusionPolicyConfig

config = DiffusionPolicyConfig(
    # Training
    data_dir="/home/andy/data/choose_block_all",
    save_dir="/home/andy/robo_copilot/runs",
    batch_size=64,
    num_epochs=300,
    # Policy Parameters,
    model_type="unet",
    num_diffusion_iters=100,
    inference_delay=2,
    obs_horizon=1,
)

