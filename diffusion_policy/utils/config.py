"""
Multicam implementation of Diffusion Policy
"""

from dataclasses import dataclass
from typing import Any, Dict

model_map: Dict[str, Any] = {
    "convolutionalunet1d": "unet",
    "conditionalunet1d": "unet",
    "conv": "unet",
    "unet": "unet",
    "transformer": "transformer",
    "transformerfordiffusion": "transformer",
    "diffusiontransformer": "transformer",
    "diffusiongpt": "transformer",
    "gpt": "transformer",
}


@dataclass
class DiffusionPolicyConfig:
    # Training
    data_dir: str
    save_dir: str
    batch_size: int
    num_epochs: int

    # Policy Parameters
    model_type: str
    num_diffusion_iters: int
    inference_delay: int
    obs_horizon: int

    # Try not to touch these
    views: tuple = (0, 1)
    vision_feature_dim: int = 512  # ResNet18 has output dim of 512
    proprio_dim: int = 7  # (x, y, z, r, p, y, g)
    action_dim: int = 7

    @property
    def model(self):
        normalized: str = self.model_type.strip().lower()
        try:
            return model_map[normalized]
        except KeyError:
            raise ValueError(f"Invalid model_type '{self.model_type}'. Must be one of: {[k for k in model_map.keys()]}")

    @property
    def obs_dim(self) -> int:
        return self.vision_feature_dim * len(self.views) + self.proprio_dim

    @property
    def pred_horizon(self) -> int:
        return 2 * self.obs_horizon + self.inference_delay

    @property
    def action_horizon(self) -> int:
        return self.obs_horizon + self.inference_delay

    def as_dict(self) -> Dict[str, Any]:
        return {
            "data_dir": self.data_dir,
            "save_dir": self.save_dir,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "model_type": self.model_type,
            "num_diffusion_iters": self.num_diffusion_iters,
            "inference_delay": self.inference_delay,
            "obs_horizon": self.obs_horizon,
            "views": self.views,
            "vision_feature_dim": self.vision_feature_dim,
            "proprio_dim": self.proprio_dim,
            "action_dim": self.action_dim,
            "obs_dim": self.obs_dim,
            "pred_horizon": self.pred_horizon,
            "action_horizon": self.action_horizon,
            "model": self.model,
        }


@dataclass
class ZarrDataConfig:
    data_dir: str = "/home/andy/data/random_init/train"
    save_dir: str = "/home/andy/robo_copilot/runs"
    views: tuple = (0, 1)
    # parameters
    pred_horizon: int = 16
    obs_horizon: int = 2
    action_horizon: int = 8
    # |o|o|                             observations: 2
    # | |a|a|a|a|a|a|a|a|               actions executed: 8
    # |p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16
    batch_size: int = 64
    # ResNet18 has output dim of 512
    vision_feature_dim: int = 512
    # (x, y, z, r, p, y, g)
    lowdim_obs_dim: int = 7
    # observation feature has 514 dims in total per step
    obs_dim: int = vision_feature_dim * len(views) + lowdim_obs_dim
    action_dim: int = 7

    num_diffusion_iters: int = 100
    num_epochs: int = 1  # 000


@dataclass
class BlockPushConfig:
    data_dir: str = "/home/andy/data/random_init/train"
    save_dir: str = "/home/andy/robo_copilot/runs"
    views: tuple = ("cam0_left", "cam1_left", "mini_left")
    # parameters
    pred_horizon: int = 16
    obs_horizon: int = 2
    action_horizon: int = 8
    # |o|o|                             observations: 2
    # | |a|a|a|a|a|a|a|a|               actions executed: 8
    # |p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16
    batch_size: int = 64
    # ResNet18 has output dim of 512
    vision_feature_dim: int = 512
    # agent_pos is 2 dimensional
    lowdim_obs_dim: int = 4
    # observation feature has 514 dims in total per step
    obs_dim: int = vision_feature_dim * len(views) + lowdim_obs_dim
    action_dim: int = 4

    num_diffusion_iters: int = 100
    num_epochs: int = 1  # 000
