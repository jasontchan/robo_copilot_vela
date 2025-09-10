"""
inference.py

FOR AUTONOMOUS INFERENCE, MAKE SEPARATE FILE FOR SHARED AUTONOMY

A simple inference pipeline for the diffusion policy.
It defines functions to load pre-trained models, load dataset statistics,
preprocess incoming data, and generate action predictions via reverse diffusion.
The generate_actions() function can run a full reverse diffusion (starting from pure noise)
or a partial reverse diffusion (starting from a provided, partially noised action).
"""

from collections import OrderedDict
from typing import Deque

import numpy as np
import torch
import torch.nn as nn

from diffusion_policy.models import ConditionalUnet1D, get_resnet, replace_bn_with_gn
from diffusion_policy.models.transformer import TransformerForDiffusion
from diffusion_policy.utils import DiffusionPolicyConfig


def initialize_networks(config: DiffusionPolicyConfig) -> nn.ModuleDict:
    """
    Initialize the noise prediction network and vision encoders for each camera view.

    Args:
        config (ZarrDataConfig): Configuration object with attributes such as 'action_dim', 'obs_dim', 'obs_horizon', and 'views'.

    Returns:
        nn.ModuleDict: Dictionary of network modules.
    """
    nets: nn.ModuleDict = nn.ModuleDict()
    if config.model == "unet":
        nets["noise_pred_net"] = ConditionalUnet1D(input_dim=config.action_dim, n_obs_steps=config.obs_horizon, cond_dim=config.obs_dim)
    elif config.model == "transformer":
        nets["noise_pred_net"] = TransformerForDiffusion(
            input_dim=config.action_dim,
            output_dim=config.action_dim,
            horizon=config.pred_horizon,
            n_obs_steps=config.obs_horizon,
            cond_dim=config.obs_dim,
            obs_as_cond=True,
        )
    else:
        raise ValueError(f"Invalid model type: {config.model_type}")
    for view in config.views:
        encoder = get_resnet("resnet18")
        encoder = replace_bn_with_gn(encoder)
        nets[f"vision_encoder_{view}"] = encoder
    return nets


def load_pretrained_nets(checkpoint_path, config, device=torch.device("cpu")):
    """
    Load pre-trained diffusion policy models.

    Args:
        checkpoint_path (str or Path): Path to the model checkpoint file.
        config (dict): Configuration dictionary for model architecture.
        device (torch.device): Device to load the model onto.

    Returns:
        nn.ModuleDict: Loaded model networks in evaluation mode.
    """
    nets = initialize_networks(config).to(device)
    state_dict = torch.load(str(checkpoint_path), map_location=device, weights_only=True)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v
    nets.load_state_dict(new_state_dict)
    nets.to(device)
    nets.eval()
    return nets


def deque2tensor(d: Deque[np.ndarray]) -> torch.Tensor:
    """
    Convert an observation double-ended queue to a torch tensor.

    Args:
        d (Deque[np.ndarray]): must be size obs_horizon of numpy arrays, which could represent proprioceptives or images.

    Returns:
        torch.Tensor: size (batch, obs_horizon, ...)
    """
    return torch.stack([torch.from_numpy(x) for x in d], dim=0).unsqueeze(0)


def get_vision_features(img_batch: torch.Tensor, nets: nn.ModuleDict) -> torch.Tensor:
    """
    Apply vision encoders to batch of multicam images

    Args:
        img_batch (torch.Tensor): size (batch, obs_horizon, views, C, H, W)
        nets (nn.ModuleDict): {..., "vision_encoder_{x}": ResNet18, ...}

    Returns:
        torch.Tensor: size (batch, obs_horizon, views * vision_feature_dims)
    """
    B, obs_horizon, views, C, H, W = img_batch.size()
    features_list: list[torch.Tensor] = []
    for view in range(views):
        img: torch.Tensor = img_batch[:, :, view]  # (B, obs_horizon, C, H, W)
        flat_img: torch.Tensor = img.flatten(end_dim=1)  # (B * obs_horizon, C, H, W)
        flat_features: torch.Tensor = nets[f"vision_encoder_{view}"](flat_img)  # (B * obs_horizon, vision_feature_dims)
        features: torch.Tensor = flat_features.reshape(B, obs_horizon, -1)  # (B, obs_horizon, vision_feature_dims)
        features_list.append(features)
    return torch.cat(features_list, dim=-1)  # (B, obs_horizon, views * vision_feature_dims)


def get_condition(proprio_batch: torch.Tensor, img_batch: torch.Tensor, nets: nn.ModuleDict) -> torch.Tensor:
    """
    Get global conditioning features for diffusion policy noise predictor network
    lowdim_obs_dim = 7
    vision_feature_dims = 512
    obs_dim = lowdim_obs_dim + views * vision_feature_dims
            = 7 + 2 * 512

    Args:
        proprio_batch (torch.Tensor): size (batch, obs_horizon, lowdim_obs_dim)
        img_batch (torch.Tensor): size (batch, obs_horizon, views, C, H, W)
        nets (nn.ModuleDict): {..., "vision_encoder_{x}": ResNet18, ...}

    Returns:
        torch.Tensor: size (batch, obs_horizon, obs_dim)
    """
    vision_features: torch.Tensor = get_vision_features(img_batch, nets)
    condition: torch.Tensor = torch.cat([vision_features, proprio_batch], dim=-1)  # (B, obs_horizon, obs_dim)
    return condition  # (B, obs_horizon, obs_dim)
