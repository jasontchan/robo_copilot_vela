from typing import List

import torch
import torch.nn as nn

from diffusion_policy.models import ConditionalUnet1D, TransformerForDiffusion, get_resnet, replace_bn_with_gn

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def get_vision_features(img_batch: torch.Tensor, encoders: List[nn.Module]) -> torch.Tensor:
    B, obs_horizon, views, C, H, W = img_batch.size()
    features_list: list[torch.Tensor] = []
    for view in range(views):
        img: torch.Tensor = img_batch[:, :, view]  # (B, obs_horizon, C, H, W)
        flat_img: torch.Tensor = img.flatten(end_dim=1)  # (B * obs_horizon, C, H, W)
        flat_features: torch.Tensor = encoders[view](flat_img)  # (B * obs_horizon, vision_feature_dims)
        features: torch.Tensor = flat_features.reshape(B, obs_horizon, -1)  # (B, obs_horizon, vision_feature_dims)
        features_list.append(features)
    return torch.cat(features_list, dim=-1)  # (B, obs_horizon, views * vision_feature_dims)


def get_condition(proprio_batch: torch.Tensor, img_batch: torch.Tensor, encoders: List[nn.Module]) -> torch.Tensor:
    vision_features: torch.Tensor = get_vision_features(img_batch, encoders)
    condition: torch.Tensor = torch.cat([vision_features, proprio_batch], dim=-1)  # (B, obs_horizon, obs_dim)
    return condition  # (B, obs_horizon, obs_dim)


O = 5
D = 2
P = 2 * O + D
views = (0, 1)
action_dim = 7
obs_dim = len(views) * 512 + 7

cond_dim = obs_dim * O

encoders = []
for view in views:
    encoder = get_resnet("resnet18")
    encoder = replace_bn_with_gn(encoder)
    encoders.append(encoder.to(device))

unet = ConditionalUnet1D(input_dim=action_dim, n_obs_steps=O, cond_dim=obs_dim).to(device)
transformer = TransformerForDiffusion(input_dim=action_dim, output_dim=action_dim, horizon=P, n_obs_steps=O, cond_dim=obs_dim, obs_as_cond=True).to(device)
opt = transformer.configure_optimizers()
model = unet


fake_prop: torch.Tensor = torch.randn(1, O, action_dim, dtype=torch.float, device=device)
fake_imgs: torch.Tensor = torch.randn(1, O, len(views), 3, 224, 224, dtype=torch.float, device=device)
fake_cond: torch.Tensor = get_condition(fake_prop, fake_imgs, encoders)
fake_samp: torch.Tensor = torch.randn(1, P, action_dim, dtype=torch.float, device=device)

print(f"{fake_samp.shape = }")  # (1, 12, 7)
print(f"{fake_cond.shape = }")  # (1, 5, 1031)

output = model(fake_samp, torch.tensor(0), fake_cond)
output.shape
