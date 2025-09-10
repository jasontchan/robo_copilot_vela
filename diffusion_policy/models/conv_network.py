"""
Multicam implementation of Diffusion Policy

This module defines a 1D UNet architecture `ConditionalUnet1D` used as the noise
prediction network for diffusion policy models. The network is composed of the following components:
- `SinusoidalPosEmb`: Positional encoding for the diffusion iteration k.
- `Downsample1d`: Strided convolution to reduce temporal resolution.
- `Upsample1d`: Transposed convolution to increase temporal resolution.
- `Conv1dBlock`: A block composed of Conv1d --> GroupNorm --> Mish activation.
- `ConditionalResidualBlock1D`: A residual block that takes two inputs (`x` and `cond`).
  The feature `x` is processed through two stacked `Conv1dBlock` layers (with a residual connection), while
  `cond` is used with FiLM conditioning (see https://arxiv.org/abs/1709.07871) to modulate the features.
"""

from typing import List, Optional, Union

import torch
import torch.nn as nn

from diffusion_policy.models.conv_blocks import ConditionalResidualBlock1D, Conv1dBlock, Downsample1d, SinusoidalPosEmb, Upsample1d
from diffusion_policy.models.model_mixin import ModelMixin


class ConditionalUnet1D(ModelMixin):
    """
    ConditionalUnet1D implements a conditional U-Net architecture for 1D signals
    with diffusion step embedding and FiLM conditioning.

    The network takes an input sample, encodes the diffusion time step using a sinusoidal
    positional embedding (with further processing), and optionally concatenates a global conditioning vector.
    The architecture follows a U-Net structure, with a downsampling path, two middle layers,
    and an upsampling path. The intermediate features are modulated with the conditioning information
    via conditional residual blocks.
    """

    def __init__(
        self,
        input_dim: int,
        n_obs_steps: int,
        cond_dim: int,
        diffusion_step_embed_dim: int = 256,
        down_dims: List[int] = [256, 512, 1024],
        kernel_size: int = 5,
        n_groups: int = 8,
    ) -> None:
        """
        Initialize the ConditionalUnet1D module.

        Args:
            input_dim (int): Dimensionality of the input actions.
            global_cond_dim (int): Dimensionality of the global conditioning vector. Typically, this is calculated as obs_horizon * obs_dim.
            diffusion_step_embed_dim (int, optional): Dimensionality of the diffusion step positional embedding. Default is 256.
            down_dims (List[int], optional): Channel sizes at each U-Net level. The length of this list determines the number of levels. Default is [256, 512, 1024].
            kernel_size (int, optional): Convolutional kernel size for all Conv1d operations. Default is 5.
            n_groups (int, optional): Number of groups to use for GroupNorm layers. Default is 8.
        """
        super().__init__()
        global_cond_dim: int = n_obs_steps * cond_dim
        # Combine input dimension and downsampling dimensions for the U-Net design.
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]
        dsed = diffusion_step_embed_dim

        # Diffusion step encoder: Applies sinusoidal positional encoding followed by two linear layers with Mish activation.
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )

        # Concatenate diffusion step embedding with global conditioning.
        cond_dim = dsed + global_cond_dim

        # Setup downsampling modules.
        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups),
            ]
        )

        down_modules = nn.ModuleList()
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(dim_in, dim_out, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups),
                        ConditionalResidualBlock1D(dim_out, dim_out, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )
        self.down_modules = down_modules

        # Setup upsampling modules.
        up_modules = nn.ModuleList()
        # Reverse the dimensions for the upsampling path, skipping the first (input) dimension.
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(
                nn.ModuleList(
                    [
                        # Concatenated features double the number of channels.
                        ConditionalResidualBlock1D(dim_out * 2, dim_in, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups),
                        ConditionalResidualBlock1D(dim_in, dim_in, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )
        self.up_modules = up_modules

        # Final convolution to map back to the input dimension.
        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, kernel_size=1),
        )
        self.final_conv = final_conv
        self.diffusion_step_encoder = diffusion_step_encoder

        # Print network properties.
        print("\n===================================")
        print(f"Initialized {type(self).__name__}")
        print(f"  Number of parameters: {self.num_parameters:,}")
        print(f"  Model size: {self.size / (1024**2):,.2f} MB")
        print(f"  Running on {self.device}")
        print(f"  Dtype: {self.dtype}")
        print("===================================\n")

    def forward(self, sample: torch.Tensor, timestep: Union[torch.Tensor, float, int], cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the U-Net.

        Args:
            sample (torch.Tensor): Input tensor of shape (B, T, input_dim), where B is the batch size, T is the temporal length.
            timestep (Union[torch.Tensor, float, int]): Diffusion step. Can be a tensor of shape (B,) or a scalar.
            condition (Optional[torch.Tenosr]): Conditioning vector of shape (B, T, obs_dim). Default is None.

        Returns:
            torch.Tensor: Output tensor of shape (B, T, input_dim).
        """
        if cond is not None:
            global_cond = cond.flatten(start_dim=1)
        # Rearrange sample from (B, T, input_dim) to (B, input_dim, T) for Conv1d processing.
        sample = sample.moveaxis(-1, -2)

        # Process the diffusion timestep:
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # Ensure timesteps is a tensor on the same device as the input sample.
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # Broadcast timesteps to the batch dimension.
        timesteps = timesteps.expand(sample.shape[0])

        # Encode the diffusion step.
        global_feature = self.diffusion_step_encoder(timesteps)
        # If a global conditioning vector is provided, concatenate it.
        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], dim=-1)

        x = sample
        h = []
        # Downsampling path.
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        # Middle modules.
        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        # Upsampling path.
        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            # Concatenate the skip connection from the downsampling path.
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        # Final convolution.
        x = self.final_conv(x)
        # Rearrange back to (B, T, input_dim) from (B, input_dim, T).
        x = x.moveaxis(-1, -2)
        return x
