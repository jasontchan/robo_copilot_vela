import math

import torch
import torch.nn as nn

from diffusion_policy.models.model_mixin import ModelMixin


class SinusoidalPosEmb(ModelMixin):
    """
    Computes a sinusoidal positional embedding for a given input tensor.

    The embedding is computed by multiplying the input tensor with exponentially
    scaled frequencies and then concatenating the sine and cosine transforms.

    Attributes:
        dim (int): The embedding dimension. Must be an even number.
    """

    def __init__(self, dim: int) -> None:
        """
        Initialize the SinusoidalPosEmb module.

        Args:
            dim (int): The dimensionality of the positional embedding.
        """
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute the sinusoidal positional embedding.

        Args:
            x (torch.Tensor): The input tensor of shape [batch_size, ...].
                              The embedding is computed for the last dimension.

        Returns:
            torch.Tensor: The computed positional embedding with shape
                          [batch_size, dim].
        """
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(ModelMixin):
    """
    1D Downsampling layer using a convolution with stride=2.
    """

    def __init__(self, dim: int) -> None:
        """
        Initialize the Downsample1d module.

        Args:
            dim (int): Number of input (and output) channels.
        """
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for downsampling.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, dim, length].

        Returns:
            torch.Tensor: Downsampled tensor of shape [batch_size, dim, new_length].
        """
        return self.conv(x)


class Upsample1d(ModelMixin):
    """
    1D Upsampling layer using a transposed convolution to increase sequence length.
    """

    def __init__(self, dim: int) -> None:
        """
        Initialize the Upsample1d module.

        Args:
            dim (int): Number of input (and output) channels.
        """
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for upsampling.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, dim, length].

        Returns:
            torch.Tensor: Upsampled tensor of shape [batch_size, dim, new_length].
        """
        return self.conv(x)


class Conv1dBlock(ModelMixin):
    """
    A convolutional block that applies a 1D convolution, group normalization,
    and the Mish activation function.
    """

    def __init__(self, inp_channels: int, out_channels: int, kernel_size: int, n_groups: int = 8) -> None:
        """
        Initialize the Conv1dBlock module.

        Args:
            inp_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the convolutional kernel.
            n_groups (int, optional): Number of groups for GroupNorm. Default is 8.
        """
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the convolutional block.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, inp_channels, length].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, out_channels, length].
        """
        return self.block(x)


class ConditionalResidualBlock1D(ModelMixin):
    """
    A conditional residual block for 1D signals that applies two convolutional blocks
    with FiLM-based conditioning and a residual connection.

    The block uses two sequential Conv1dBlock modules, where the first block's output
    is modulated by FiLM (Feature-wise Linear Modulation) using the provided conditioning vector.
    """

    def __init__(self, in_channels: int, out_channels: int, cond_dim: int, kernel_size: int = 3, n_groups: int = 8) -> None:
        """
        Initialize the ConditionalResidualBlock1D module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            cond_dim (int): Dimensionality of the conditioning vector.
            kernel_size (int, optional): Convolution kernel size. Default is 3.
            n_groups (int, optional): Number of groups for GroupNorm in the conv blocks. Default is 8.
        """
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
                Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
            ]
        )

        # FiLM modulation (https://arxiv.org/abs/1709.07871): predicts per-channel scale and bias.
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, cond_channels), nn.Unflatten(dim=-1, unflattened_size=(-1, 1)))

        # Adjust dimensions to match for residual addition if needed.
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the conditional residual block.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, in_channels, length].
            cond (torch.Tensor): Conditioning vector of shape [batch_size, cond_dim].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, out_channels, length].
        """
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        # Reshape embed to have separate scale and bias for each channel.
        embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
        scale = embed[:, 0, ...]
        bias = embed[:, 1, ...]
        out = scale * out + bias

        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out
