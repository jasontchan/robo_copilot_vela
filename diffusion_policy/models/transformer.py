from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from diffusion_policy.models.conv_blocks import SinusoidalPosEmb
from diffusion_policy.models.model_mixin import ModelMixin


class TransformerForDiffusion(ModelMixin):
    """
    Transformer-based model for diffusion with conditional embeddings.

    This module implements a Transformer architecture tailored for diffusion modeling.
    It supports conditioning on diffusion timesteps and, optionally, on additional observation inputs.

    Attributes:
        input_emb (nn.Linear): Linear layer to embed the input features.
        pos_emb (nn.Parameter): Learnable positional embeddings for the main trunk tokens.
        drop (nn.Dropout): Dropout layer applied to the input and condition embeddings.
        time_emb (SinusoidalPosEmb): Sinusoidal positional encoder for the diffusion timestep.
        cond_obs_emb (Optional[nn.Linear]): Linear layer for observation-based conditioning, if used.
        cond_pos_emb (Optional[torch.nn.Parameter]): Learnable positional embeddings for condition tokens.
        encoder (Optional[Union[nn.TransformerEncoder, nn.Sequential]]): Transformer encoder (or MLP) for conditioning tokens.
        decoder (Optional[nn.TransformerDecoder]): Transformer decoder to produce the output sequence.
        mask (Optional[torch.Tensor]): Causal mask for self-attention if causal attention is enabled.
        memory_mask (Optional[torch.Tensor]): Memory mask for cross-attention when both time and observation conditioning are used.
        ln_f (nn.LayerNorm): Final layer normalization.
        head (nn.Linear): Final linear head to map to the output dimension.
        T (int): Number of tokens for the main trunk (input sequence length, possibly adjusted).
        T_cond (int): Number of tokens for the condition encoder.
        horizon (int): Original sequence length.
        time_as_cond (bool): Flag indicating if time is used as part of the conditioning input.
        obs_as_cond (bool): Flag indicating if observations are used for conditioning.
        encoder_only (bool): Flag indicating if the model follows an encoder-only (BERT-like) architecture.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        horizon: int,
        n_obs_steps: Optional[int] = None,
        cond_dim: int = 0,
        n_layer: int = 12,
        n_head: int = 12,
        n_emb: int = 768,
        p_drop_emb: float = 0.1,
        p_drop_attn: float = 0.1,
        causal_attn: bool = False,
        time_as_cond: bool = True,
        obs_as_cond: bool = False,
        n_cond_layers: int = 0,
    ) -> None:
        """
        Initialize the TransformerForDiffusion module.

        Args:
            input_dim (int): Dimensionality of the input features.
            output_dim (int): Dimensionality of the output features.
            horizon (int): Total sequence length.
            n_obs_steps (Optional[int]): Number of observation steps. Defaults to `horizon` if None.
            cond_dim (int): Dimensionality of the observation conditioning vector. Default is 0.
            n_layer (int): Number of layers for the main Transformer decoder (or encoder-only model).
            n_head (int): Number of attention heads.
            n_emb (int): Dimensionality of the embeddings.
            p_drop_emb (float): Dropout probability for embedding layers.
            p_drop_attn (float): Dropout probability for attention layers.
            causal_attn (bool): Whether to use causal attention (mask future tokens). Default is False.
            time_as_cond (bool): Whether to use the time token as part of the conditioning input. Default is True.
            obs_as_cond (bool): Whether to use observations as additional conditioning. Default is False.
            n_cond_layers (int): Number of layers for the condition encoder. If 0, a simple MLP is used.
        """
        super().__init__()

        # Compute number of tokens for main trunk and condition encoder
        if n_obs_steps is None:
            n_obs_steps = horizon

        T: int = horizon
        T_cond: int = 1
        if not time_as_cond:
            T += 1
            T_cond -= 1
        obs_as_cond = cond_dim > 0
        if obs_as_cond:
            assert time_as_cond, "Observation conditioning requires time_as_cond to be True."
            T_cond += n_obs_steps

        # Input embedding stem.
        self.input_emb = nn.Linear(input_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, T, n_emb))
        self.drop = nn.Dropout(p_drop_emb)

        # Condition encoder: time embedding and optional observation embedding.
        self.time_emb = SinusoidalPosEmb(n_emb)
        self.cond_obs_emb: Optional[nn.Linear] = None

        if obs_as_cond:
            self.cond_obs_emb = nn.Linear(cond_dim, n_emb)

        self.cond_pos_emb: Optional[torch.nn.Parameter] = None
        self.encoder: Optional[Union[nn.TransformerEncoder, nn.Sequential]] = None
        self.decoder: Optional[nn.TransformerDecoder] = None
        encoder_only: bool = False
        if T_cond > 0:
            self.cond_pos_emb = nn.Parameter(torch.zeros(1, T_cond, n_emb))
            if n_cond_layers > 0:
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=n_emb, nhead=n_head, dim_feedforward=4 * n_emb, dropout=p_drop_attn, activation="gelu", batch_first=True, norm_first=True
                )
                self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=n_cond_layers)
            else:
                self.encoder = nn.Sequential(nn.Linear(n_emb, 4 * n_emb), nn.Mish(), nn.Linear(4 * n_emb, n_emb))
            # Decoder.
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=4 * n_emb,
                dropout=p_drop_attn,
                activation="gelu",
                batch_first=True,
                norm_first=True,  # important for stability
            )
            self.decoder = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=n_layer)
        else:
            # Encoder-only (BERT-like) architecture.
            encoder_only = True
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=n_emb, nhead=n_head, dim_feedforward=4 * n_emb, dropout=p_drop_attn, activation="gelu", batch_first=True, norm_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=n_layer)

        # Setup attention masks if causal attention is enabled.
        if causal_attn:
            # Causal mask: disallow attention to future tokens.
            sz = T
            mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
            self.register_buffer("mask", mask)

            if time_as_cond and obs_as_cond:
                S: int = T_cond
                t, s = torch.meshgrid(torch.arange(T), torch.arange(S), indexing="ij")
                mask = t >= (s - 1)  # Adjust mask since time is the first token in condition.
                mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
                self.register_buffer("memory_mask", mask)
            else:
                self.memory_mask = None
        else:
            self.mask = None
            self.memory_mask = None

        # Decoder head.
        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, output_dim)

        # Constant and configuration parameters.
        self.T = T
        self.T_cond = T_cond
        self.horizon = horizon
        self.time_as_cond = time_as_cond
        self.obs_as_cond = obs_as_cond
        self.encoder_only = encoder_only

        # Initialize model weights.
        self.apply(self._init_weights)

        # Print network properties.
        print("\n===================================")
        print(f"Initialized {type(self).__name__}")
        print(f"  Number of parameters: {self.num_parameters:,}")
        print(f"  Model size: {self.size / (1024**2):,.2f} MB")
        print(f"  Running on {self.device}")
        print(f"  Dtype: {self.dtype}")
        print("===================================\n")

    def _init_weights(self, module: nn.Module) -> None:
        """
        Initialize weights of the model's submodules.

        The function applies custom initialization based on the module type. Linear and Embedding
        layers are initialized with a normal distribution, while biases are zeroed. Special cases are
        provided for multi-head attention and LayerNorm.

        Args:
            module (nn.Module): A module instance within the model.
        """
        ignore_types = (
            nn.Dropout,
            SinusoidalPosEmb,
            nn.TransformerEncoderLayer,
            nn.TransformerDecoderLayer,
            nn.TransformerEncoder,
            nn.TransformerDecoder,
            nn.ModuleList,
            nn.Mish,
            nn.Sequential,
        )
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            weight_names = ["in_proj_weight", "q_proj_weight", "k_proj_weight", "v_proj_weight"]
            for name in weight_names:
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)
            bias_names = ["in_proj_bias", "bias_k", "bias_v"]
            for name in bias_names:
                bias = getattr(module, name)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, TransformerForDiffusion):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
            if module.cond_pos_emb is not None:
                torch.nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            pass  # Modules without trainable parameters.
        else:
            raise RuntimeError("Unaccounted module {}".format(module))

    def get_optim_groups(self, weight_decay: float = 1e-3) -> List[Dict[str, Any]]:
        """
        Separate model parameters into groups that should and should not undergo weight decay.

        This method splits parameters into two sets: one for parameters that will experience weight decay
        (e.g., weights of linear layers) and one for parameters that will not (e.g., biases and LayerNorm weights).

        Args:
            weight_decay (float, optional): Weight decay factor for regularization. Default is 1e-3.

        Returns:
            List[Dict[str, Any]]: A list of two dictionaries, each specifying parameters and their weight decay.
        """
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn
                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.startswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        # Special case: do not decay positional embeddings.
        no_decay.add("pos_emb")
        # no_decay.add("_dummy_variable")
        if self.cond_pos_emb is not None:
            no_decay.add("cond_pos_emb")

        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "Parameters {} made it into both decay/no_decay sets!".format(inter_params)
        assert len(param_dict.keys() - union_params) == 0, "Parameters {} were not separated into either decay/no_decay set!".format(
            param_dict.keys() - union_params
        )

        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        return optim_groups

    def configure_optimizers(self, learning_rate: float = 1e-4, weight_decay: float = 1e-3, betas: Tuple[float, float] = (0.9, 0.95)) -> torch.optim.Optimizer:
        """
        Configure and return the optimizer for the model.

        Args:
            learning_rate (float, optional): Learning rate for the optimizer. Default is 1e-4.
            weight_decay (float, optional): Weight decay factor for regularization. Default is 1e-3.
            betas (Tuple[float, float], optional): Beta coefficients for the AdamW optimizer. Default is (0.9, 0.95).

        Returns:
            torch.optim.Optimizer: The AdamW optimizer configured with the parameter groups.
        """
        optim_groups = self.get_optim_groups(weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer

    def forward(self, sample: torch.Tensor, timestep: Union[torch.Tensor, float, int], cond: Optional[torch.Tensor] = None, **kwargs: Any) -> torch.Tensor:
        """
        Forward pass through the TransformerForDiffusion model.

        Args:
            sample (torch.Tensor): Input tensor of shape (B, T, input_dim), where B is the batch size and T is the sequence length.
            timestep (Union[torch.Tensor, float, int]): Diffusion timestep; can be a tensor of shape (B,) or a scalar.
            cond (Optional[torch.Tensor], optional): Conditioning tensor of shape (B, T', cond_dim) if provided.

        Returns:
            torch.Tensor: Output tensor of shape (B, T, output_dim).
        """
        # Process the diffusion timestep.
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])
        time_emb = self.time_emb(timesteps).unsqueeze(1)  # (B, 1, n_emb)

        # Process the main input.
        input_emb = self.input_emb(sample)

        if self.encoder_only:
            # Encoder-only (BERT-like) approach.
            token_embeddings = torch.cat([time_emb, input_emb], dim=1)
            t = token_embeddings.shape[1]
            position_embeddings = self.pos_emb[:, :t, :]
            x = self.drop(token_embeddings + position_embeddings)
            x = self.encoder(src=x, mask=self.mask)
            x = x[:, 1:, :]  # Remove the time token.
        else:
            # Use the condition encoder and decoder.
            cond_embeddings = time_emb
            if self.obs_as_cond:
                cond_obs_emb = self.cond_obs_emb(cond)  # (B, T_obs, n_emb)
                cond_embeddings = torch.cat([cond_embeddings, cond_obs_emb], dim=1)
            tc = cond_embeddings.shape[1]
            position_embeddings = self.cond_pos_emb[:, :tc, :]
            x = self.drop(cond_embeddings + position_embeddings)
            x = self.encoder(x)
            memory = x  # (B, T_cond, n_emb)

            token_embeddings = input_emb
            t = token_embeddings.shape[1]
            position_embeddings = self.pos_emb[:, :t, :]
            x = self.drop(token_embeddings + position_embeddings)
            x = self.decoder(tgt=x, memory=memory, tgt_mask=self.mask, memory_mask=self.memory_mask)
        x = self.ln_f(x)
        x = self.head(x)
        return x


def test() -> None:
    """
    Run tests for TransformerForDiffusion with various configurations.

    This function creates instances of the model with different parameter settings,
    generates dummy inputs, and performs forward passes to verify that the implementations work.
    """
    # GPT with time embedding only.
    transformer = TransformerForDiffusion(
        input_dim=16,
        output_dim=16,
        horizon=8,
        n_obs_steps=4,
        causal_attn=True,
    )
    opt = transformer.configure_optimizers()

    timestep = torch.tensor(0)
    sample = torch.zeros((4, 8, 16))
    out = transformer(sample, timestep)

    # GPT with time embedding and observation conditioning.
    transformer = TransformerForDiffusion(
        input_dim=16,
        output_dim=16,
        horizon=8,
        n_obs_steps=4,
        cond_dim=10,
        causal_attn=True,
    )
    opt = transformer.configure_optimizers()

    timestep = torch.tensor(0)
    sample = torch.zeros((4, 8, 16))
    cond = torch.zeros((4, 4, 10))
    out = transformer(sample, timestep, cond)

    # GPT with time embedding, observation conditioning, and a condition encoder.
    transformer = TransformerForDiffusion(input_dim=16, output_dim=16, horizon=8, n_obs_steps=4, cond_dim=10, causal_attn=True, n_cond_layers=4)
    opt = transformer.configure_optimizers()

    timestep = torch.tensor(0)
    sample = torch.zeros((4, 8, 16))
    cond = torch.zeros((4, 4, 10))
    out = transformer(sample, timestep, cond)

    # BERT with time embedding token (encoder-only).
    transformer = TransformerForDiffusion(
        input_dim=16,
        output_dim=16,
        horizon=8,
        n_obs_steps=4,
        time_as_cond=False,
    )
    opt = transformer.configure_optimizers()

    timestep = torch.tensor(0)
    sample = torch.zeros((4, 8, 16))
    out = transformer(sample, timestep)
