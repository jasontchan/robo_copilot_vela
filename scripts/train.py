"""
```bash
torchrun --nnodes=1 --node_rank=0 --nproc_per_node=4 \
train_ddp.py --config config_files/blockpush_config.py
```
"""

from collections import OrderedDict
import argparse
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from diffusers import DDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm

import wandb
from diffusion_policy.data import ZarrTrialDataset
from diffusion_policy.utils import DiffusionPolicyConfig, get_condition, initialize_networks, load_config


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for distributed training.

    Returns:
        argparse.Namespace: Parsed command-line arguments with attributes:
            - config: Path to the config file.
            - resume: Optional path to a checkpoint to resume from.
            - wandb_run_id: Optional WandB run id to resume logging.
            - local_rank: Local GPU index (set automatically by torchrun/launch).
    """
    parser = argparse.ArgumentParser(description="Train a diffusion policy with multi-camera inputs using DistributedDataParallel.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the config file (e.g., 'config_files/blockpush_config.py').",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a local checkpoint file to resume training from.",
    )
    parser.add_argument(
        "--wandb_run_id",
        type=str,
        default=None,
        help="WandB run id to resume logging (if not provided, a new run is created).",
    )
    # This argument is required for DDP; it is set automatically by torchrun/launch.
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training")
    return parser.parse_args()


def process_batch(
    batch: Dict[str, torch.Tensor], config: DiffusionPolicyConfig, nets: nn.ModuleDict, device: torch.device, scheduler: DDPMScheduler
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Process one batch of data: perform device transfers, encode vision features,
    prepare global conditioning, and sample noise.

    Args:
        batch (Dict[str, torch.Tensor]): A batch from the DataLoader containing keys 'image', 'agent_pos', and 'action'.
        config (DiffusionPolicyConfig): Configuration object with necessary hyperparameters.
        nets (nn.ModuleDict): Network modules (wrapped with DDP) containing vision encoders and noise prediction network.
        device (torch.device): The device for computation.
        noise_scheduler (DDPMScheduler): Scheduler used for adding noise.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            - noisy_actions: Actions with added noise.
            - noise: The noise tensor.
            - timesteps: Random timesteps for noise scheduling.
            - cond: Conditioning tensor.
    """
    image_batch: torch.Tensor = batch["image"][:, : config.obs_horizon].to(device)  # (B, obs_horizon, view, C, H, W)
    proprio_batch: torch.Tensor = batch["agent_pos"][:, : config.obs_horizon].to(device)  # (B, obs_horizon, obs_dim)
    action_batch: torch.Tensor = batch["action"].to(device)  # (B, action_dim)
    batch_size: int = image_batch.shape[0]

    cond: torch.Tensor = get_condition(proprio_batch, image_batch, nets)
    noise: torch.Tensor = torch.randn_like(action_batch, device=device)
    timesteps: torch.Tensor = torch.randint(0, scheduler.config.num_train_timesteps, (batch_size,), device=device).long()

    noisy_actions: torch.Tensor = scheduler.add_noise(action_batch, noise, timesteps)

    return noisy_actions, noise, timesteps, cond


def train_one_epoch(
    dataloader: DataLoader,
    nets_ddp: nn.parallel.DistributedDataParallel,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    noise_scheduler: DDPMScheduler,
    config: DiffusionPolicyConfig,
    device: torch.device,
    ema: EMAModel,
    rank: int,
) -> float:
    """
    Train the model for one epoch.

    Args:
        dataloader (DataLoader): DataLoader providing training batches.
        nets_ddp (torch.DistributedDataParallel): The model wrapped in DDP.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        noise_scheduler (DDPMScheduler): Scheduler used for noise addition.
        config (DiffusionPolicyConfig): Configuration object.
        device (torch.device): Device used for training.
        ema (EMAModel): Exponential Moving Average model for parameters.
        rank (int): Rank of the current process (0 is main).

    Returns:
        float: Average training loss for the epoch.
    """
    nets: nn.ModuleDict = nets_ddp.module
    nets.train()
    epoch_losses = []
    batch_bar = tqdm(dataloader, desc="Training Batches", leave=False) if rank == 0 else dataloader
    # batch_bar = dataloader

    batch_it = iter(batch_bar)
    local_rank = int(os.environ["LOCAL_RANK"])
    i = 0
    while True:
        try:
            i += 1
            batch = next(batch_it)  # use iterator method to catch problems

            optimizer.zero_grad(set_to_none=True)
            noisy_actions, noise, timesteps, cond = process_batch(batch, config, nets, device, noise_scheduler)
            noise_pred = nets["noise_pred_net"](noisy_actions, timesteps, cond)
            loss = nn.functional.mse_loss(noise_pred, noise)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()  # Step per batch.
            ema.step(nets.parameters())
            loss_value = loss.item()
            epoch_losses.append(loss_value)
            if rank == 0:
                wandb.log({"batch_loss": loss_value, "learning_rate": lr_scheduler.get_last_lr()[0]})
                batch_bar.set_postfix({"loss": loss_value})
        except StopIteration:
            # print(f"{local_rank = }, {i = }")
            break
        except Exception:
            print(f"data has been thrown away on {local_rank = } at {i = }")
            # print(e)
            raise
            pass

    avg_loss: float = np.mean(epoch_losses) if epoch_losses else float("inf")
    return avg_loss


def main(config: DiffusionPolicyConfig, resume_checkpoint: Optional[str] = None, wandb_run_id: Optional[str] = None, local_rank: int = 0) -> None:
    """
    Main training function that sets up distributed training, loads data, and executes training epochs.

    Args:
        config (DiffusionPolicyConfig): Configuration object containing hyperparameters and file paths.
        resume_checkpoint (Optional[str]): Path to a checkpoint file for resuming training.
        wandb_run_id (Optional[str]): WandB run id for resuming logging.
        local_rank (int): Local GPU index for distributed training.
    """
    # Initialize distributed training
    dist.init_process_group(backend="nccl")
    device: torch.device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    # Only rank 0 logs to WandB.
    is_main: bool = dist.get_rank() == 0
    if is_main:
        if wandb_run_id is not None:
            wandb.init(project="diffusion_policy_multicam", resume="must", id=wandb_run_id, config=config.as_dict())
        else:
            wandb.init(project="diffusion_policy_multicam", config=config.as_dict())
        config = wandb.config
    else:
        # For non-main processes, we still want access to the config.
        config = load_config(args.config)

    # Set up the dataset and distributed sampler.
    dataset = ZarrTrialDataset(root_dir=config.data_dir, pred_horizon=config.pred_horizon, obs_horizon=config.obs_horizon, action_horizon=config.action_horizon)
    sampler: DistributedSampler = DistributedSampler(dataset, shuffle=True)
    dataloader: DataLoader = DataLoader(dataset, batch_size=config.batch_size, sampler=sampler)

    nets: nn.Module = initialize_networks(config)
    nets.to(device)
    nets_ddp = DistributedDataParallel(nets, device_ids=[local_rank], output_device=local_rank)

    # Resume checkpoint if provided (only rank 0 prints logging info).
    if resume_checkpoint is not None:
        ckpt_path: Path = Path(resume_checkpoint)
        if ckpt_path.exists():
            state_dict = torch.load(str(ckpt_path), map_location=device)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = f"module.{k}" if not k.startswith("module.") else k
                new_state_dict[name] = v
            nets_ddp.load_state_dict(new_state_dict)
            if is_main:
                print(f"Resumed from local checkpoint: {ckpt_path}")
                wandb.log({"resume_checkpoint": str(ckpt_path)})
        else:
            if is_main:
                print(f"Local checkpoint file not found: {ckpt_path}")

    scheduler: DDPMScheduler = DDPMScheduler(
        num_train_timesteps=config.num_diffusion_iters, beta_schedule="squaredcos_cap_v2", clip_sample=True, prediction_type="epsilon"
    )
    ema: EMAModel = EMAModel(parameters=nets_ddp.parameters(), power=0.75)
    if config.model == "unet":
        optimizer = torch.optim.AdamW(nets_ddp.parameters(), lr=1e-4, weight_decay=1e-6)
    elif config.model == "transformer":
        optimizer = nets_ddp.module["noise_pred_net"].configure_optimizers()
    else:
        raise ValueError
    total_steps: int = len(dataloader) * config.num_epochs
    lr_scheduler = get_scheduler(name="cosine", optimizer=optimizer, num_warmup_steps=500, num_training_steps=total_steps)

    if is_main:
        run_ckpt_dir: Path = Path(config.save_dir) / wandb.run.name
        run_ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_epoch_loss: float = float("inf")
    for epoch in range(config.num_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        if is_main:
            print(f"Epoch {epoch + 1}/{config.num_epochs}")
        epoch_loss: float = train_one_epoch(dataloader, nets_ddp, optimizer, lr_scheduler, scheduler, config, device, ema, dist.get_rank())
        if is_main:
            print(f"Average Epoch Loss: {epoch_loss:.4f}")
            wandb.log({"epoch": epoch + 1, "epoch_loss": epoch_loss})
            if epoch_loss < best_epoch_loss:
                best_epoch_loss = epoch_loss
                ckpt_filename: str = f"{wandb.run.name}_epoch{epoch + 1}_best.ckpt"
                ckpt_path: Path = run_ckpt_dir / ckpt_filename
                torch.save(nets.state_dict(), str(ckpt_path))
                wandb.log({"best_epoch_loss": best_epoch_loss})
                print(f"Saved best checkpoint: {ckpt_path}")

    ema.copy_to(nets_ddp.parameters())

    if is_main:
        final_ckpt_path: Path = run_ckpt_dir / f"{wandb.run.name}_final.ckpt"
        torch.save(nets.state_dict(), str(final_ckpt_path))
        print(f"Saved final checkpoint: {final_ckpt_path}")
        print("Training complete!")

    dist.destroy_process_group()


if __name__ == "__main__":
    args: argparse.Namespace = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    print(f"Local rank: {local_rank}")
    # print(f"Local rank: {args.local_rank}")

    config: DiffusionPolicyConfig = load_config(args.config)
    main(config, resume_checkpoint=args.resume, wandb_run_id=args.wandb_run_id, local_rank=local_rank)  # args.local_rank)
