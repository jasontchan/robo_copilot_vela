#!/usr/bin/env python
"""
Launch distributed diffusion-policy training optimized for G6e.
Example bash:
 torchrun --nnodes=1 --nproc_per_node=4 \
   train_ddp_g6e_optimized.py --config config_files/blockpush_config.py
"""

import argparse
import os
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from diffusers import DDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm

import wandb
from diffusion_policy.data import ZarrTrialDataset
from diffusion_policy.utils import DiffusionPolicyConfig, get_condition, initialize_networks, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="G6e-optimized distributed diffusion-policy training.")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file.")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint file to resume from.")
    parser.add_argument("--wandb_run_id", type=str, default=None, help="WandB run id to resume logging.")
    return parser.parse_args()


def process_batch(
    batch: Dict[str, torch.Tensor], config: DiffusionPolicyConfig, nets: nn.ModuleDict, device: torch.device, scheduler: DDPMScheduler
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    images = batch["image"][:, : config.obs_horizon].to(device, non_blocking=True)
    proprio = batch["agent_pos"][:, : config.obs_horizon].to(device, non_blocking=True)
    actions = batch["action"].to(device, non_blocking=True)

    cond = get_condition(proprio, images, nets)
    noise = torch.randn_like(actions, device=device)
    timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (actions.shape[0],), device=device).long()
    noisy_actions = scheduler.add_noise(actions, noise, timesteps)
    return noisy_actions, noise, timesteps, cond


def train_one_epoch(
    dataloader: DataLoader,
    nets_ddp: DistributedDataParallel,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    noise_scheduler: DDPMScheduler,
    config: DiffusionPolicyConfig,
    device: torch.device,
    ema: EMAModel,
    scaler: GradScaler,
    rank: int,
) -> float:
    # Set model to train and extract the ModuleDict
    nets_ddp.train()
    nets_dict: nn.ModuleDict = nets_ddp.module
    losses = []
    loader = tqdm(dataloader, desc="Train Batches", disable=(rank != 0))

    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            noisy_actions, noise, timesteps, cond = process_batch(batch, config, nets_dict, device, noise_scheduler)
            # Only the diffusion network makes predictions
            pred = nets_dict["noise_pred_net"](noisy_actions, timesteps, cond)
            loss = nn.functional.mse_loss(pred, noise)
        # Mixed-precision backward and step
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()
        ema.step(nets_dict.parameters())

        loss_val = loss.item()
        losses.append(loss_val)
        if rank == 0:
            wandb.log({"batch_loss": loss_val, "lr": lr_scheduler.get_last_lr()[0]})
            loader.set_postfix(loss=loss_val)

    return float(np.mean(losses))


def main():
    args = parse_args()
    # NCCL & cuDNN auto-tune
    os.environ.setdefault("NCCL_SOCKET_NTHREADS", "4")
    torch.backends.cudnn.benchmark = True

    dist.init_process_group(backend="nccl")
    local_rank = dist.get_rank() % torch.cuda.device_count()
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    is_main = dist.get_rank() == 0
    if is_main:
        wandb.init(
            project="diffusion_policy_multicam_g6e", config=load_config(args.config).as_dict(), id=args.wandb_run_id, resume=args.wandb_run_id is not None
        )
    config = DiffusionPolicyConfig(**wandb.config) if is_main else load_config(args.config)

    dataset = ZarrTrialDataset(root_dir=config.data_dir, pred_horizon=config.pred_horizon, obs_horizon=config.obs_horizon, action_horizon=config.action_horizon)
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, sampler=sampler, num_workers=8, pin_memory=True, prefetch_factor=4, persistent_workers=True)

    nets = initialize_networks(config).to(device)
    nets_ddp = DistributedDataParallel(nets, device_ids=[local_rank])

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        state = OrderedDict({("module." + k if not k.startswith("module.") else k): v for k, v in ckpt.items()})
        nets_ddp.load_state_dict(state)

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config.num_diffusion_iters, beta_schedule="squaredcos_cap_v2", clip_sample=True, prediction_type="epsilon"
    )
    ema = EMAModel(nets_ddp.parameters(), power=0.75)

    optimizer = (
        torch.optim.AdamW(nets_ddp.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        if config.model == "unet"
        else nets_ddp.module["noise_pred_net"].configure_optimizers()
    )

    total_steps = len(dataloader) * config.num_epochs
    lr_sched = get_scheduler(name="cosine", optimizer=optimizer, num_warmup_steps=config.warmup_steps, num_training_steps=total_steps)

    scaler = GradScaler()

    best_loss = float("inf")
    for epoch in range(config.num_epochs):
        sampler.set_epoch(epoch)
        if is_main:
            print(f"Epoch {epoch + 1}/{config.num_epochs}")
        epoch_loss = train_one_epoch(dataloader, nets_ddp, optimizer, lr_sched, noise_scheduler, config, device, ema, scaler, dist.get_rank())
        if is_main:
            print(f"Epoch Loss: {epoch_loss:.4f}")
            wandb.log({"epoch_loss": epoch_loss, "epoch": epoch + 1})
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                ckpt_folder = Path(config.save_dir) / wandb.run.name
                ckpt_folder.mkdir(exist_ok=True, parents=True)
                ckpt_file = ckpt_folder / f"best_epoch{epoch + 1}.pt"
                torch.save(nets_ddp.module.state_dict(), ckpt_file)
                wandb.log({"best_ckpt": str(ckpt_file)})

    if is_main:
        final_path = Path(config.save_dir) / wandb.run.name / "final.pt"
        torch.save(nets_ddp.module.state_dict(), final_path)
        print("Training complete.")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
