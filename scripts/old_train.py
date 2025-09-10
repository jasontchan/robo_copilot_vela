"""
train.py

A training script for a diffusion policy with multi-camera inputs.
This script integrates Weights & Biases (WandB) for experiment tracking.
You can specify a config file via a file path, and optionally resume training
from a local checkpoint or from a WandB artifact. (not anymore, can't afford WandB Pro)
"""


# os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2, 3"

import argparse
import importlib.util

# import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import wandb

# Local imports (ensure these modules are in your PYTHONPATH)
from diffusion_policy.data import ZarrTrialDataset
from diffusion_policy.models import ConditionalUnet1D, get_resnet, replace_bn_with_gn
from diffusion_policy.models.transformer import TransformerForDiffusion


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Train a diffusion policy with multi-camera inputs.")
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
    return parser.parse_args()


def load_config(config_path):
    """
    Load a configuration file given its file path.

    Args:
        config_path (str): File path to the config file.

    Returns:
        The configuration object named `config` defined in the file.
    """
    config_path = Path(config_path)
    spec = importlib.util.spec_from_file_location(config_path.stem, str(config_path))
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    # Assumes the config file defines a variable named `config`.
    return config_module.config


def initialize_networks(config):
    """
    Initialize the noise prediction network and vision encoders for each camera view.

    Args:
        config: Configuration object with necessary parameters.

    Returns:
        nn.ModuleDict: Dictionary containing the noise prediction network and vision encoders.
    """
    nets = nn.ModuleDict()

    # Initialize the noise prediction network.
    if config.model is ConditionalUnet1D:
        nets["noise_pred_net"] = ConditionalUnet1D(input_dim=config.action_dim, n_obs_steps=config.obs_horizon, cond_dim=config.obs_dim)
    elif config.model is TransformerForDiffusion:
        nets["noise_pred_net"] = TransformerForDiffusion(
            input_dim=config.action_dim,
            output_dim=config.action_dim,
            horizon=2 * config.obs_horizon + config.inference_delay,
            n_obs_steps=config.obs_horizon,
            cond_dim=config.obs_dim,
            obs_as_cond=True,
        )

    # For each camera view, initialize a separate ResNet18-based vision encoder.
    for view in config.views:
        encoder = get_resnet("resnet18")
        # Replace BatchNorm with GroupNorm (critical for EMA stability)
        encoder = replace_bn_with_gn(encoder)
        nets[f"vision_encoder_{view}"] = encoder

    return nets


def process_batch(batch, config, nets, device, noise_scheduler):
    """
    Process one batch of data: perform device transfers, encode vision features,
    prepare global conditioning, and sample noise.

    Args:
        batch (dict): Batch of data from the DataLoader.
        config: Configuration object.
        nets (nn.ModuleDict): Dictionary containing network modules.
        device (torch.device): The device on which data should reside.
        noise_scheduler (DDPMScheduler): Scheduler used for diffusion noise.

    Returns:
        tuple: (noisy_actions, noise, timesteps, conditioning_features)
    """
    try:
        images = batch["image"][:, : config.obs_horizon].to(device)  # (B, obs_horizon, view, C, H, W)
        agent_positions = batch["agent_pos"][:, : config.obs_horizon].to(device)  # (B, obs_horizon, obs_dim)
        actions = batch["action"].to(device)  # (B, pred_horizon, action_dim)
        batch_size = images.shape[0]

        vision_feature_list = []
        for idx, view in enumerate(config.views):
            view_images = images[:, :, idx]  # (B, obs_horizon, C, H, W)
            flat_view_images = view_images.flatten(end_dim=1)  # (B*obs_horizon, C, H, W)
            view_features = nets.module[f"vision_encoder_{view}"](flat_view_images)
            view_features = view_features.reshape(images.shape[0], images.shape[1], -1)  # (B, obs_horizon, D)
            vision_feature_list.append(view_features)

        vision_features = torch.cat(vision_feature_list, dim=-1)  # (B, obs_horizon, total_feature_dim)
        conditioning_features = torch.cat([vision_features, agent_positions], dim=-1)

        noise = torch.randn(actions.shape, device=device)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=device).long()
        noisy_actions = noise_scheduler.add_noise(actions, noise, timesteps)

        return noisy_actions, noise, timesteps, conditioning_features

    except Exception as e:
        print(f"Error processing batch: {e}")
        raise


def train_one_epoch(dataloader, nets, optimizer, lr_scheduler, noise_scheduler, config, device, ema):
    """
    Train for one epoch.

    Args:
        dataloader (DataLoader): DataLoader for training data.
        nets (nn.ModuleDict): Dictionary containing network modules.
        optimizer (torch.optim.Optimizer): Optimizer.
        lr_scheduler: Learning rate scheduler.
        noise_scheduler (DDPMScheduler): Scheduler for diffusion noise.
        config: Configuration object.
        device (torch.device): Device for training.
        ema (EMAModel): Exponential Moving Average (EMA) model.

    Returns:
        float: Average training loss for the epoch.
    """
    nets.train()
    epoch_losses = []
    batch_bar = tqdm(dataloader, desc="Training Batches", leave=False)
    for batch in batch_bar:
        try:
            optimizer.zero_grad(set_to_none=True)

            noisy_actions, noise, timesteps, cond = process_batch(batch, config, nets, device, noise_scheduler)
            noise_pred = nets.module["noise_pred_net"](noisy_actions, timesteps, cond)
            loss = nn.functional.mse_loss(noise_pred, noise)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()  # Step per batch.
            ema.step(nets.parameters())

            loss_value = loss.item()
            epoch_losses.append(loss_value)
            batch_bar.set_postfix({"loss": loss_value})
            wandb.log({"batch_loss": loss_value, "learning_rate": lr_scheduler.get_last_lr()[0]})
        except Exception as e:
            print(f"Error in training batch: {e}")
            continue
    avg_loss = np.mean(epoch_losses) if epoch_losses else float("inf")
    return avg_loss


def main(config, resume_checkpoint=None, wandb_run_id=None):
    """
    Main training function.

    Args:
        config: The configuration object with hyperparameters and file paths.
        resume_checkpoint (str, optional): Path to a local checkpoint file to resume training.
        wandb_run_id (str, optional): The WandB run id to resume logging.
    """
    if wandb_run_id is not None:
        wandb.init(project="diffusion_policy_multicam", resume="must", id=wandb_run_id, config=config)
    else:
        wandb.init(project="diffusion_policy_multicam", config=config)
    config = wandb.config

    dataset = ZarrTrialDataset(root_dir=config.data_dir, pred_horizon=config.pred_horizon, obs_horizon=config.obs_horizon, action_horizon=config.action_horizon)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    nets = initialize_networks(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nets = torch.nn.DataParallel(nets)
    nets.to(device)

    # RESUME LOGIC
    if resume_checkpoint is not None:
        ckpt_path = Path(resume_checkpoint)
        if ckpt_path.exists():
            state_dict = torch.load(str(ckpt_path), map_location=device)
            nets.load_state_dict(state_dict)
            print(f"Resumed from local checkpoint: {ckpt_path}")
            wandb.log({"resume_checkpoint": str(ckpt_path)})
        else:
            print(f"Local checkpoint file not found: {ckpt_path}")
    #    elif wandb_run_id is not None:
    #        try:
    #            artifact_path = f"{wandb.run.entity}/{wandb.run.project}/{wandb.run.name}_final:latest"
    #            artifact = wandb.use_artifact(artifact_path, type="model")
    #            with tempfile.TemporaryDirectory() as temp_dir:
    #                artifact_dir = artifact.download(root=temp_dir)
    #                artifact_ckpt = Path(artifact_dir) / f"{wandb.run.name}_final.ckpt"
    #                if artifact_ckpt.exists():
    #                    state_dict = torch.load(str(artifact_ckpt), map_location=device)
    #                    nets.load_state_dict(state_dict)
    #                    print(f"Resumed from artifact checkpoint: {artifact_ckpt}")
    #                    wandb.log({"resume_checkpoint": str(artifact_ckpt)})
    #                else:
    #                    print("Artifact checkpoint file not found.")
    #            # The temporary directory and its contents are automatically removed here.
    #        except Exception as e:
    #            print(f"Could not load artifact checkpoint: {e}")

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config.num_diffusion_iters, beta_schedule="squaredcos_cap_v2", clip_sample=True, prediction_type="epsilon"
    )
    ema = EMAModel(parameters=nets.parameters(), power=0.75)
    if config.model is ConditionalUnet1D:
        optimizer = torch.optim.AdamW(nets.parameters(), lr=1e-4, weight_decay=1e-6)
    elif config.model is TransformerForDiffusion:
        optimizer = nets.module["noise_pred_net"].configure_optimizers()
    else:
        raise ValueError
    total_steps = len(dataloader) * config.num_epochs
    lr_scheduler = get_scheduler(name="cosine", optimizer=optimizer, num_warmup_steps=500, num_training_steps=total_steps)

    run_ckpt_dir = Path(config.save_dir) / wandb.run.name
    run_ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_epoch_loss = float("inf")
    for epoch in range(config.num_epochs):
        print(f"Epoch {epoch + 1}/{config.num_epochs}")
        epoch_loss = train_one_epoch(dataloader, nets, optimizer, lr_scheduler, noise_scheduler, config, device, ema)
        print(f"Average Epoch Loss: {epoch_loss:.4f}")
        wandb.log({"epoch": epoch + 1, "epoch_loss": epoch_loss})

        if epoch_loss < best_epoch_loss:
            best_epoch_loss = epoch_loss
            ckpt_filename = f"{wandb.run.name}_epoch{epoch + 1}_best.ckpt"
            ckpt_path = run_ckpt_dir / ckpt_filename
            torch.save(nets.state_dict(), str(ckpt_path))
            wandb.log({"best_epoch_loss": best_epoch_loss})
            print(f"Saved best checkpoint: {ckpt_path}")

    ema.copy_to(nets.parameters())
    final_ckpt_path = run_ckpt_dir / f"{wandb.run.name}_final.ckpt"
    torch.save(nets.state_dict(), str(final_ckpt_path))
    print(f"Saved final checkpoint: {final_ckpt_path}")

    #    model_artifact = wandb.Artifact(name=f"{wandb.run.name}_final", type="model", description="Final model checkpoint")
    #    model_artifact.add_file(str(final_ckpt_path))
    #    wandb.log_artifact(model_artifact)
    #    print("Logged final model as artifact.")

    print("Training complete!")


if __name__ == "__main__":
    args = parse_args()
    # Load config using a file path (directory notation)
    config = load_config(args.config)
    main(config, resume_checkpoint=args.resume, wandb_run_id=args.wandb_run_id)
