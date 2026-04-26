"""
Pretrain Frangi-Masked SIGReg on XCA coronary angiography dataset.

Single-student SSL using Frangi vesselness-guided conditional loss:
- SIGReg on background patches (Gaussian constraint)
- VICReg on vessel patches (variance + invariance + covariance)
- Soft weighting via Frangi vesselness masks
- Adaptive normalization via EMA running means

Usage:
    python train.py --config configs/pretrain.yaml
    python train.py --config configs/pretrain.yaml --resume /path/to/checkpoint.pt
"""

import argparse
import logging
import math
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data.dataset import XcaDataset
from data.transforms import XcaViewGenerator
from loss.conditional_loss import FrangiMaskedConditionalLoss
from models.frangi_masked_jepa import FrangiMaskedJEPA
from utils.checkpointing import load_checkpoint, save_checkpoint

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Frangi-Masked SIGReg pretraining on XCA"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (cuda:0, cpu, etc.)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: single batch, short run",
    )
    return parser.parse_args()


def load_config(path: str) -> dict:
    """Load YAML config file."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def build_model(cfg: dict, device: torch.device) -> Tuple[
    nn.Module,
    FrangiMaskedConditionalLoss,
    torch.optim.Optimizer,
    Optional[torch.optim.lr_scheduler.LambdaLR],
]:
    """
    Build model, loss function, optimizer, and LR scheduler from config.

    Args:
        cfg: full experiment config dict
        device: target device

    Returns:
        model: FrangiMaskedJEPA on device
        loss_fn: FrangiMaskedConditionalLoss on device
        optimizer: AdamW optimizer
        scheduler: cosine decay with linear warmup (or None)
    """
    # --- Model ---
    backbone_cfg = {
        "img_size": cfg["data"]["image_size"],
        "patch_size": 8,
        "in_chans": 1,
        "embed_dim": cfg["model"]["embed_dim"],
        "depth": cfg["model"]["depth"],
        "num_heads": cfg["model"]["num_heads"],
        "out_layers": cfg["model"]["hierarchical_layers"],
    }
    proj_cfg = {
        "in_dim": cfg["model"]["embed_dim"],
        "hidden_dim": cfg["model"]["proj_hidden"],
        "out_dim": cfg["model"]["proj_dim"],
        "n_layers": 2,
    }
    frangi_cfg = {
        "patch_vessel_weight_scale": cfg["frangi"]["patch_vessel_weight_scale"],
    }

    model = FrangiMaskedJEPA(
        backbone_cfg=backbone_cfg,
        proj_cfg=proj_cfg,
        frangi_cfg=frangi_cfg,
    ).to(device)

    # --- Loss ---
    sigreg_cfg = {
        "proj_dim": cfg["loss"]["sigreg_proj_dim"],
        "knots": cfg["loss"]["sigreg_knots"],
        "beta": cfg["loss"]["sigreg_beta"],
    }
    vicreg_cfg = {
        "sim_coeff": cfg["loss"]["vicreg_sim_coeff"],
        "std_coeff": cfg["loss"]["vicreg_std_coeff"],
        "cov_coeff": cfg["loss"]["vicreg_cov_coeff"],
        "target_std": cfg["loss"]["target_std"],
    }
    loss_fn = FrangiMaskedConditionalLoss(
        sigreg_cfg=sigreg_cfg,
        vicreg_cfg=vicreg_cfg,
        lambda_bg=cfg["loss"]["lambda_bg"],
        lambda_vessel=cfg["loss"]["lambda_vessel"],
        loss_ema_decay=cfg["loss"]["loss_ema_decay"],
    ).to(device)

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
        betas=tuple(cfg["training"]["betas"]),
    )

    # --- LR Scheduler: cosine decay with linear warmup ---
    total_epochs = cfg["training"]["epochs"]
    warmup_epochs = cfg["training"]["warmup_epochs"]

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = float(epoch - warmup_epochs) / float(
            max(1, total_epochs - warmup_epochs)
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return model, loss_fn, optimizer, scheduler


def build_dataloaders(cfg: dict) -> Tuple[DataLoader, Optional[DataLoader], XcaViewGenerator]:
    """
    Build train/val dataloaders and view generator from config.

    Args:
        cfg: full experiment config dict

    Returns:
        train_loader: DataLoader for training
        val_loader: DataLoader for validation (or None)
        view_generator: XcaViewGenerator for SSL views
    """
    aug_cfg = cfg["data"]["augmentation"]
    loader_cfg = cfg["data"]["loader"]

    # View generator
    view_generator = XcaViewGenerator(
        image_size=cfg["data"]["image_size"],
        global_scale=tuple(aug_cfg["global_scale"]),
        n_global=2,
        intensity_jitter=aug_cfg.get("color_jitter", 0.4),
        blur_prob=0.1,
        blur_sigma=(0.1, 2.0),
    )

    # Training dataset
    train_dataset = XcaDataset(
        root=cfg["data"]["train_path"],
        transform=None,  # View generator handles transforms
        sigmas=cfg["frangi"]["sigmas"],
        frangi_beta=cfg["frangi"]["beta"],
        frangi_threshold=cfg["frangi"]["frangi_threshold"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=loader_cfg["batch_size"],
        shuffle=loader_cfg["shuffle"],
        num_workers=loader_cfg["num_workers"],
        pin_memory=loader_cfg["pin_memory"],
        drop_last=loader_cfg["drop_last"],
    )

    # Validation dataset (optional)
    val_path = cfg["data"].get("val_path")
    if val_path and Path(val_path).exists():
        val_dataset = XcaDataset(
            root=val_path,
            transform=None,
            sigmas=cfg["frangi"]["sigmas"],
            frangi_beta=cfg["frangi"]["beta"],
            frangi_threshold=cfg["frangi"]["frangi_threshold"],
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=loader_cfg["batch_size"],
            shuffle=False,
            num_workers=loader_cfg["num_workers"],
            pin_memory=loader_cfg["pin_memory"],
            drop_last=False,
        )
    else:
        val_loader = None

    return train_loader, val_loader, view_generator


def train_epoch(
    model: nn.Module,
    loss_fn: FrangiMaskedConditionalLoss,
    loader: DataLoader,
    view_generator: XcaViewGenerator,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    device: torch.device,
    epoch: int,
    cfg: dict,
    writer: Optional[SummaryWriter],
) -> Dict[str, float]:
    """
    Train for one epoch.

    Args:
        model: FrangiMaskedJEPA
        loss_fn: FrangiMaskedConditionalLoss
        loader: training DataLoader
        view_generator: XcaViewGenerator for multi-view augmentation
        optimizer: AdamW optimizer
        scaler: GradScaler for mixed precision (or None)
        device: target device
        epoch: current epoch number (for logging)
        cfg: config dict
        writer: TensorBoard SummaryWriter (or None)

    Returns:
        dict of average metrics for the epoch
    """
    model.train()
    loss_fn.train()

    total_loss = 0.0
    total_sigreg = 0.0
    total_vicreg = 0.0
    total_vessel_patches = 0
    total_patches = 0
    num_batches = 0
    log_freq = cfg["logging"]["log_freq"]
    mixed_precision = cfg["training"]["mixed_precision"] and scaler is not None

    start_time = time.time()

    for batch_idx, batch in enumerate(loader):
        # Move data to device
        images = batch["image"].to(device, non_blocking=True)        # [B, 1, H, W]
        masks = batch["frangi_mask"].to(device, non_blocking=True)   # [B, 1, H, W]

        # Generate 2 views + corresponding mask crops
        views, mask_views = view_generator(images, masks)
        # views: list of 2 [B, 1, H', W']
        # mask_views: list of 2 [B, 1, H', W']

        # Forward pass with mixed precision
        with torch.autocast(
            device_type=device.type,
            enabled=mixed_precision,
            dtype=torch.bfloat16,
        ):
            out1 = model(views[0], mask_views[0])
            out2 = model(views[1], mask_views[1])

            # Vessel weights: average of both views
            # w_p from view 1 (spatially aligned with view1 projections)
            w_p = 0.5 * (out1["vessel_weights"] + out2["vessel_weights"])  # [B, N]

            # Conditional loss
            losses = loss_fn(out1["proj"], out2["proj"], w_p)
            loss = losses["loss"]

        # Backward pass
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None and mixed_precision:
            scaler.scale(loss).backward()
            if cfg["training"]["grad_clip"] > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg["training"]["grad_clip"]
                )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if cfg["training"]["grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg["training"]["grad_clip"]
                )
            optimizer.step()

        # Accumulate metrics
        total_loss += loss.item()
        total_sigreg += losses["loss_sigreg_raw"].item()
        total_vicreg += losses["loss_vicreg_raw"].item()
        total_vessel_patches += losses["n_vessel_patches"]
        total_patches += losses["n_total_patches"]
        num_batches += 1

        # Logging
        if batch_idx % log_freq == 0:
            elapsed = time.time() - start_time
            lr = optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch {epoch:3d} [{batch_idx:4d}/{len(loader):4d}] "
                f"loss={loss.item():.4f} "
                f"sigreg={losses['loss_sigreg_raw'].item():.4f} "
                f"vicreg={losses['loss_vicreg_raw'].item():.4f} "
                f"vessel%={100.0 * losses['n_vessel_patches'] / max(losses['n_total_patches'], 1):.1f} "
                f"lr={lr:.2e} "
                f"time={elapsed:.1f}s"
            )

            if writer is not None:
                step = epoch * len(loader) + batch_idx
                writer.add_scalar("train/loss", loss.item(), step)
                writer.add_scalar("train/loss_sigreg_raw", losses["loss_sigreg_raw"].item(), step)
                writer.add_scalar("train/loss_vicreg_raw", losses["loss_vicreg_raw"].item(), step)
                writer.add_scalar("train/vessel_pct", 100.0 * losses["n_vessel_patches"] / max(losses["n_total_patches"], 1), step)
                writer.add_scalar("train/lr", lr, step)

        if cfg.get("debug", False) and batch_idx >= 3:
            break

    avg_loss = total_loss / max(num_batches, 1)
    avg_sigreg = total_sigreg / max(num_batches, 1)
    avg_vicreg = total_vicreg / max(num_batches, 1)
    vessel_pct = 100.0 * total_vessel_patches / max(total_patches, 1)

    return {
        "loss": avg_loss,
        "sigreg_raw": avg_sigreg,
        "vicreg_raw": avg_vicreg,
        "vessel_pct": vessel_pct,
    }


def validate(
    model: nn.Module,
    loss_fn: FrangiMaskedConditionalLoss,
    loader: DataLoader,
    view_generator: XcaViewGenerator,
    device: torch.device,
    cfg: dict,
) -> Dict[str, float]:
    """
    Evaluate on validation set (no gradients).

    Args:
        model: FrangiMaskedJEPA
        loss_fn: FrangiMaskedConditionalLoss
        loader: validation DataLoader
        view_generator: XcaViewGenerator
        device: target device
        cfg: config dict

    Returns:
        dict of average validation metrics
    """
    model.eval()
    loss_fn.eval()

    total_loss = 0.0
    total_sigreg = 0.0
    total_vicreg = 0.0
    num_batches = 0
    mixed_precision = cfg["training"]["mixed_precision"]

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["frangi_mask"].to(device, non_blocking=True)

            views, mask_views = view_generator(images, masks)

            with torch.autocast(
                device_type=device.type,
                enabled=mixed_precision,
                dtype=torch.bfloat16,
            ):
                out1 = model(views[0], mask_views[0])
                out2 = model(views[1], mask_views[1])
                w_p = 0.5 * (out1["vessel_weights"] + out2["vessel_weights"])
                losses = loss_fn(out1["proj"], out2["proj"], w_p)

            total_loss += losses["loss"].item()
            total_sigreg += losses["loss_sigreg_raw"].item()
            total_vicreg += losses["loss_vicreg_raw"].item()
            num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    avg_sigreg = total_sigreg / max(num_batches, 1)
    avg_vicreg = total_vicreg / max(num_batches, 1)

    return {
        "loss": avg_loss,
        "sigreg_raw": avg_sigreg,
        "vicreg_raw": avg_vicreg,
    }


def main():
    """Full training orchestration."""
    args = parse_args()
    cfg = load_config(args.config)

    # Debug mode overrides
    if args.debug:
        cfg["training"]["epochs"] = 2
        cfg["data"]["loader"]["batch_size"] = 2
        cfg["logging"]["log_freq"] = 1
        cfg["debug"] = True

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger.info(f"Config: {cfg}")

    # Seed
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Build components
    model, loss_fn, optimizer, scheduler = build_model(cfg, device)
    train_loader, val_loader, view_generator = build_dataloaders(cfg)

    logger.info(
        f"Model parameters: {sum(p.numel() for p in model.parameters()):,}"
    )
    logger.info(
        f"Train samples: {len(train_loader.dataset)}, "
        f"Batches per epoch: {len(train_loader)}"
    )

    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if (
        cfg["training"]["mixed_precision"] and device.type == "cuda"
    ) else None
    if scaler:
        logger.info("Using mixed precision training (GradScaler + autocast)")

    # TensorBoard
    writer = SummaryWriter(log_dir="./logs") if cfg["logging"].get("use_wandb", False) else None

    # Resume from checkpoint
    start_epoch = 0
    best_loss = float("inf")
    resume_path = args.resume or cfg["checkpoint"].get("resume")
    if resume_path:
        epoch_loaded, metrics, _ = load_checkpoint(
            resume_path, model, optimizer, scheduler, device
        )
        start_epoch = epoch_loaded + 1
        if metrics:
            best_loss = metrics.get("loss", best_loss)
        logger.info(f"Resumed from epoch {epoch_loaded}")

    # Training loop
    logger.info("=" * 60)
    logger.info("Starting training")
    logger.info("=" * 60)

    for epoch in range(start_epoch, cfg["training"]["epochs"]):
        epoch_start = time.time()

        # Train
        train_metrics = train_epoch(
            model=model,
            loss_fn=loss_fn,
            loader=train_loader,
            view_generator=view_generator,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
            cfg=cfg,
            writer=writer,
        )

        # LR scheduler step
        scheduler.step()

        # Validate
        val_metrics = {}
        if val_loader is not None:
            val_metrics = validate(
                model=model,
                loss_fn=loss_fn,
                loader=val_loader,
                view_generator=view_generator,
                device=device,
                cfg=cfg,
            )

        epoch_time = time.time() - epoch_start

        # Log epoch summary
        logger.info(
            f"Epoch {epoch:3d}/{cfg['training']['epochs'] - 1} "
            f"| Train loss: {train_metrics['loss']:.4f} "
            f"(sigreg={train_metrics['sigreg_raw']:.4f}, "
            f"vicreg={train_metrics['vicreg_raw']:.4f}, "
            f"vessel%={train_metrics['vessel_pct']:.1f})"
            + (f" | Val loss: {val_metrics.get('loss', 0):.4f}" if val_metrics else "")
            + f" | Time: {epoch_time:.1f}s"
        )

        if writer is not None:
            writer.add_scalar("epoch/train_loss", train_metrics["loss"], epoch)
            writer.add_scalar("epoch/val_loss", val_metrics.get("loss", 0), epoch)
            writer.add_scalar("epoch/lr", optimizer.param_groups[0]["lr"], epoch)

        # Checkpoint
        save_every = cfg["checkpoint"]["save_every"]
        is_best = val_metrics.get("loss", train_metrics["loss"]) < best_loss

        if is_best:
            best_loss = val_metrics.get("loss", train_metrics["loss"])
            save_checkpoint(
                f"checkpoints/best_model.pt",
                model, optimizer, scheduler, epoch,
                {**train_metrics, **val_metrics}, cfg,
            )

        if (epoch + 1) % save_every == 0:
            save_checkpoint(
                f"checkpoints/epoch_{epoch:04d}.pt",
                model, optimizer, scheduler, epoch,
                {**train_metrics, **val_metrics}, cfg,
            )

    # Final save
    save_checkpoint(
        "checkpoints/final_model.pt",
        model, optimizer, scheduler,
        cfg["training"]["epochs"] - 1,
        {"loss": train_metrics["loss"]}, cfg,
    )

    # Save backbone for segmentation
    from utils.checkpointing import save_backbone_for_segmentation
    save_backbone_for_segmentation(
        "checkpoints/backbone_for_segmentation.pth", model
    )

    logger.info("Training complete!")

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
