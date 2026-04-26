"""
Checkpoint save/load utilities for Frangi-Masked SIGReg pretraining.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    metrics: Dict[str, float],
    cfg: dict,
) -> None:
    """
    Save full training state to disk.

    Args:
        path: checkpoint file path
        model: model (state_dict only)
        optimizer: optimizer state
        scheduler: learning rate scheduler state
        epoch: current epoch number
        metrics: dict of metrics (loss, etc.)
        cfg: full experiment config
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "metrics": metrics,
        "config": cfg,
    }

    torch.save(state, str(path))
    print(f"Checkpoint saved to {path}")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Any = None,
    device: torch.device = torch.device("cpu"),
) -> Tuple[int, Dict[str, float], Optional[dict]]:
    """
    Load training state from checkpoint.

    Args:
        path: checkpoint file path
        model: model to load weights into
        optimizer: optimizer to load state into (optional)
        scheduler: scheduler to load state into (optional)
        device: device to map tensors to

    Returns:
        epoch: int — last completed epoch
        metrics: dict — last logged metrics
        cfg: dict — config from checkpoint (optional)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    state = torch.load(str(path), map_location=device, weights_only=False)

    model.load_state_dict(state["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in state:
        scheduler.load_state_dict(state["scheduler_state_dict"])

    epoch = state.get("epoch", 0)
    metrics = state.get("metrics", {})
    cfg = state.get("config", None)

    print(f"Checkpoint loaded from {path} (epoch {epoch})")
    return epoch, metrics, cfg


def save_backbone_for_segmentation(
    path: str,
    model: nn.Module,
) -> None:
    """
    Extract and save only the backbone (encoder) weights for downstream
    segmentation fine-tuning. This strips the projection head.

    The saved state_dict can be loaded into any HierarchicalViT instance.

    Args:
        path: output .pth file path
        model: full FrangiMaskedJEPA model
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    backbone_state = model.backbone.state_dict()
    torch.save(backbone_state, str(path))
    print(f"Backbone weights saved to {path}")
