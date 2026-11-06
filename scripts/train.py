"""
Multi-Scale Training Pipeline for EG-DepthNet (2x, 4x, 8x Joint Training).

Optimized for:
- Teammate's NVIDIA RTX 5060 (8GB VRAM, 24GB RAM) via `--config configs/rtx5060_nyuv2.yaml`
- Local NVIDIA RTX 4050 (6GB VRAM) via `--quick-pretrain` for fast checkpoint initialization & verification.
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from rich.console import Console
from rich.table import Table
from torch.utils.data import DataLoader

from depth_refine.data.dataset_registry import load_dataset
from depth_refine.data.degradation import DegradationConfig
from depth_refine.evaluate.metrics import compute_depth_metrics
from depth_refine.models.eg_depth_net import EGDepthNet
from depth_refine.models.losses import GeometryAwareDepthLoss

console = Console()


@torch.no_grad()
def update_ema(ema_model: EGDepthNet, model: EGDepthNet, decay: float = 0.995) -> None:
    """Update Exponential Moving Average (EMA) model weights."""
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.data.mul_(decay).add_(p.data, alpha=1.0 - decay)


@torch.no_grad()
def validate(
    model: EGDepthNet,
    val_loader: DataLoader,
    device: torch.device,
    scale: int = 4,
) -> Dict[str, float]:
    model.eval()
    totals: Dict[str, float] = {}
    count = 0

    for batch in val_loader:
        depth_lr = batch["depth_lr"].to(device)
        valid_lr = batch["valid_mask_lr"].to(device)
        rgb_hr = batch["rgb_hr"].to(device)
        depth_gt = batch["depth_gt"].to(device)
        valid_gt = batch["valid_mask_gt"].to(device)
        boundary_gt = batch["boundary_mask_gt"].to(device)

        out = model(depth_lr, rgb_hr=rgb_hr, scale=scale, valid_mask_lr=valid_lr)
        metrics = compute_depth_metrics(
            pred=out.depth_hr,
            gt=depth_gt,
            valid_mask=valid_gt,
            boundary_mask=boundary_gt,
            rgb_hr=rgb_hr,
        )
        for k, v in metrics.items():
            totals[k] = totals.get(k, 0.0) + v
        count += 1

    return {k: v / max(count, 1) for k, v in totals.items()}


def train(config: Dict[str, Any], quick_pretrain: bool = False) -> Path:
    hw_cfg = config.get("hardware", {})
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    loss_cfg = config.get("loss", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    amp_dtype = (
        torch.bfloat16
        if (use_amp and torch.cuda.is_bf16_supported() and hw_cfg.get("precision", "bf16") == "bf16")
        else torch.float16
    )

    epochs = 8 if quick_pretrain else int(train_cfg.get("epochs", 60))
    batch_size = 4 if quick_pretrain else int(train_cfg.get("batch_size", 12))
    crop_size = 192 if quick_pretrain else int(data_cfg.get("crop_size", 256))
    lr = float(train_cfg.get("learning_rate", 4e-4))
    save_dir = Path(train_cfg.get("save_dir", "checkpoints"))
    save_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = save_dir / "eg_depthnet_best.pt"

    deg = DegradationConfig(
        scale=4,
        hole_ratio=float(data_cfg.get("hole_ratio", 0.08)),
        structured_boundary_holes=bool(data_cfg.get("structured_boundary_holes", True)),
        noise_std=float(data_cfg.get("noise_std", 0.015)),
        outlier_prob=float(data_cfg.get("outlier_prob", 0.005)),
    )

    dataset_name = "synthetic" if quick_pretrain else data_cfg.get("train_dataset", "nyuv2")
    train_ds = load_dataset(
        name=dataset_name,
        split="train",
        root_dir=data_cfg.get("root_dir", "data"),
        degradation=deg,
        crop_size=crop_size,
        max_samples=48 if quick_pretrain else None,
        multi_scale_choices=None,  # Batch-uniform scale sampled per step below
        fallback_to_synthetic=True,
    )
    val_ds = load_dataset(
        name=dataset_name,
        split="test",
        root_dir=data_cfg.get("root_dir", "data"),
        degradation=DegradationConfig(scale=4, hole_ratio=0.08, noise_std=0.015),
        crop_size=crop_size,
        max_samples=12 if quick_pretrain else 64,
        fallback_to_synthetic=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0 if quick_pretrain else int(hw_cfg.get("num_workers", 2)),
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    model = EGDepthNet(
        base_channels=int(model_cfg.get("base_channels", 48)),
        prop_steps=int(model_cfg.get("prop_steps", 6)),
    ).to(device)

    if best_ckpt_path.exists():
        try:
            ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
            console.print(f"[cyan]Warm-started from existing checkpoint {best_ckpt_path}[/cyan]")
        except Exception:
            pass

    ema_model = copy.deepcopy(model).to(device)
    ema_decay = float(model_cfg.get("ema_decay", 0.99))

    criterion = GeometryAwareDepthLoss(
        w_charbonnier=float(loss_cfg.get("w_charbonnier", 1.0)),
        w_grad=float(loss_cfg.get("w_grad", 0.65)),
        w_normal=float(loss_cfg.get("w_normal", 0.35)),
        w_smooth=float(loss_cfg.get("w_smooth", 0.15)),
        w_uncertainty=float(loss_cfg.get("w_uncertainty", 0.05)),
        boundary_weight_multiplier=float(loss_cfg.get("boundary_weight_multiplier", 3.0)),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=float(train_cfg.get("min_lr", 1e-5))
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

    console.print(
        f"[bold green]Starting EG-DepthNet Training on {device} "
        f"(AMP={amp_dtype if use_amp else 'fp32'}, Epochs={epochs}, Samples={len(train_ds)})[/bold green]"
    )

    best_rmse = float("inf")
    scales_cycle = [2, 4, 8]

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.perf_counter()
        running_loss = 0.0

        for step, batch in enumerate(train_loader):
            # Dynamically train across 2x, 4x, and 8x upsampling factors
            scale_step = scales_cycle[(epoch + step) % len(scales_cycle)]
            rgb_hr = batch["rgb_hr"].to(device, non_blocking=True)
            depth_gt = batch["depth_gt"].to(device, non_blocking=True)
            valid_gt = batch["valid_mask_gt"].to(device, non_blocking=True)
            boundary_gt = batch["boundary_mask_gt"].to(device, non_blocking=True)

            # Downsample batch cleanly to `scale_step`
            H, W = depth_gt.shape[-2], depth_gt.shape[-1]
            h_s, w_s = H // scale_step, W // scale_step
            depth_lr = torch.nn.functional.adaptive_avg_pool2d(depth_gt, (h_s, w_s))
            valid_lr = (torch.nn.functional.adaptive_avg_pool2d(valid_gt, (h_s, w_s)) > 0.5).float()
            # Add sensor noise & random holes during training
            noise = torch.randn_like(depth_lr) * 0.015 * valid_lr
            drop = (torch.rand_like(valid_lr) > 0.08).float()
            valid_lr = valid_lr * drop
            depth_lr = torch.clamp(depth_lr + noise, min=0.1, max=10.0) * valid_lr

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                out = model(depth_lr, rgb_hr=rgb_hr, scale=scale_step, valid_mask_lr=valid_lr)
                loss_dict = criterion(
                    out,
                    depth_gt=depth_gt,
                    valid_mask_gt=valid_gt,
                    boundary_mask_gt=boundary_gt,
                )
                loss = loss_dict["total"]

            if use_amp and amp_dtype == torch.float16:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            update_ema(ema_model, model, decay=ema_decay)
            running_loss += float(loss.item())

        scheduler.step()
        avg_loss = running_loss / max(len(train_loader), 1)
        val_metrics = validate(ema_model, val_loader, device=device, scale=4)
        dt = time.perf_counter() - t0

        console.print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({dt:.1f}s) | Loss: {avg_loss:.4f} | "
            f"Val 4x RMSE: {val_metrics['RMSE']:.4f}m | Boundary-RMSE: {val_metrics['Boundary_RMSE']:.4f}m | "
            f"PSNR: {val_metrics['PSNR']:.2f}dB | Edge-F1: {val_metrics['Edge_F1']:.4f}"
        )

        if val_metrics["RMSE"] <= best_rmse:
            best_rmse = val_metrics["RMSE"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "ema_state_dict": ema_model.state_dict(),
                    "val_metrics": val_metrics,
                    "config": config,
                },
                best_ckpt_path,
            )

    console.print(f"[bold green]✓ Saved best checkpoint to {best_ckpt_path} (Best 4x RMSE={best_rmse:.4f}m)[/bold green]")
    return best_ckpt_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train EG-DepthNet on RTX 5060 / RTX 4050.")
    parser.add_argument("--config", type=str, default="configs/rtx5060_nyuv2.yaml", help="Path to YAML config")
    parser.add_argument(
        "--quick-pretrain",
        action="store_true",
        help="Run a fast 8-epoch multi-scale pre-training pass on 3D geometric scenes (<40s)",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if cfg_path.exists():
        config = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    else:
        config = {}

    train(config, quick_pretrain=args.quick_pretrain)


if __name__ == "__main__":
    main()
