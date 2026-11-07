"""
Multi-Scale Training Pipeline for EG-DepthNet (2x, 4x, 8x Joint Training) with Early Stopping
and Automated Training/Validation Curve Plotting.

Optimized for:
- Teammate's NVIDIA RTX 5060 (8GB VRAM, 24GB RAM) via `--config configs/rtx5060_nyuv2.yaml`
- Local NVIDIA RTX 4050 (6GB VRAM) via `--quick-pretrain`
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import torch
import yaml
from rich.console import Console
from torch.utils.data import DataLoader

from depth_refine.data.dataset_registry import load_dataset
from depth_refine.data.degradation import DegradationConfig, degrade_depth
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


def save_training_curves(history: List[Dict[str, Any]], output_dir: Path = Path("outputs")) -> None:
    """Save training & validation learning curves to JSON and PNG."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "training_history.json"
    json_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    if len(history) < 2:
        return

    epochs = [h["epoch"] for h in history]
    losses = [h["train_loss"] for h in history]
    rmses = [h["val_rmse"] for h in history]
    b_rmses = [h["val_boundary_rmse"] for h in history]
    psnrs = [h["val_psnr"] for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), dpi=140)
    axes[0].plot(epochs, losses, marker="o", color="#2563eb", linewidth=2, label="Train Composite Loss")
    axes[0].set_title("Training Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, rmses, marker="s", color="#059669", linewidth=2, label="Val 4× RMSE (m)")
    axes[1].plot(epochs, b_rmses, marker="^", color="#d97706", linewidth=2, label="Val 4× Boundary-RMSE (m)")
    axes[1].set_title("Validation RMSE & Boundary-RMSE ↓")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Error (meters)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(epochs, psnrs, marker="d", color="#7c3aed", linewidth=2, label="Val 4× PSNR (dB)")
    axes[2].set_title("Validation PSNR (dB) ↑")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("PSNR (dB)")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", bbox_inches="tight")
    plt.close(fig)


def train(config: Dict[str, Any], quick_pretrain: bool = False, resume_path: Optional[str] = None) -> Path:
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
    patience = int(train_cfg.get("early_stopping_patience", 12))
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
    # Strictly split train (first 795 on NYUv2) vs test/val (remaining 654 held-out scenes)
    train_ds = load_dataset(
        name=dataset_name,
        split="train",
        root_dir=data_cfg.get("root_dir", "data"),
        degradation=deg,
        crop_size=crop_size,
        max_samples=48 if quick_pretrain else None,
        fallback_to_synthetic=True,
    )
    val_ds = load_dataset(
        name=dataset_name,
        split="test",
        root_dir=data_cfg.get("root_dir", "data"),
        degradation=DegradationConfig(scale=4, hole_ratio=0.08, noise_std=0.015, seed=42),
        crop_size=crop_size,
        max_samples=12 if quick_pretrain else 64,
        fallback_to_synthetic=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0 if (quick_pretrain or sys.platform == "win32") else int(hw_cfg.get("num_workers", 0)),
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    model = EGDepthNet(
        base_channels=int(model_cfg.get("base_channels", 48)),
        prop_steps=int(model_cfg.get("prop_steps", 4)),
    ).to(device)

    ema_model = copy.deepcopy(model).to(device)
    ema_decay = float(model_cfg.get("ema_decay", 0.992))

    best_rmse = float("inf")
    baseline_rmse = float("inf")
    history: List[Dict[str, Any]] = []
    start_epoch = 1

    if resume_path and Path(resume_path).exists():
        console.print(f"[bold cyan]Resuming/fine-tuning weights from checkpoint: {resume_path}[/bold cyan]")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        if "model_state_dict" in ckpt:
            res_m = model.load_state_dict(ckpt["model_state_dict"], strict=False)
            if res_m.missing_keys:
                console.print(f"[yellow]Loaded with newly initialized keys: {res_m.missing_keys}[/yellow]")
        if "ema_state_dict" in ckpt:
            ema_model.load_state_dict(ckpt["ema_state_dict"], strict=False)
        if "val_metrics" in ckpt and "RMSE" in ckpt["val_metrics"]:
            baseline_rmse = float(ckpt["val_metrics"]["RMSE"])
            console.print(f"[bold cyan]Baseline checkpoint Best RMSE: {baseline_rmse:.4f}m[/bold cyan]")
        if "history" in ckpt and isinstance(ckpt["history"], list):
            history = list(ckpt["history"])

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
        f"(AMP={amp_dtype if use_amp else 'fp32'}, Epochs={epochs}, EarlyStopPatience={patience}, "
        f"TrainSamples={len(train_ds)}, HeldOutValSamples={len(val_ds)})[/bold green]"
    )

    epochs_without_improvement = 0
    scales_cycle = [2, 4, 8]
    end_epoch = start_epoch + epochs - 1

    for epoch in range(start_epoch, start_epoch + epochs):
        model.train()
        t0 = time.perf_counter()
        running_loss = 0.0

        for step, batch in enumerate(train_loader):
            scale_step = scales_cycle[(epoch + step) % len(scales_cycle)]
            rgb_hr = batch["rgb_hr"].to(device, non_blocking=True)
            depth_gt = batch["depth_gt"].to(device, non_blocking=True)
            valid_gt = batch["valid_mask_gt"].to(device, non_blocking=True)
            boundary_gt = batch["boundary_mask_gt"].to(device, non_blocking=True)

            step_deg = degrade_depth(
                depth_gt,
                rgb_hr,
                DegradationConfig(
                    scale=scale_step,
                    hole_ratio=deg.hole_ratio,
                    structured_boundary_holes=True,
                    noise_std=deg.noise_std,
                    outlier_prob=deg.outlier_prob,
                ),
            )

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                out = model(
                    step_deg.depth_lr,
                    rgb_hr=rgb_hr,
                    scale=scale_step,
                    valid_mask_lr=step_deg.valid_mask_lr,
                )
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

        history.append(
            {
                "epoch": epoch,
                "train_loss": avg_loss,
                "val_rmse": val_metrics["RMSE"],
                "val_boundary_rmse": val_metrics["Boundary_RMSE"],
                "val_psnr": val_metrics["PSNR"],
                "val_edge_f1": val_metrics["Edge_F1"],
            }
        )
        save_training_curves(history)

        console.print(
            f"Epoch [{epoch:02d}/{end_epoch:02d}] ({dt:.1f}s) | Loss: {avg_loss:.4f} | "
            f"Val 4x RMSE: {val_metrics['RMSE']:.4f}m | Boundary-RMSE: {val_metrics['Boundary_RMSE']:.4f}m | "
            f"PSNR: {val_metrics['PSNR']:.2f}dB | Edge-F1: {val_metrics['Edge_F1']:.4f}"
        )

        improved = val_metrics["RMSE"] < best_rmse - 1e-5
        if improved:
            best_rmse = val_metrics["RMSE"]
            epochs_without_improvement = 0
            console.print(f"[bold green]★ New Best Validation RMSE: {best_rmse:.4f}m (Saved to {best_ckpt_path})[/bold green]")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "ema_state_dict": ema_model.state_dict(),
                    "val_metrics": val_metrics,
                    "history": history,
                    "config": config,
                },
                best_ckpt_path,
            )
        else:
            epochs_without_improvement += 1
            console.print(f"[yellow]No improvement for {epochs_without_improvement}/{patience} epoch(s) (Best: {best_rmse:.4f}m)[/yellow]")
            if not quick_pretrain and epochs_without_improvement >= patience:
                console.print(
                    f"[bold yellow]Early stopping triggered at epoch {epoch} "
                    f"(no improvement for {patience} epochs).[/bold yellow]"
                )
                break

    console.print(
        f"[bold green]✓ Saved best checkpoint to {best_ckpt_path} (Best 4x RMSE={best_rmse:.4f}m)[/bold green]"
    )
    return best_ckpt_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train EG-DepthNet on RTX 5060 / RTX 4050.")
    parser.add_argument("--config", type=str, default="configs/rtx5060_nyuv2.yaml", help="Path to YAML config")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override training batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--patience", type=int, default=None, help="Override early stopping patience")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training or fine-tune from",
    )
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

    if args.epochs is not None:
        if "training" not in config:
            config["training"] = {}
        config["training"]["epochs"] = args.epochs

    if args.batch_size is not None:
        if "training" not in config:
            config["training"] = {}
        config["training"]["batch_size"] = args.batch_size

    if args.lr is not None:
        if "training" not in config:
            config["training"] = {}
        config["training"]["learning_rate"] = args.lr

    if args.patience is not None:
        if "training" not in config:
            config["training"] = {}
        config["training"]["early_stopping_patience"] = args.patience

    train(config, quick_pretrain=args.quick_pretrain, resume_path=args.resume)


if __name__ == "__main__":
    main()
