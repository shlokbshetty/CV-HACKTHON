"""
Generate high-resolution benchmark and qualitative visual diagrams in outputs/
for ProbID-P4 (NYUv2, Middlebury, and KITTI).
"""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch

from depth_refine.classical.filters import reconstruct_classical
from depth_refine.data.dataset_registry import load_dataset
from depth_refine.data.degradation import DegradationConfig
from depth_refine.hybrid.hybrid_pipeline import HybridReconstructor


def colorize(depth: np.ndarray, vmin: float | None = None, vmax: float | None = None) -> np.ndarray:
    valid = depth[(depth > 0.05) & np.isfinite(depth)]
    v_min = float(np.percentile(valid, 2)) if valid.size > 0 else 0.5
    v_max = float(np.percentile(valid, 98)) if valid.size > 0 else 5.0
    if vmin is not None:
        v_min = vmin
    if vmax is not None:
        v_max = vmax
    norm = 1.0 - np.clip((depth - v_min) / max(v_max - v_min, 1e-6), 0.0, 1.0)
    cmap = plt.get_cmap("turbo")
    return cmap(norm)[:, :, :3]


def generate_qualitative_figures(checkpoint_path: str = "checkpoints/eg_depthnet_best.pt", output_dir: str = "outputs") -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    recon = HybridReconstructor(checkpoint_path=checkpoint_path)

    datasets = [
        ("nyuv2", "NYU Depth V2 (Indoor Kinect RGB-D)", "qualitative_nyuv2.png"),
        ("middlebury", "Middlebury Stereo Benchmark (Discontinuity Focus)", "qualitative_middlebury.png"),
        ("kitti", "KITTI Vision Suite (Outdoor Sparse LiDAR)", "qualitative_kitti.png"),
    ]

    for ds_name, title, filename in datasets:
        try:
            ds = load_dataset(ds_name, split="test", degradation=DegradationConfig(scale=4, hole_ratio=0.10, noise_std=0.015), max_samples=1)
            sample = ds[0]
            rgb = sample["rgb_hr"]
            depth_lr = sample["depth_lr"]
            valid_lr = sample["valid_mask_lr"]
            depth_gt = sample["depth_gt"]

            if rgb.ndim == 3:
                rgb_in = rgb.unsqueeze(0)
                rgb_np = rgb.permute(1, 2, 0).cpu().numpy()
            else:
                rgb_in = rgb
                rgb_np = rgb[0].permute(1, 2, 0).cpu().numpy()

            if depth_lr.ndim == 3:
                depth_lr_in = depth_lr.unsqueeze(0)
            else:
                depth_lr_in = depth_lr

            if valid_lr.ndim == 3:
                valid_lr_in = valid_lr.unsqueeze(0)
            else:
                valid_lr_in = valid_lr

            # Classical Baselines
            b_bicubic = reconstruct_classical(depth_lr_in, rgb_in, scale=4, method="bicubic", valid_mask_lr=valid_lr_in)
            b_mb = reconstruct_classical(depth_lr_in, rgb_in, scale=4, method="mask_bicubic", valid_mask_lr=valid_lr_in)

            # Deep & Hybrid Reconstructions
            res = recon.reconstruct(depth_lr_in, rgb_in, scale=4, valid_mask_lr=valid_lr_in)
            d_deep = res.depth_deep.squeeze().cpu().numpy()
            d_hyb = res.depth_hybrid.squeeze().cpu().numpy()
            d_gt = depth_gt.squeeze().cpu().numpy()
            d_bic = b_bicubic.squeeze().cpu().numpy()
            d_mb = b_mb.squeeze().cpu().numpy()

            d_lr_up = torch.nn.functional.interpolate(depth_lr_in, size=rgb_in.shape[-2:], mode="nearest").squeeze().cpu().numpy()

            fig, axes = plt.subplots(1, 6, figsize=(20, 3.6), dpi=150)
            fig.suptitle(f"{title} — 4× Upsampling & Discontinuity Refinement", fontsize=14, fontweight="bold", y=1.03)

            axes[0].imshow(np.clip(rgb_np, 0.0, 1.0))
            axes[0].set_title("Input High-Res RGB", fontsize=11)
            axes[0].axis("off")

            axes[1].imshow(colorize(d_lr_up, vmin=float(d_gt.min()), vmax=float(d_gt.max())))
            axes[1].set_title("Degraded 4× LR Input", fontsize=11)
            axes[1].axis("off")

            axes[2].imshow(colorize(d_bic, vmin=float(d_gt.min()), vmax=float(d_gt.max())))
            axes[2].set_title("Bicubic Baseline (Blurred)", fontsize=11)
            axes[2].axis("off")

            axes[3].imshow(colorize(d_mb, vmin=float(d_gt.min()), vmax=float(d_gt.max())))
            axes[3].set_title("Mask-Bicubic (Stage 1)", fontsize=11)
            axes[3].axis("off")

            axes[4].imshow(colorize(d_hyb, vmin=float(d_gt.min()), vmax=float(d_gt.max())))
            axes[4].set_title("Hybrid EG-DepthNet (Ours)", fontsize=11, fontweight="bold", color="#059669")
            axes[4].axis("off")

            axes[5].imshow(colorize(d_gt, vmin=float(d_gt.min()), vmax=float(d_gt.max())))
            axes[5].set_title("Ground Truth Target", fontsize=11)
            axes[5].axis("off")

            plt.tight_layout()
            out_file = out_dir / filename
            plt.savefig(out_file, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {out_file}")
        except Exception as e:
            print(f"Skipping {ds_name} qualitative figure due to: {e}")


if __name__ == "__main__":
    generate_qualitative_figures()
