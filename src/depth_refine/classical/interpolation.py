"""
Classical Depth Interpolation Baselines:
- Nearest-Neighbour Interpolation
- Bilinear Interpolation (Multi-Scale Pyramid Hole-Normalized)
- Bicubic Interpolation (Multi-Scale Pyramid Hole-Normalized & Clamped)
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import torch
import torch.nn.functional as F


def _pyramid_fill_holes_lr(depth_lr: torch.Tensor, valid_mask_lr: torch.Tensor) -> torch.Tensor:
    """
    Fill missing sensor pockets (0-holes) in low-resolution space using a coarse-to-fine
    spatial pyramid followed by harmonic boundary diffusion.
    Eliminates flat global-median "pancake" artifacts inside wide Kinect/LiDAR shadow holes.
    """
    filled = depth_lr * valid_mask_lr
    mask_cur = (valid_mask_lr > 0.5).to(depth_lr.dtype)

    if (mask_cur > 0.5).all():
        return filled

    _, _, h, w = depth_lr.shape

    # 1. Local 3x3 normalized propagation for narrow holes
    for _ in range(4):
        if (mask_cur > 0.5).all():
            break
        avg_num = F.avg_pool2d(filled * mask_cur, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        avg_den = F.avg_pool2d(mask_cur, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        can_fill = (mask_cur <= 0.5) & (avg_den > 1e-4)
        filled = torch.where(can_fill, avg_num / torch.clamp(avg_den, min=1e-5), filled)
        mask_cur = torch.where(can_fill, torch.ones_like(mask_cur), mask_cur)

    # 2. Coarse-to-fine pyramid propagation for wide Kinect shadow holes
    if (mask_cur <= 0.5).any():
        for div in (16, 8, 4, 2):
            ph, pw = max(2, h // div), max(2, w // div)
            num_c = F.adaptive_avg_pool2d(filled * mask_cur, (ph, pw))
            den_c = F.adaptive_avg_pool2d(mask_cur, (ph, pw))
            valid_c = den_c > 1e-4
            val_c = torch.where(valid_c, num_c / torch.clamp(den_c, min=1e-5), torch.zeros_like(num_c))
            up_val = F.interpolate(val_c, size=(h, w), mode="bilinear", align_corners=False)
            up_ok = F.interpolate(valid_c.to(depth_lr.dtype), size=(h, w), mode="bilinear", align_corners=False) > 0.1
            fill_here = (mask_cur <= 0.5) & up_ok
            filled = torch.where(fill_here, up_val, filled)
            mask_cur = torch.where(fill_here, torch.ones_like(mask_cur), mask_cur)

    # 3. Fallback global median only if the entire image was empty
    if (mask_cur <= 0.5).any():
        valid_vals = filled[mask_cur > 0.5]
        fallback_val = valid_vals.median() if valid_vals.numel() > 0 else torch.tensor(2.0, device=depth_lr.device)
        filled = torch.where(mask_cur > 0.5, filled, fallback_val)

    # 4. Harmonic Laplace smoothing strictly inside originally missing hole pixels
    orig_valid = valid_mask_lr > 0.5
    for _ in range(4):
        smooth = F.avg_pool2d(F.pad(filled, (1, 1, 1, 1), mode="replicate"), kernel_size=3, stride=1)
        filled = torch.where(orig_valid, depth_lr, smooth)

    return filled


def interpolate_depth(
    depth_lr: torch.Tensor,
    scale: Optional[int] = None,
    target_size: Optional[Tuple[int, int]] = None,
    mode: Literal["nearest", "bilinear", "bicubic"] = "bicubic",
    valid_mask_lr: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Upsample a low-resolution or sparse depth map using classical interpolation baselines.
    Automatically handles missing values (0s) via multi-scale pyramid normalized inpainting
    before interpolation so zero-holes do not drag down valid boundary depths.
    """
    squeeze = False
    if depth_lr.ndim == 3:
        depth_lr = depth_lr.unsqueeze(0)
        if valid_mask_lr is not None and valid_mask_lr.ndim == 3:
            valid_mask_lr = valid_mask_lr.unsqueeze(0)
        squeeze = True

    _, _, h, w = depth_lr.shape
    if target_size is None:
        if scale is None:
            raise ValueError("Either `scale` or `target_size` must be specified.")
        target_size = (h * scale, w * scale)

    if valid_mask_lr is None:
        valid_mask_lr = (depth_lr > 1e-3).to(depth_lr.dtype)
    else:
        valid_mask_lr = valid_mask_lr.to(depth_lr.dtype)

    filled_lr = _pyramid_fill_holes_lr(depth_lr, valid_mask_lr)
    valid_min = filled_lr.min()
    valid_max = filled_lr.max()

    if mode == "nearest":
        out = F.interpolate(filled_lr, size=target_size, mode="nearest")
    elif mode == "bilinear":
        out = F.interpolate(filled_lr, size=target_size, mode="bilinear", align_corners=False)
    elif mode == "bicubic":
        out = F.interpolate(filled_lr, size=target_size, mode="bicubic", align_corners=False)
        out = torch.clamp(out, min=valid_min, max=valid_max)
    else:
        raise ValueError(f"Unsupported interpolation mode: {mode}")

    return out.squeeze(0) if squeeze else out
