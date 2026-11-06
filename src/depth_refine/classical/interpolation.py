"""
Classical Depth Interpolation Baselines:
- Nearest-Neighbour Interpolation
- Bilinear Interpolation (Mask-Normalized)
- Bicubic Interpolation (Mask-Normalized & Clamped)
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import torch
import torch.nn.functional as F


def interpolate_depth(
    depth_lr: torch.Tensor,
    scale: Optional[int] = None,
    target_size: Optional[Tuple[int, int]] = None,
    mode: Literal["nearest", "bilinear", "bicubic"] = "bicubic",
    valid_mask_lr: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Upsample a low-resolution or sparse depth map using classical interpolation baselines.
    Automatically handles missing values (0s) via normalized convolution before interpolation
    so zero-holes do not drag down valid boundary depths.

    Args:
        depth_lr: (B, 1, h, w) or (1, h, w) low-resolution depth tensor.
        scale: Upsampling factor (e.g. 2, 4, 8). Ignored if `target_size` is provided.
        target_size: Explicit (H, W) output dimensions.
        mode: 'nearest', 'bilinear', or 'bicubic'.
        valid_mask_lr: Optional binary mask of valid pixels (default: depth_lr > 1e-3).

    Returns:
        Upsampled depth tensor of shape (B, 1, H, W) or (1, H, W).
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

    # Pre-fill missing holes at LR using iterative normalized neighborhood averaging
    filled_lr = depth_lr * valid_mask_lr
    mask_cur = valid_mask_lr.clone()
    for _ in range(4):
        if (mask_cur > 0.5).all():
            break
        avg_num = F.avg_pool2d(filled_lr, kernel_size=3, stride=1, padding=1, count_include_pad=False) * 9.0
        avg_den = F.avg_pool2d(mask_cur, kernel_size=3, stride=1, padding=1, count_include_pad=False) * 9.0
        fillable = (mask_cur <= 0.5) & (avg_den > 0.5)
        filled_lr = torch.where(fillable, avg_num / torch.clamp(avg_den, min=1e-5), filled_lr)
        mask_cur = torch.where(fillable, torch.ones_like(mask_cur), mask_cur)

    # Fallback global median/mean if completely disconnected empty regions remain
    if (mask_cur <= 0.5).any():
        valid_vals = filled_lr[mask_cur > 0.5]
        fallback_val = valid_vals.median() if valid_vals.numel() > 0 else torch.tensor(2.0, device=depth_lr.device)
        filled_lr = torch.where(mask_cur > 0.5, filled_lr, fallback_val)

    valid_min = filled_lr.min()
    valid_max = filled_lr.max()

    if mode == "nearest":
        out = F.interpolate(filled_lr, size=target_size, mode="nearest")
    elif mode == "bilinear":
        out = F.interpolate(filled_lr, size=target_size, mode="bilinear", align_corners=False)
    elif mode == "bicubic":
        out = F.interpolate(filled_lr, size=target_size, mode="bicubic", align_corners=False)
        # Clamp bicubic ringing overshoot to valid sensor range
        out = torch.clamp(out, min=valid_min, max=valid_max)
    else:
        raise ValueError(f"Unsupported interpolation mode: {mode}")

    return out.squeeze(0) if squeeze else out

