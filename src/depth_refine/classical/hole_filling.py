"""
Classical Noise Removal, Outlier Suppression, and Hole Filling for Depth Maps:
1. Statistical / Median Outlier Spike Removal (`remove_depth_outliers`)
2. Fast Marching Method Inpainting (`telea`)
3. Navier-Stokes Fluid Dynamics Inpainting (`navier_stokes`)
4. Multi-Scale Cross-Bilateral Hole Filling (`multiscale_bilateral` - NYUv2 Toolbox equivalent)
5. Guided Affinity Colorization Inpainting (`colorization` - Levin et al. inspired)
"""

from __future__ import annotations

from typing import Literal, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from depth_refine.classical.interpolation import _pyramid_fill_holes_lr


def remove_depth_outliers(
    depth: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    jump_threshold: float = 0.65,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Detect and suppress isolated salt-and-pepper sensor outlier spikes and stray floating pixels
    while strictly preserving legitimate step edges (which have consistent contiguous support).

    Uses a dual-stage filter:
    1. Prune isolated pixels that have fewer than 2 valid 8-neighbors in a 3x3 window.
    2. Statistical Patch Median & MAD outlier suppression across valid 5x5 patches.

    Args:
        depth: (B, 1, H, W) or (1, H, W) depth tensor.
        valid_mask: Binary float mask of valid measurements.
        jump_threshold: Depth difference threshold in meters to flag an isolated spike.

    Returns:
        Tuple of (cleaned_depth, updated_valid_mask).
    """
    squeeze = False
    if depth.ndim == 3:
        depth = depth.unsqueeze(0)
        if valid_mask is not None and valid_mask.ndim == 3:
            valid_mask = valid_mask.unsqueeze(0)
        squeeze = True

    if valid_mask is None:
        valid_mask = ((depth > 1e-3) & torch.isfinite(depth)).to(depth.dtype)
    else:
        valid_mask = ((valid_mask > 0.5) & (depth > 1e-3) & torch.isfinite(depth)).to(depth.dtype)

    B, C, H, W = depth.shape
    d_clean = depth.clone()
    m_clean = valid_mask.clone()

    # Pass 1: Prune completely isolated floating pixels with < 2 valid 8-neighbors
    pad_m = F.pad(m_clean, (1, 1, 1, 1), mode="constant", value=0.0)
    patches_m = F.unfold(pad_m, kernel_size=3).view(B, 9, H, W)
    # Exclude center pixel (index 4)
    neighbor_valid_cnt = patches_m[:, :4, :, :].sum(dim=1, keepdim=True) + patches_m[:, 5:, :, :].sum(dim=1, keepdim=True)
    isolated = (m_clean > 0.5) & (neighbor_valid_cnt < 2)
    m_clean = torch.where(isolated, torch.zeros_like(m_clean), m_clean)
    d_clean = torch.where(isolated, torch.zeros_like(d_clean), d_clean)

    # Pass 2: Two-Iteration Valid-Masked 5x5 Patch Median & MAD Z-Score Outlier Rejection
    # Uses exact order-statistic median over valid pixels so outlier spikes never corrupt local statistics
    for _ in range(2):
        pad_d = F.pad(d_clean, (2, 2, 2, 2), mode="replicate")
        pad_m5 = F.pad(m_clean, (2, 2, 2, 2), mode="constant", value=0.0)
        patches_d5 = F.unfold(pad_d, kernel_size=5).view(B, 25, H, W)
        patches_m5 = F.unfold(pad_m5, kernel_size=5).view(B, 25, H, W)

        n_valid5 = patches_m5.sum(dim=1, keepdim=True)
        # Compute valid patch median by replacing invalid entries with +inf and indexing the middle valid rank
        valid_bool5 = patches_m5 > 0.5
        d_sorted, _ = torch.where(valid_bool5, patches_d5, torch.full_like(patches_d5, 1e6)).sort(dim=1)
        mid_idx = torch.clamp((n_valid5.long() - 1) // 2, min=0, max=24)
        median_d = torch.gather(d_sorted, dim=1, index=mid_idx)
        median_d = torch.where(n_valid5 > 0.5, median_d, d_clean)

        # Compute Median Absolute Deviation (MAD) over valid neighbors
        abs_dev = torch.abs(patches_d5 - median_d)
        dev_sorted, _ = torch.where(valid_bool5, abs_dev, torch.full_like(abs_dev, 1e6)).sort(dim=1)
        mad_d = torch.gather(dev_sorted, dim=1, index=mid_idx)
        mad_d = torch.where(n_valid5 > 0.5, mad_d, torch.zeros_like(mad_d))

        # Z-score threshold (robust to clustered spikes and high sparsity)
        z_thresh = torch.clamp(3.2 * (1.4826 * mad_d) + 0.12, min=0.18, max=jump_threshold)
        center_dev = torch.abs(d_clean - median_d)

        # Count how many valid neighbors in 5x5 agree with the center pixel's depth
        diff_center = torch.abs(patches_d5 - d_clean)
        agree_cnt = ((diff_center < 0.22) & valid_bool5).sum(dim=1, keepdim=True).to(depth.dtype)
        min_quorum = torch.clamp(0.24 * n_valid5, min=2.0, max=5.0)

        is_spike = (m_clean > 0.5) & (n_valid5 >= 3) & ((center_dev > z_thresh) | (agree_cnt < min_quorum)) & (center_dev > 0.18)
        d_clean = torch.where(is_spike, median_d, d_clean)

    if squeeze:
        return d_clean.squeeze(0), m_clean.squeeze(0)
    return d_clean, m_clean


def fill_depth_holes(
    depth: torch.Tensor,
    rgb: Optional[torch.Tensor] = None,
    valid_mask: Optional[torch.Tensor] = None,
    method: Literal["telea", "navier_stokes", "multiscale_bilateral", "colorization"] = "multiscale_bilateral",
) -> torch.Tensor:
    """
    Fill missing depth holes (where depth <= 0 or valid_mask == 0) using classical computer vision
    or RGB-guided colorization/cross-bilateral schemes.

    Args:
        depth: (B, 1, H, W) or (1, H, W) tensor.
        rgb: Optional (B, 3, H, W) or (3, H, W) RGB tensor in [0, 1] for guided hole filling.
        valid_mask: Optional valid mask tensor.
        method: Hole filling algorithm.

    Returns:
        Dense hole-filled depth tensor of same shape as `depth`.
    """
    squeeze = False
    if depth.ndim == 3:
        depth = depth.unsqueeze(0)
        if rgb is not None and rgb.ndim == 3:
            rgb = rgb.unsqueeze(0)
        if valid_mask is not None and valid_mask.ndim == 3:
            valid_mask = valid_mask.unsqueeze(0)
        squeeze = True

    if valid_mask is None:
        valid_mask = (depth > 1e-3).to(depth.dtype)
    else:
        valid_mask = valid_mask.to(depth.dtype)

    # First remove isolated sensor spikes
    depth_clean, mask_clean = remove_depth_outliers(depth, valid_mask)

    if method in ("telea", "navier_stokes"):
        out_list = []
        cv_flag = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
        for b in range(depth_clean.shape[0]):
            d_np = depth_clean[b, 0].detach().cpu().numpy().astype(np.float32)
            m_np = (mask_clean[b, 0].detach().cpu().numpy() <= 0.5).astype(np.uint8)
            if m_np.any() and (m_np == 0).any():
                filled_np = cv2.inpaint(d_np, m_np, inpaintRadius=7.0, flags=cv_flag)
                # Preserve valid measurements while preventing OpenCV from
                # producing physically invalid near-zero values in holes.
                filled_np = np.where(m_np > 0, np.maximum(filled_np, 0.3001), filled_np)
            else:
                filled_np = d_np
            out_list.append(torch.from_numpy(filled_np).to(device=depth.device, dtype=depth.dtype))
        out = torch.stack(out_list, dim=0).unsqueeze(1)
        return out.squeeze(0) if squeeze else out

    if method == "multiscale_bilateral":
        # Multi-scale Pyramid + RGB-Guided Cross-Bilateral Propagation (NYUv2 `fill_depth_cross_bf` equivalent)
        B, _, H, W = depth_clean.shape

        # Step 1: High-precision fine-to-coarse pyramid inpainting with harmonic boundary diffusion
        d_cur = _pyramid_fill_holes_lr(depth_clean, mask_clean)

        # Step 2: Refine filled hole pixels using iterative RGB-guided cross-bilateral propagation
        # so holes adjacent to object boundaries fill strictly from the same-color side of the edge
        if rgb is not None:
            rgb_resized = (
                F.interpolate(rgb, size=(H, W), mode="bilinear", align_corners=False)
                if rgb.shape[-2:] != (H, W)
                else rgb
            )
            pad_rgb = F.pad(rgb_resized, (2, 2, 2, 2), mode="replicate")
            patches_rgb = F.unfold(pad_rgb, kernel_size=5).view(B, 3, 25, H, W)
            color_diff2 = ((patches_rgb - rgb_resized.unsqueeze(2)) ** 2).sum(dim=1)
            w_color = torch.exp(-color_diff2 / (2.0 * (0.075**2)))

            # Anchor valid sensor readings with 3x higher confidence than pyramid-interpolated hole estimates
            conf_map = torch.where(mask_clean > 0.5, torch.ones_like(mask_clean) * 3.0, torch.ones_like(mask_clean) * 0.4)
            pad_m = F.pad(conf_map, (2, 2, 2, 2), mode="replicate")
            patches_m = F.unfold(pad_m, kernel_size=5).view(B, 25, H, W)
            weights = w_color * patches_m
            norm_weights = weights / torch.clamp(weights.sum(dim=1, keepdim=True), min=1e-6)

            for _ in range(3):
                pad_d = F.pad(d_cur, (2, 2, 2, 2), mode="replicate")
                patches_d = F.unfold(pad_d, kernel_size=5).view(B, 25, H, W)
                d_refined = (patches_d * norm_weights).sum(dim=1, keepdim=True)
                # Keep valid measurements anchored; propagate iso-color depth into missing regions
                d_cur = torch.where(mask_clean > 0.5, depth_clean, d_refined)

        return d_cur.squeeze(0) if squeeze else d_cur

    if method == "colorization":
        # Levin et al. Colorization-inspired iterative anisotropic affinity diffusion
        # Solves (I - W) D = D_known on missing pixels using RGB affinity weights
        d_init = fill_depth_holes(depth_clean, rgb=rgb, valid_mask=mask_clean, method="multiscale_bilateral")
        if rgb is None:
            return d_init.squeeze(0) if squeeze else d_init

        B, _, H, W = d_init.shape
        rgb_resized = F.interpolate(rgb, size=(H, W), mode="bilinear", align_corners=False) if rgb.shape[-2:] != (H, W) else rgb

        # Compute 8-connected neighbor RGB affinities
        pad_rgb = F.pad(rgb_resized, (1, 1, 1, 1), mode="replicate")
        patches_rgb = F.unfold(pad_rgb, kernel_size=3).view(B, 3, 9, H, W)
        # Variance in 3x3 window
        mean_rgb = patches_rgb.mean(dim=2, keepdim=True)
        var_rgb = ((patches_rgb - mean_rgb) ** 2).mean(dim=(1, 2), keepdim=True) + 1e-4  # (B, 1, 1, H, W)
        diff2 = ((patches_rgb - rgb_resized.unsqueeze(2)) ** 2).mean(dim=1, keepdim=True)  # (B, 1, 9, H, W)
        affinity = torch.exp(-diff2 / (2.0 * var_rgb)).squeeze(1)  # (B, 9, H, W)
        affinity[:, 4, :, :] = 0.0  # exclude self
        affinity = affinity / torch.clamp(affinity.sum(dim=1, keepdim=True), min=1e-6)

        d_iter = d_init.clone()
        for _ in range(12):
            pad_d = F.pad(d_iter, (1, 1, 1, 1), mode="replicate")
            patches_d = F.unfold(pad_d, kernel_size=3).view(B, 9, H, W)
            d_prop = (patches_d * affinity).sum(dim=1, keepdim=True)
            # Clamp known pixels to anchor measurements, diffuse smoothly along RGB isophotes in holes
            d_iter = torch.where(mask_clean > 0.5, depth_clean, d_prop)

        return d_iter.squeeze(0) if squeeze else d_iter

    raise ValueError(f"Unknown hole filling method: {method}")

