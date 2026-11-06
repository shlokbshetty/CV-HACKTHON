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
    Detect and suppress isolated salt-and-pepper sensor outlier spikes while preserving
    legitimate step edges (which have consistent support on at least 2-3 neighbors).

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
        valid_mask = (depth > 1e-3).to(depth.dtype)
    else:
        valid_mask = valid_mask.to(depth.dtype)

    # Extract 3x3 patches to check how many neighbors agree with the center pixel
    B, C, H, W = depth.shape
    padded_d = F.pad(depth * valid_mask, (1, 1, 1, 1), mode="replicate")
    padded_m = F.pad(valid_mask, (1, 1, 1, 1), mode="constant", value=0.0)

    patches_d = F.unfold(padded_d, kernel_size=3).view(B, 9, H, W)
    patches_m = F.unfold(padded_m, kernel_size=3).view(B, 9, H, W)

    # Compute valid neighborhood median approximation and agreement count
    diff = torch.abs(patches_d - depth) * patches_m
    close_neighbors = ((diff < jump_threshold) & (patches_m > 0.5)).sum(dim=1, keepdim=True)
    total_valid_neighbors = (patches_m > 0.5).sum(dim=1, keepdim=True)

    # Adaptive neighbor quorum: flags isolated spikes even in sparse / hole-adjacent regions
    min_required = torch.clamp((total_valid_neighbors * 0.4).long(), min=2, max=4)
    is_spike = (valid_mask > 0.5) & (total_valid_neighbors >= 3) & (close_neighbors < min_required)

    # Replace spike with mean of valid non-center neighbors
    neighbor_mask = patches_m.clone()
    neighbor_mask[:, 4, :, :] = 0.0
    neighbor_sum = (patches_d * neighbor_mask).sum(dim=1, keepdim=True)
    neighbor_cnt = neighbor_mask.sum(dim=1, keepdim=True)

    replacement = neighbor_sum / torch.clamp(neighbor_cnt, min=1.0)
    cleaned_depth = torch.where(is_spike, replacement, depth)
    updated_mask = valid_mask.clone()

    if squeeze:
        return cleaned_depth.squeeze(0), updated_mask.squeeze(0)
    return cleaned_depth, updated_mask


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

