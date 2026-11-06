"""
Edge-Aware Classical Filtering & Discontinuity Detection:
1. Joint Bilateral Filter & Upsampler (JBF / JBU, Kopf et al.) with Trilateral Depth-Coherence Guard
2. Fast Guided Filter (He et al., O(N) exact solver with Edge-Aware Regularization)
3. Edge Detection & Depth Discontinuity Classifier (`detect_edges_and_discontinuities`)
4. Unified Classical Reconstruction Seam (`reconstruct_classical`)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch
import torch.nn.functional as F

from depth_refine.classical.hole_filling import fill_depth_holes
from depth_refine.classical.interpolation import interpolate_depth


@dataclass
class DiscontinuityAnalysis:
    """Outputs of geometric edge and discontinuity analysis."""

    depth_grad_mag: torch.Tensor  # (B, 1, H, W) Sobel gradient magnitude of depth
    rgb_grad_mag: torch.Tensor  # (B, 1, H, W) Sobel gradient magnitude of RGB luminance
    discontinuity_mask: torch.Tensor  # (B, 1, H, W) binary mask of true depth step edges
    coherence_gate: torch.Tensor  # (B, 1, H, W) in [0, 1] - high where RGB edge matches depth jump, low on flat texture


def compute_spatial_gradients(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute Sobel X, Y gradients and magnitude for an image/depth tensor (B, C, H, W)."""
    device = x.device
    dtype = x.dtype
    C = x.shape[1]

    kx = (
        torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=device, dtype=dtype)
        .view(1, 1, 3, 3)
        .repeat(C, 1, 1, 1)
        / 8.0
    )
    ky = (
        torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], device=device, dtype=dtype)
        .view(1, 1, 3, 3)
        .repeat(C, 1, 1, 1)
        / 8.0
    )

    padded = F.pad(x, (1, 1, 1, 1), mode="replicate")
    gx = F.conv2d(padded, kx, groups=C)
    gy = F.conv2d(padded, ky, groups=C)
    mag = torch.sqrt((gx * gx + gy * gy).mean(dim=1, keepdim=True) + 1e-12)
    return gx, gy, mag


def detect_edges_and_discontinuities(
    depth: torch.Tensor,
    rgb: Optional[torch.Tensor] = None,
    rel_jump_thresh: float = 0.025,
) -> DiscontinuityAnalysis:
    """
    Analyze depth discontinuities vs. RGB appearance edges.
    Separates true 3D occlusion boundaries from 2D surface texture edges (preventing texture copying).
    """
    squeeze = False
    if depth.ndim == 3:
        depth = depth.unsqueeze(0)
        if rgb is not None and rgb.ndim == 3:
            rgb = rgb.unsqueeze(0)
        squeeze = True

    _, _, d_mag = compute_spatial_gradients(depth)
    rel_d_grad = d_mag / torch.clamp(depth, min=0.25)
    disc_mask = (rel_d_grad > rel_jump_thresh).to(depth.dtype)

    if rgb is not None:
        _, _, rgb_mag = compute_spatial_gradients(rgb)
        # Local depth range in a 9x9 neighborhood tells us if we are near a true 3D boundary
        d_max = F.max_pool2d(depth, kernel_size=9, stride=1, padding=4)
        d_min = -F.max_pool2d(-depth, kernel_size=9, stride=1, padding=4)
        local_span = (d_max - d_min) / torch.clamp(depth, min=0.25)

        # Coherence gate is near 1.0 when local depth span indicates a 3D jump, and near 0.0 on flat textured walls
        coherence = torch.sigmoid((local_span - 0.03) * 60.0)
    else:
        rgb_mag = torch.zeros_like(d_mag)
        coherence = torch.ones_like(d_mag)

    if squeeze:
        return DiscontinuityAnalysis(
            depth_grad_mag=d_mag.squeeze(0),
            rgb_grad_mag=rgb_mag.squeeze(0),
            discontinuity_mask=disc_mask.squeeze(0),
            coherence_gate=coherence.squeeze(0),
        )
    return DiscontinuityAnalysis(
        depth_grad_mag=d_mag,
        rgb_grad_mag=rgb_mag,
        discontinuity_mask=disc_mask,
        coherence_gate=coherence,
    )


def joint_bilateral_filter(
    depth: torch.Tensor,
    rgb: torch.Tensor,
    kernel_size: int = 9,
    sigma_spatial: float = 3.5,
    sigma_color: float = 0.04,
    sigma_depth: float = 0.25,
    iterations: int = 2,
) -> torch.Tensor:
    """
    Edge-Preserving Joint Bilateral Filter / Upsampler (JBF / JBU) with Depth-Coherence Guard.

    At true 3D depth discontinuities (where `coherence_gate` is high), uses purely RGB photometric
    affinity + spatial Gaussian weights (classic Joint Bilateral Upsampling, Kopf et al.) so blurry
    bicubic ramp pixels across an edge are replaced by the crisp foreground or background depth whose
    RGB color matches the target pixel!
    On flat surfaces (where `coherence_gate` is low), widens sigma_color and uses depth-range weighting
    so RGB wallpaper textures never bleed into flat walls.
    """
    squeeze = False
    if depth.ndim == 3:
        depth = depth.unsqueeze(0)
        rgb = rgb.unsqueeze(0)
        squeeze = True

    B, _, H, W = depth.shape
    device = depth.device
    dtype = depth.dtype
    radius = kernel_size // 2

    # Precompute spatial Gaussian weights (1, K*K, 1, 1)
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    spatial_dist2 = (xx * xx + yy * yy).reshape(1, -1, 1, 1)
    w_spatial = torch.exp(-spatial_dist2 / (2.0 * sigma_spatial * sigma_spatial))

    # Precompute RGB patches and color weights
    pad_rgb = F.pad(rgb, (radius, radius, radius, radius), mode="replicate")
    patches_rgb = F.unfold(pad_rgb, kernel_size=kernel_size).view(B, 3, kernel_size * kernel_size, H, W)
    color_dist2 = ((patches_rgb - rgb.unsqueeze(2)) ** 2).mean(dim=1)  # (B, K*K, H, W)

    # Detect where true 3D depth jumps exist vs flat textured surfaces
    disc_info = detect_edges_and_discontinuities(depth, rgb)
    gate_2d = disc_info.coherence_gate  # (B, 1, H, W)

    # Tight sigma_color at true 3D boundaries to snap edges sharply to RGB;
    # Wide sigma_color on flat surfaces so RGB textures do not create false depth ripples
    eff_sigma_c = sigma_color * gate_2d + 0.45 * (1.0 - gate_2d)
    w_color = torch.exp(-color_dist2 / (2.0 * (eff_sigma_c * eff_sigma_c) + 1e-8))

    d_cur = depth.clone()
    for _ in range(iterations):
        pad_d = F.pad(d_cur, (radius, radius, radius, radius), mode="replicate")
        patches_d = F.unfold(pad_d, kernel_size=kernel_size).view(B, kernel_size * kernel_size, H, W)

        # Notice: at blurred bicubic ramp pixels (where gate_2d is near 1), `d_cur` itself is an invalid
        # mid-air average ("flying pixel"), so we must NOT lock to `d_cur` via w_depth at boundaries!
        # Instead, w_depth is only enforced on flat interior surfaces (1 - gate_2d).
        depth_dist2 = (patches_d - d_cur) ** 2
        w_depth_planar = torch.exp(-depth_dist2 / (2.0 * sigma_depth * sigma_depth))
        w_depth = gate_2d + (1.0 - gate_2d) * w_depth_planar

        weights = w_spatial * w_color * w_depth
        d_cur = (patches_d * weights).sum(dim=1, keepdim=True) / torch.clamp(
            weights.sum(dim=1, keepdim=True), min=1e-8
        )

    return d_cur.squeeze(0) if squeeze else d_cur


def _box_filter(x: torch.Tensor, radius: int) -> torch.Tensor:
    """Fast O(N) exact mean filter using reflection padding and avg_pool2d."""
    k = 2 * radius + 1
    padded = F.pad(x, (radius, radius, radius, radius), mode="reflect")
    return F.avg_pool2d(padded, kernel_size=k, stride=1)


def guided_filter(
    depth: torch.Tensor,
    rgb: torch.Tensor,
    radius: int = 5,
    eps: float = 2e-4,
    texture_suppression: bool = True,
) -> torch.Tensor:
    """
    Fast Multi-Channel Guided Filter (He et al., TPAMI 2013) with Texture-Copy Suppression.
    """
    squeeze = False
    if depth.ndim == 3:
        depth = depth.unsqueeze(0)
        rgb = rgb.unsqueeze(0)
        squeeze = True

    if texture_suppression:
        disc_info = detect_edges_and_discontinuities(depth, rgb)
        gate = disc_info.coherence_gate
        adaptive_eps = eps * gate + 0.15 * (1.0 - gate)
    else:
        adaptive_eps = eps

    # Run per-channel guided filter across R, G, B and average linear models (handles isoluminant color edges!)
    q_sum = torch.zeros_like(depth)
    for c in range(rgb.shape[1]):
        guide = rgb[:, c : c + 1]
        mean_I = _box_filter(guide, radius)
        mean_p = _box_filter(depth, radius)
        mean_Ip = _box_filter(guide * depth, radius)
        cov_Ip = mean_Ip - mean_I * mean_p

        mean_II = _box_filter(guide * guide, radius)
        var_I = torch.clamp(mean_II - mean_I * mean_I, min=0.0)

        a = cov_Ip / (var_I + adaptive_eps)
        b = mean_p - a * mean_I

        mean_a = _box_filter(a, radius)
        mean_b = _box_filter(b, radius)
        q_sum = q_sum + (mean_a * guide + mean_b)

    q = q_sum / float(rgb.shape[1])
    return q.squeeze(0) if squeeze else q


def reconstruct_classical(
    depth_lr: torch.Tensor,
    rgb_hr: Optional[torch.Tensor] = None,
    scale: int = 4,
    method: Literal[
        "nearest",
        "bilinear",
        "bicubic",
        "jbf",
        "guided",
        "boundary_aware_classical",
    ] = "jbf",
    valid_mask_lr: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Deep Module Interface for all Classical Computer Vision depth reconstruction methods.
    """
    squeeze = False
    if depth_lr.ndim == 3:
        depth_lr = depth_lr.unsqueeze(0)
        if rgb_hr is not None and rgb_hr.ndim == 3:
            rgb_hr = rgb_hr.unsqueeze(0)
        if valid_mask_lr is not None and valid_mask_lr.ndim == 3:
            valid_mask_lr = valid_mask_lr.unsqueeze(0)
        squeeze = True

    if rgb_hr is not None:
        target_size = (rgb_hr.shape[-2], rgb_hr.shape[-1])
        rgb_lr = F.interpolate(rgb_hr, size=depth_lr.shape[-2:], mode="bilinear", align_corners=False)
    else:
        target_size = (depth_lr.shape[-2] * scale, depth_lr.shape[-1] * scale)
        rgb_lr = None

    # Step 1: Remove outlier spikes and fill holes in LR sensor space
    depth_lr_filled = fill_depth_holes(
        depth_lr, rgb=rgb_lr, valid_mask=valid_mask_lr, method="multiscale_bilateral"
    )

    if method in ("nearest", "bilinear", "bicubic"):
        out = interpolate_depth(depth_lr_filled, target_size=target_size, mode=method)
        return out.squeeze(0) if squeeze else out

    if rgb_hr is None:
        out = interpolate_depth(depth_lr_filled, target_size=target_size, mode="bicubic")
        return out.squeeze(0) if squeeze else out

    # Step 2: Initial upsampling
    d_bicubic = interpolate_depth(depth_lr_filled, target_size=target_size, mode="bicubic")

    if method == "guided":
        out = guided_filter(d_bicubic, rgb_hr, radius=max(4, scale + 1), eps=2e-4, texture_suppression=True)
        return out.squeeze(0) if squeeze else out

    k_size = min(11, 2 * scale + 3)
    if k_size % 2 == 0:
        k_size += 1

    if method == "jbf":
        out = joint_bilateral_filter(
            d_bicubic,
            rgb_hr,
            kernel_size=k_size,
            sigma_spatial=float(scale) * 0.9,
            sigma_color=0.038,
            sigma_depth=0.20,
            iterations=2,
        )
        return out.squeeze(0) if squeeze else out

    if method == "boundary_aware_classical":
        d_jbf = joint_bilateral_filter(
            d_bicubic,
            rgb_hr,
            kernel_size=k_size,
            sigma_spatial=float(scale) * 0.95,
            sigma_color=0.035,
            sigma_depth=0.18,
            iterations=2,
        )
        out = guided_filter(d_jbf, rgb_hr, radius=3, eps=1e-4, texture_suppression=True)
        return out.squeeze(0) if squeeze else out

    raise ValueError(f"Unknown classical method: {method}")
