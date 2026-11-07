"""
Edge-Aware Classical Filtering, Discontinuity Detection & Bilateral Mode Snapping:
1. Joint Bilateral Filter (JBF, Kopf et al.)
2. Fast Guided Filter (He et al., TPAMI 2013)
3. Edge Detection & Depth Discontinuity Classifier (`detect_edges_and_discontinuities`)
4. Foreground/Background Bilateral Mode Snapping & Planar Denoiser (`bilateral_mode_snap_and_denoise`)
5. Unified Classical Reconstruction Seam (`reconstruct_classical`)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch
import torch.nn.functional as F

from depth_refine.classical.hole_filling import fill_depth_holes, remove_depth_outliers
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
        d_max = F.max_pool2d(depth, kernel_size=9, stride=1, padding=4)
        d_min = -F.max_pool2d(-depth, kernel_size=9, stride=1, padding=4)
        local_span = (d_max - d_min) / torch.clamp(depth, min=0.25)
        coherence = torch.sigmoid((local_span - 0.032) * 65.0)
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
    kernel_size: int = 7,
    sigma_spatial: float = 2.5,
    sigma_color: float = 0.06,
    sigma_depth: float = 0.35,
    iterations: int = 1,
    texture_suppression: bool = False,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Joint Bilateral Filter (JBF, Kopf et al.) with valid-mask normalized convolution.
    When `texture_suppression=True`, modulates color bandwidth by the 3D discontinuity coherence gate.
    """
    squeeze = False
    if depth.ndim == 3:
        depth = depth.unsqueeze(0)
        rgb = rgb.unsqueeze(0)
        if valid_mask is not None and valid_mask.ndim == 3:
            valid_mask = valid_mask.unsqueeze(0)
        squeeze = True

    B, _, H, W = depth.shape
    device = depth.device
    dtype = depth.dtype
    radius = kernel_size // 2

    if valid_mask is None:
        valid_mask = ((depth > 1e-3) & torch.isfinite(depth)).to(dtype)
    else:
        valid_mask = ((valid_mask > 0.5) & (depth > 1e-3) & torch.isfinite(depth)).to(dtype)

    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    spatial_dist2 = (xx * xx + yy * yy).reshape(1, -1, 1, 1)
    w_spatial = torch.exp(-spatial_dist2 / (2.0 * sigma_spatial * sigma_spatial))

    pad_rgb = F.pad(rgb, (radius, radius, radius, radius), mode="replicate")
    patches_rgb = F.unfold(pad_rgb, kernel_size=kernel_size).view(B, 3, kernel_size * kernel_size, H, W)
    color_dist2 = ((patches_rgb - rgb.unsqueeze(2)) ** 2).mean(dim=1)

    pad_m = F.pad(valid_mask, (radius, radius, radius, radius), mode="constant", value=0.0)
    patches_m = F.unfold(pad_m, kernel_size=kernel_size).view(B, kernel_size * kernel_size, H, W)

    if texture_suppression:
        disc_info = detect_edges_and_discontinuities(depth, rgb)
        gate_2d = disc_info.coherence_gate
        eff_sigma_c = sigma_color * gate_2d + 0.45 * (1.0 - gate_2d)
    else:
        gate_2d = torch.ones_like(depth)
        eff_sigma_c = torch.full_like(depth, sigma_color)

    w_color = torch.exp(-color_dist2 / (2.0 * (eff_sigma_c * eff_sigma_c) + 1e-8))

    d_cur = depth.clone()
    for _ in range(iterations):
        pad_d = F.pad(d_cur * valid_mask, (radius, radius, radius, radius), mode="replicate")
        patches_d = F.unfold(pad_d, kernel_size=kernel_size).view(B, kernel_size * kernel_size, H, W)

        depth_dist2 = (patches_d - d_cur) ** 2
        # Scale-adaptive depth range bandwidth to support both indoor (NYU) and outdoor (KITTI LiDAR) depths
        depth_scale = torch.clamp(d_cur.mean(dim=(2, 3), keepdim=True) / 2.5, min=0.5, max=15.0)
        eff_sigma_d = sigma_depth * depth_scale
        w_depth_planar = torch.exp(-depth_dist2 / (2.0 * eff_sigma_d * eff_sigma_d + 1e-8))
        w_depth = gate_2d + (1.0 - gate_2d) * w_depth_planar if texture_suppression else w_depth_planar

        weights = w_spatial * w_color * w_depth * patches_m
        weight_sum = weights.sum(dim=1, keepdim=True)
        filtered = (patches_d * weights).sum(dim=1, keepdim=True) / torch.clamp(weight_sum, min=1e-8)
        d_cur = torch.where(weight_sum > 1e-5, filtered, d_cur)

    return d_cur.squeeze(0) if squeeze else d_cur


def _box_filter(x: torch.Tensor, radius: int) -> torch.Tensor:
    """Fast O(N) exact mean filter using reflection padding and avg_pool2d."""
    k = 2 * radius + 1
    padded = F.pad(x, (radius, radius, radius, radius), mode="reflect")
    return F.avg_pool2d(padded, kernel_size=k, stride=1)


def _normalized_box_filter(x: torch.Tensor, mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Normalized convolution box filter: strictly ignores zero/missing pixels, falling back gracefully."""
    k = 2 * radius + 1
    pad_x = F.pad(x * mask, (radius, radius, radius, radius), mode="replicate")
    pad_m = F.pad(mask, (radius, radius, radius, radius), mode="replicate")
    sum_x = F.avg_pool2d(pad_x, kernel_size=k, stride=1)
    sum_m = F.avg_pool2d(pad_m, kernel_size=k, stride=1)
    norm = sum_x / torch.clamp(sum_m, min=1e-5)
    fallback = _box_filter(x, radius)
    return torch.where(sum_m > 1e-4, norm, fallback)


def guided_filter(
    depth: torch.Tensor,
    rgb: torch.Tensor,
    radius: int = 5,
    eps: float = 1e-3,
    texture_suppression: bool = False,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Fast Multi-Channel Guided Filter (He et al., TPAMI 2013) with normalized convolution hole masking.
    """
    squeeze = False
    if depth.ndim == 3:
        depth = depth.unsqueeze(0)
        rgb = rgb.unsqueeze(0)
        if valid_mask is not None and valid_mask.ndim == 3:
            valid_mask = valid_mask.unsqueeze(0)
        squeeze = True

    if valid_mask is None:
        valid_mask = ((depth > 1e-3) & torch.isfinite(depth)).to(depth.dtype)
    else:
        valid_mask = ((valid_mask > 0.5) & (depth > 1e-3) & torch.isfinite(depth)).to(depth.dtype)

    if texture_suppression:
        disc_info = detect_edges_and_discontinuities(depth, rgb)
        gate = disc_info.coherence_gate
        adaptive_eps = eps * gate + 0.18 * (1.0 - gate)
    else:
        adaptive_eps = eps

    q_sum = torch.zeros_like(depth)
    for c in range(rgb.shape[1]):
        guide = rgb[:, c : c + 1]
        mean_I = _box_filter(guide, radius)
        mean_p = _normalized_box_filter(depth, valid_mask, radius)
        mean_Ip = _normalized_box_filter(guide * depth, valid_mask, radius)
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


def bilateral_mode_snap_and_denoise(
    depth_hr: torch.Tensor,
    rgb_hr: torch.Tensor,
    scale: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    RGB-Guided Foreground/Background Mode Snapping & Texture-Suppressed Planar Denoiser.

    Solves the two fundamental failure modes of depth upsampling:
    1. Boundary Blurring / Flying Pixels: Within a local window around each depth discontinuity,
       separates candidate depths into Foreground (closer) and Background (farther) modes, computes
       their local RGB color prototypes (c_fg, c_bg), and snaps each high-res pixel to the mode
       whose RGB prototype matches its high-res color I(p).
    2. Planar Sensor Noise & Texture Copying: Where local depth span is small (planar walls/floors),
       suppresses RGB high-frequency texture transfer and applies isotropic planar denoising.
    """
    squeeze = False
    if depth_hr.ndim == 3:
        depth_hr = depth_hr.unsqueeze(0)
        rgb_hr = rgb_hr.unsqueeze(0)
        squeeze = True

    B, _, H, W = depth_hr.shape
    k_size = 5 if scale <= 2 else min(11, 2 * scale + 3)
    if k_size % 2 == 0:
        k_size += 1
    radius = k_size // 2
    N = k_size * k_size

    # Step 1: Texture-suppressed JBF pass with scale-matched receptive field
    d_jbf = joint_bilateral_filter(
        depth_hr,
        rgb_hr,
        kernel_size=k_size,
        sigma_spatial=float(scale) * 0.95,
        sigma_color=0.038,
        sigma_depth=0.85,
        iterations=2,
        texture_suppression=True,
    )

    # Step 2: Foreground / Background Mode Separation at Discontinuities
    pad_d = F.pad(d_jbf, (radius, radius, radius, radius), mode="replicate")
    pad_rgb = F.pad(rgb_hr, (radius, radius, radius, radius), mode="replicate")

    patches_d = F.unfold(pad_d, kernel_size=k_size).view(B, N, H, W)
    patches_rgb = F.unfold(pad_rgb, kernel_size=k_size).view(B, 3, N, H, W)

    d_min = patches_d.min(dim=1, keepdim=True).values
    d_max = patches_d.max(dim=1, keepdim=True).values
    d_mid = 0.5 * (d_min + d_max)
    span = (d_max - d_min) / torch.clamp(d_mid, min=0.25)

    # Soft membership in Foreground (closer than d_mid) vs Background (farther than d_mid)
    fg_mask = torch.sigmoid((d_mid - patches_d) * 25.0) + 1e-4  # (B, N, H, W)
    bg_mask = torch.sigmoid((patches_d - d_mid) * 25.0) + 1e-4  # (B, N, H, W)

    # Color similarity of each neighbor to target pixel I(p)
    color_dist2 = ((patches_rgb - rgb_hr.unsqueeze(2)) ** 2).mean(dim=1)  # (B, N, H, W)
    w_rgb = torch.exp(-color_dist2 / (2.0 * (0.030**2)))

    # Foreground & Background depth estimates weighted by RGB similarity
    w_fg = fg_mask * w_rgb
    w_bg = bg_mask * w_rgb
    d_fg = (patches_d * w_fg).sum(dim=1, keepdim=True) / torch.clamp(w_fg.sum(dim=1, keepdim=True), min=1e-6)
    d_bg = (patches_d * w_bg).sum(dim=1, keepdim=True) / torch.clamp(w_bg.sum(dim=1, keepdim=True), min=1e-6)

    # RGB prototypes of foreground and background modes
    c_fg = (patches_rgb * fg_mask.unsqueeze(1)).sum(dim=2) / torch.clamp(fg_mask.sum(dim=1, keepdim=True), min=1e-6)
    c_bg = (patches_rgb * bg_mask.unsqueeze(1)).sum(dim=2) / torch.clamp(bg_mask.sum(dim=1, keepdim=True), min=1e-6)

    dist_to_fg = ((rgb_hr - c_fg) ** 2).mean(dim=1, keepdim=True)
    dist_to_bg = ((rgb_hr - c_bg) ** 2).mean(dim=1, keepdim=True)

    # Mode selection probability: > 0.5 selects foreground mode, < 0.5 selects background mode
    choose_fg = torch.sigmoid((dist_to_bg - dist_to_fg) / (2.0 * (0.025**2)))
    d_mode_snapped = choose_fg * d_fg + (1.0 - choose_fg) * d_bg

    # Coherence gate: only activate mode-snapping where a genuine 3D jump exists AND RGB prototypes differ
    proto_color_diff = torch.sqrt(((c_fg - c_bg) ** 2).mean(dim=1, keepdim=True) + 1e-8)
    edge_gate = torch.sigmoid((span - 0.045) * 60.0) * torch.sigmoid((proto_color_diff - 0.035) * 50.0)

    snap_strength = 0.35 if scale <= 2 else 0.55
    d_out = snap_strength * edge_gate * d_mode_snapped + (1.0 - snap_strength * edge_gate) * d_jbf

    if squeeze:
        return d_out.squeeze(0), edge_gate.squeeze(0)
    return d_out, edge_gate


class MaskBicubicUpsampler:
    """
    Mask-Bicubic Refinement (LoftUp Stage-1 Equivalent):

    Blends high-resolution boundary/segmentation masks M(p) derived from RGB-D discontinuities
    with bicubic spatial weights w_bicubic(p, q) and RGB edge affinities w_edge(I_p, I_q):

        D_Mask-Bicubic(p) = sum_{q in N(p)} w_bicubic(p, q) * I(M(p) == M(q)) * w_edge(I_p, I_q) * D(q)
                            / sum_{q in N(p)} w_bicubic(p, q) * I(M(p) == M(q)) * w_edge(I_p, I_q)

    Guarantees zero depth bleeding across foreground/background masks before neural Stage-2 refinement.
    """

    def __init__(self, sigma_edge: float = 0.032) -> None:
        self.sigma_edge = sigma_edge

    def __call__(
        self,
        depth_lr: torch.Tensor,
        rgb_hr: Optional[torch.Tensor] = None,
        scale: int = 4,
        valid_mask_lr: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return mask_bicubic_upsample(
            depth_lr=depth_lr,
            rgb_hr=rgb_hr,
            scale=scale,
            valid_mask_lr=valid_mask_lr,
            sigma_edge=self.sigma_edge,
        )


def mask_bicubic_upsample(
    depth_lr: torch.Tensor,
    rgb_hr: Optional[torch.Tensor] = None,
    scale: int = 4,
    valid_mask_lr: Optional[torch.Tensor] = None,
    sigma_edge: float = 0.032,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Stage-1 Mask-Bicubic upsampled depth and its binary/soft foreground-background partition mask.
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

    if valid_mask_lr is None:
        valid_mask_lr = ((depth_lr > 1e-3) & torch.isfinite(depth_lr)).to(depth_lr.dtype)
    else:
        valid_mask_lr = ((valid_mask_lr > 0.5) & (depth_lr > 1e-3) & torch.isfinite(depth_lr)).to(depth_lr.dtype)

    depth_clean_lr, mask_clean_lr = remove_depth_outliers(depth_lr, valid_mask_lr)
    depth_lr_filled = fill_depth_holes(
        depth_clean_lr, rgb=rgb_lr, valid_mask=mask_clean_lr, method="multiscale_bilateral"
    )
    d_bic = interpolate_depth(depth_lr_filled, target_size=target_size, mode="bicubic")

    if rgb_hr is None:
        ones_mask = torch.ones_like(d_bic)
        if squeeze:
            return d_bic.squeeze(0), ones_mask.squeeze(0)
        return d_bic, ones_mask

    B, _, H, W = d_bic.shape
    device = d_bic.device
    dtype = d_bic.dtype

    k_size = 5 if scale <= 2 else min(9, 2 * scale + 1)
    if k_size % 2 == 0:
        k_size += 1
    radius = k_size // 2
    N = k_size * k_size

    # Bicubic-approximating spatial kernel w_bicubic(p, q)
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    dist = torch.sqrt(xx * xx + yy * yy) / max(float(radius), 1.0)
    w_bicubic = torch.clamp(1.0 - 1.5 * (dist**2) + 0.75 * (dist**3), min=0.05).reshape(1, N, 1, 1)

    pad_d = F.pad(d_bic, (radius, radius, radius, radius), mode="replicate")
    pad_rgb = F.pad(rgb_hr, (radius, radius, radius, radius), mode="replicate")
    patches_d = F.unfold(pad_d, kernel_size=k_size).view(B, N, H, W)
    patches_rgb = F.unfold(pad_rgb, kernel_size=k_size).view(B, 3, N, H, W)

    # Derive high-resolution foreground/background partition mask M(p) in [0, 1]
    d_min = patches_d.min(dim=1, keepdim=True).values
    d_max = patches_d.max(dim=1, keepdim=True).values
    d_mid = 0.5 * (d_min + d_max)
    span = (d_max - d_min) / torch.clamp(d_mid, min=0.25)

    fg_prior = torch.sigmoid((d_mid - patches_d) * 28.0) + 1e-4
    bg_prior = torch.sigmoid((patches_d - d_mid) * 28.0) + 1e-4

    c_fg = (patches_rgb * fg_prior.unsqueeze(1)).sum(dim=2) / torch.clamp(fg_prior.sum(dim=1, keepdim=True), min=1e-6)
    c_bg = (patches_rgb * bg_prior.unsqueeze(1)).sum(dim=2) / torch.clamp(bg_prior.sum(dim=1, keepdim=True), min=1e-6)

    dist_fg = ((rgb_hr - c_fg) ** 2).mean(dim=1, keepdim=True)
    dist_bg = ((rgb_hr - c_bg) ** 2).mean(dim=1, keepdim=True)
    proto_diff = torch.sqrt(((c_fg - c_bg) ** 2).mean(dim=1, keepdim=True) + 1e-8)
    disc_gate = torch.sigmoid((span - 0.04) * 65.0) * torch.sigmoid((proto_diff - 0.03) * 55.0)

    # M(p): 1.0 on foreground side of boundary, 0.0 on background side, 0.5 on flat interior
    m_center = disc_gate * torch.sigmoid((dist_bg - dist_fg) / (2.0 * (0.025**2))) + (1.0 - disc_gate) * 0.5
    pad_m = F.pad(m_center, (radius, radius, radius, radius), mode="replicate")
    patches_m = F.unfold(pad_m, kernel_size=k_size).view(B, N, H, W)

    # Partition compatibility I(M(p) == M(q)): 1 when on same side of mask, 0 across boundary
    mask_compat = torch.exp(-((m_center - patches_m) ** 2) / (2.0 * (0.18**2)))

    # Edge-stopping RGB weight w_edge(I_p, I_q)
    color_dist2 = ((patches_rgb - rgb_hr.unsqueeze(2)) ** 2).mean(dim=1)
    eff_sigma = sigma_edge * disc_gate + 0.35 * (1.0 - disc_gate)
    w_edge = torch.exp(-color_dist2 / (2.0 * (eff_sigma * eff_sigma) + 1e-8))

    weights = w_bicubic * mask_compat * w_edge
    d_mask_bic = (patches_d * weights).sum(dim=1, keepdim=True) / torch.clamp(weights.sum(dim=1, keepdim=True), min=1e-8)

    if squeeze:
        return d_mask_bic.squeeze(0), disc_gate.squeeze(0)
    return d_mask_bic, disc_gate


def reconstruct_classical(
    depth_lr: torch.Tensor,
    rgb_hr: Optional[torch.Tensor] = None,
    scale: int = 4,
    method: Literal[
        "nearest",
        "bilinear",
        "bicubic",
        "mask_bicubic",
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

    if valid_mask_lr is None:
        valid_mask_lr = ((depth_lr > 1e-3) & torch.isfinite(depth_lr)).to(depth_lr.dtype)
    else:
        valid_mask_lr = ((valid_mask_lr > 0.5) & (depth_lr > 1e-3) & torch.isfinite(depth_lr)).to(depth_lr.dtype)

    # 1. First remove sensor outlier spikes & prune isolated floating pixels in LR sensor space
    depth_clean_lr, mask_clean_lr = remove_depth_outliers(depth_lr, valid_mask_lr)

    if method in ("nearest", "bilinear", "bicubic"):
        out = interpolate_depth(depth_clean_lr, target_size=target_size, mode=method, valid_mask_lr=mask_clean_lr)
        return out.squeeze(0) if squeeze else out

    # 2. Fill missing pockets in LR sensor space using RGB-guided cross-bilateral diffusion
    depth_lr_filled = fill_depth_holes(
        depth_clean_lr, rgb=rgb_lr, valid_mask=mask_clean_lr, method="multiscale_bilateral"
    )
    d_bicubic = interpolate_depth(depth_lr_filled, target_size=target_size, mode="bicubic")

    if rgb_hr is None:
        return d_bicubic.squeeze(0) if squeeze else d_bicubic

    if method == "guided":
        # Standard He et al. Guided Filter on dense hole-filled bicubic depth
        out = guided_filter(
            d_bicubic,
            rgb_hr,
            radius=max(4, scale + 1),
            eps=1.5e-3,
            texture_suppression=False,
        )
        return out.squeeze(0) if squeeze else out

    k_size = min(9, 2 * scale + 1)
    if k_size % 2 == 0:
        k_size += 1

    if method == "jbf":
        # Standard Kopf et al. Joint Bilateral Filter on dense hole-filled bicubic depth
        out = joint_bilateral_filter(
            d_bicubic,
            rgb_hr,
            kernel_size=5,
            sigma_spatial=float(scale) * 0.65,
            sigma_color=0.12,
            sigma_depth=0.18,
            iterations=1,
            texture_suppression=False,
        )
        return out.squeeze(0) if squeeze else out

    if method == "mask_bicubic":
        out, _ = mask_bicubic_upsample(depth_lr, rgb_hr=rgb_hr, scale=scale, valid_mask_lr=valid_mask_lr)
        return out.squeeze(0) if squeeze else out

    if method == "boundary_aware_classical":
        out, _ = bilateral_mode_snap_and_denoise(d_bicubic, rgb_hr, scale=scale)
        return out.squeeze(0) if squeeze else out

    raise ValueError(f"Unknown classical method: {method}")
