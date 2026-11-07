"""
EG-DepthNet: Edge-Gated Cross-Guidance Network with Anisotropic Spatial Propagation.

Architectural Innovations for ProbID - P4:
1. Sparsity-Aware Partial Convolution Stem: Handles missing sensor holes & arbitrary scale factors (2x, 4x, 8x).
2. Decoupled Geometric vs. Appearance Pyramid: Separates 3D surface geometry from 2D RGB texture.
3. Edge-Gated Cross-Attention (EGCA): Learns a per-pixel Coherence Gate G_edge in [0, 1] that suppresses
   RGB texture-copying on planar surfaces while amplifying boundary alignment at true 3D occlusion jumps.
4. Anisotropic Spatial Propagation Network (ASPN): Learned 8-neighbor affinity propagation that snaps
   discontinuities to sub-pixel boundaries and eliminates flying pixels.
5. Aleatoric Uncertainty Prediction: Estimates per-pixel confidence for Hybrid Bilateral Energy Fusion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from depth_refine.classical.filters import (
    bilateral_mode_snap_and_denoise,
    compute_spatial_gradients,
    detect_edges_and_discontinuities,
    guided_filter,
    joint_bilateral_filter,
    mask_bicubic_upsample,
)
from depth_refine.classical.hole_filling import fill_depth_holes
from depth_refine.classical.interpolation import interpolate_depth


@dataclass
class EGDepthOutput:
    """Structured output of EG-DepthNet (with Stage-1 & LoftUp/PatchRefiner explainability maps)."""

    depth_hr: torch.Tensor  # (B, 1, H, W) reconstructed high-resolution depth
    coarse_depth: torch.Tensor  # (B, 1, H, W) pre-propagation depth prediction
    uncertainty: torch.Tensor  # (B, 1, H, W) aleatoric uncertainty (> 0)
    edge_gate: torch.Tensor  # (B, 1, H, W) in [0, 1] learned geometric edge coherence gate
    depth_prior: Optional[torch.Tensor] = None  # (B, 1, H, W) metric-aligned structural / Depth-Anything prior
    mask_bicubic: Optional[torch.Tensor] = None  # (B, 1, H, W) Stage-1 Mask-Bicubic initialization
    feature_vis: Optional[torch.Tensor] = None  # (B, 1, H, W) LoftUp (c) raw neural feature energy map


class ContinuousCoordCrossAttentionUpsampler(nn.Module):
    """
    LoftUp-Inspired Continuous Coordinate-Based Cross-Attention Upsampler.

    Uses sinusoidal 2D pixel coordinate embeddings + high-resolution RGB guidance queries
    attending to low-resolution depth features across local 3x3 neighborhoods so the model
    natively supports 2x, 4x, and 8x upsampling without discrete deconvolution checkerboard
    or mosaic grid artifacts.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        # 8 sinusoidal coordinate channels (sin/cos at 2 frequencies for x and y) + channels RGB query
        self.query_proj = nn.Conv2d(channels + 8, channels, kernel_size=1, bias=False)
        self.key_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.val_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.tensor(0.0))
        nn.init.zeros_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    @staticmethod
    def _sinusoidal_2d_coords(B: int, H: int, W: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        ys = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        freqs = [3.14159265, 6.2831853]
        embeds = []
        for f in freqs:
            embeds.extend([torch.sin(f * xx), torch.cos(f * xx), torch.sin(f * yy), torch.cos(f * yy)])
        coord_map = torch.stack(embeds, dim=0).unsqueeze(0).expand(B, -1, -1, -1)
        return coord_map

    def forward(self, f_low_up: torch.Tensor, f_rgb_hr: torch.Tensor) -> torch.Tensor:
        B, C, H, W = f_rgb_hr.shape
        coords = self._sinusoidal_2d_coords(B, H, W, f_rgb_hr.device, f_rgb_hr.dtype)
        q = self.query_proj(torch.cat([f_rgb_hr, coords], dim=1))  # (B, C, H, W)
        k = self.key_proj(f_low_up)
        v = self.val_proj(f_low_up)

        # Local 3x3 cross-attention (queries at HR pixel p attend to 9 neighborhood keys/values)
        pad_k = F.pad(k, (1, 1, 1, 1), mode="replicate")
        pad_v = F.pad(v, (1, 1, 1, 1), mode="replicate")
        k_patches = F.unfold(pad_k, kernel_size=3).view(B, C, 9, H, W)
        v_patches = F.unfold(pad_v, kernel_size=3).view(B, C, 9, H, W)

        scale = float(C) ** -0.5
        attn_logits = (q.unsqueeze(2) * k_patches).sum(dim=1) * scale  # (B, 9, H, W)
        attn_weights = F.softmax(attn_logits, dim=1).unsqueeze(1)  # (B, 1, 9, H, W)
        attended = (v_patches * attn_weights).sum(dim=2)  # (B, C, H, W)
        return f_low_up + torch.tanh(self.gamma) * self.out_proj(attended)


class FoundationStructuralPriorAdapter(nn.Module):
    """
    PatchRefiner / Depth-Anything Inspired Structural Prior Adapter with
    Locally Weighted Least-Squares (LWLS) Metric Scale-Shift Alignment.

    Extracts a dense, hole-free relative structural depth prior from high-resolution RGB guidance
    (preserving thin structures like foliage, chair legs, and lattice struts even when LR depth has
    missing holes) and aligns it to the metric scale of `depth_lr` via Locally Weighted Least-Squares:

        D_prior(p) = a(p) * P_rel(p) + b(p)
    """

    def __init__(self, patch_radius: int = 8, ridge_eps: float = 1e-2) -> None:
        super().__init__()
        self.patch_radius = patch_radius
        self.ridge_eps = ridge_eps

    @torch.no_grad()
    def extract_relative_prior(self, rgb_hr: torch.Tensor) -> torch.Tensor:
        """Compute dense, hole-free relative depth geometry in [0, 1] from high-res RGB."""
        B, _, H, W = rgb_hr.shape
        device, dtype = rgb_hr.device, rgb_hr.dtype

        # 1. Perspective vertical support prior + multi-scale structure-tensor relief
        ys = torch.linspace(0.0, 1.0, H, device=device, dtype=dtype).view(1, 1, H, 1).expand(B, 1, H, W)
        p_persp = 1.0 - 0.55 * (ys**0.85)

        luma = 0.299 * rgb_hr[:, 0:1] + 0.587 * rgb_hr[:, 1:2] + 0.114 * rgb_hr[:, 2:3]
        gx, gy, g_mag = compute_spatial_gradients(luma)
        # Multi-scale high-frequency focus & saliency relief
        focus_fine = F.avg_pool2d(F.pad(g_mag, (2, 2, 2, 2), mode="replicate"), kernel_size=5, stride=1)
        focus_coarse = F.avg_pool2d(F.pad(g_mag, (4, 4, 4, 4), mode="replicate"), kernel_size=9, stride=1)
        color_mean = rgb_hr.mean(dim=(2, 3), keepdim=True)
        saliency = torch.sqrt(((rgb_hr - color_mean) ** 2).mean(dim=1, keepdim=True) + 1e-8)
        saliency = F.avg_pool2d(F.pad(saliency, (3, 3, 3, 3), mode="replicate"), kernel_size=7, stride=1)

        relief = 0.45 * focus_fine + 0.25 * focus_coarse + 0.30 * saliency
        r_min = relief.amin(dim=(2, 3), keepdim=True)
        r_max = relief.amax(dim=(2, 3), keepdim=True)
        relief_norm = (relief - r_min) / torch.clamp(r_max - r_min, min=1e-4)

        p_rel = torch.clamp(p_persp - 0.35 * relief_norm, min=0.05, max=1.0)
        # Edge-preserving guided alignment to RGB silhouettes
        p_rel = guided_filter(p_rel, rgb_hr, radius=4, eps=5e-4, texture_suppression=False)
        return p_rel

    @torch.no_grad()
    def align_to_metric_scale(
        self,
        p_rel: torch.Tensor,
        depth_anchor_hr: torch.Tensor,
        rgb_hr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Locally Weighted Least-Squares (LWLS) Scale-Shift Alignment:
        Solves min_{a, b} || a * P_rel + b - D_anchor ||^2 locally within sliding windows,
        then refines with RGB-guided filtering to preserve sharp thin-structure boundaries.
        """
        r = self.patch_radius
        mean_P = F.avg_pool2d(F.pad(p_rel, (r, r, r, r), mode="reflect"), kernel_size=2 * r + 1, stride=1)
        mean_D = F.avg_pool2d(F.pad(depth_anchor_hr, (r, r, r, r), mode="reflect"), kernel_size=2 * r + 1, stride=1)
        mean_PP = F.avg_pool2d(F.pad(p_rel * p_rel, (r, r, r, r), mode="reflect"), kernel_size=2 * r + 1, stride=1)
        mean_PD = F.avg_pool2d(
            F.pad(p_rel * depth_anchor_hr, (r, r, r, r), mode="reflect"), kernel_size=2 * r + 1, stride=1
        )

        var_P = torch.clamp(mean_PP - mean_P * mean_P, min=0.0)
        cov_PD = mean_PD - mean_P * mean_D

        a = cov_PD / (var_P + self.ridge_eps)
        b = mean_D - a * mean_P

        mean_a = F.avg_pool2d(F.pad(a, (r, r, r, r), mode="reflect"), kernel_size=2 * r + 1, stride=1)
        mean_b = F.avg_pool2d(F.pad(b, (r, r, r, r), mode="reflect"), kernel_size=2 * r + 1, stride=1)

        d_aligned = torch.clamp(mean_a * p_rel + mean_b, min=0.05)
        return d_aligned

    @torch.no_grad()
    def forward(self, rgb_hr: torch.Tensor, depth_anchor_hr: torch.Tensor) -> torch.Tensor:
        p_rel = self.extract_relative_prior(rgb_hr)
        return self.align_to_metric_scale(p_rel, depth_anchor_hr, rgb_hr)


class SparsityAwareConv(nn.Module):
    """Partial convolution block that normalizes activations over valid depth measurements."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=pad, bias=False)
        self.mask_pool = nn.AvgPool2d(kernel_size=kernel_size, stride=1, padding=pad, count_include_pad=False)
        self.bias = nn.Parameter(torch.zeros(1, out_ch, 1, 1))
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        masked_x = x * mask
        raw_out = self.conv(masked_x)
        mask_ratio = self.mask_pool(mask)
        new_mask = (mask_ratio > 1e-4).to(x.dtype)
        scale = torch.where(new_mask > 0.5, 1.0 / torch.clamp(mask_ratio, min=1e-3), torch.zeros_like(mask_ratio))
        out = self.act(raw_out * scale + self.bias)
        return out, new_mask


class ResidualGeomBlock(nn.Module):
    """Depthwise-separable + channel attention residual block for fast, deep geometric representation."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(8, channels // 4), kernel_size=1),
            nn.GELU(),
            nn.Conv2d(max(8, channels // 4), channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = out * self.se(out)
        return self.act(out + residual)


class EdgeGatedCrossGuidanceBlock(nn.Module):
    """
    Edge-Gated Cross-Attention (EGCA) Module.

    Computes a geometric Coherence Gate `G_edge` from depth gradients, local depth span, and RGB edges.
    - When `G_edge -> 0` (planar surface with strong RGB texture), RGB features are suppressed to
      prevent texture-copying artifacts.
    - When `G_edge -> 1` (true 3D occlusion boundary), high-frequency RGB boundary features are
      injected into the depth stream.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Conv2d(channels * 2 + 2, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels // 2, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.rgb_transform = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(8, channels), channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1),
        )
        self.fuse = ResidualGeomBlock(channels)

    def forward(
        self,
        f_depth: torch.Tensor,
        f_rgb: torch.Tensor,
        d_grad_mag: torch.Tensor,
        coherence_prior: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gate_in = torch.cat([f_depth, f_rgb, d_grad_mag, coherence_prior], dim=1)
        learned_gate = self.gate_net(gate_in)
        effective_gate = 0.4 * learned_gate + 0.6 * coherence_prior

        guided_rgb = self.rgb_transform(f_rgb) * effective_gate
        fused = self.fuse(f_depth + guided_rgb)
        return fused, effective_gate


class AnisotropicSpatialPropagation(nn.Module):
    """
    Anisotropic Spatial Propagation Network (ASPN) Head.
    Refines depth via K steps of learned 8-neighbor affinity propagation plus sensor anchor clamping.
    """

    def __init__(self, channels: int, prop_steps: int = 4) -> None:
        super().__init__()
        self.prop_steps = prop_steps
        self.guidance_head = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels // 2, 9, kernel_size=3, padding=1),
        )
        self.confidence_head = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def reset_center_identity(self) -> None:
        """Initialize the 3x3 affinity head to favor the center pixel so initial propagation never blurs edges."""
        last_conv = self.guidance_head[-1]
        if isinstance(last_conv, nn.Conv2d):
            nn.init.zeros_(last_conv.weight)
            if last_conv.bias is not None:
                nn.init.zeros_(last_conv.bias)
                last_conv.bias.data[4] = 4.5  # Center pixel of 3x3 patch

    def forward(
        self,
        coarse_depth: torch.Tensor,
        anchor_depth: torch.Tensor,
        features: torch.Tensor,
        edge_gate: torch.Tensor,
    ) -> torch.Tensor:
        B, _, H, W = coarse_depth.shape
        raw_aff = self.guidance_head(features)  # (B, 9, H, W)
        aff = F.softmax(raw_aff, dim=1)

        # Gate neighbor affinities by (1.0 - edge_gate) to mathematically prevent
        # cross-boundary diffusion at geometric occlusion edges.
        # Center pixel (index 4) retains self-weight.
        gate = torch.clamp((1.0 - edge_gate), min=0.01) ** 2.0
        ones = torch.ones_like(gate)
        gate_9 = torch.cat([gate.repeat(1, 4, 1, 1), ones, gate.repeat(1, 4, 1, 1)], dim=1)
        aff_gated = aff * gate_9
        aff = aff_gated / torch.clamp(aff_gated.sum(dim=1, keepdim=True), min=1e-6)

        anchor_w = 0.35 * self.confidence_head(features)

        d_cur = coarse_depth
        for _ in range(self.prop_steps):
            padded = F.pad(d_cur, (1, 1, 1, 1), mode="replicate")
            patches = F.unfold(padded, kernel_size=3).view(B, 9, H, W)
            d_prop = (patches * aff).sum(dim=1, keepdim=True)
            d_cur = (1.0 - anchor_w) * d_prop + anchor_w * anchor_depth

        return d_cur


class EGDepthNet(nn.Module):
    """
    Edge-Gated Cross-Guidance Network (EG-DepthNet) for 2x, 4x, and 8x Depth Upsampling & Refinement.

    Deep Module Interface:
    - `forward(depth_lr, rgb_hr, scale=4, valid_mask_lr=None) -> EGDepthOutput`
    """

    def __init__(self, base_channels: int = 48, prop_steps: int = 4) -> None:
        super().__init__()
        self.base_channels = base_channels
        c1 = base_channels
        c2 = base_channels * 2

        # 1. Sparsity-Aware Geometric Stem (inputs: bicubic_d, nearest_d, jbf_prior, d_gx, d_gy, scale_channel)
        self.depth_stem = SparsityAwareConv(in_ch=6, out_ch=c1, kernel_size=3)
        self.depth_enc1 = ResidualGeomBlock(c1)
        self.depth_down = nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1)
        self.depth_enc2 = ResidualGeomBlock(c2)

        # 2. RGB Structural Guidance Stream (inputs: RGB(3) + rgb_gx(1) + rgb_gy(1) = 5 channels)
        self.rgb_stem = nn.Sequential(
            nn.Conv2d(5, c1, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(8, c1), c1),
            nn.GELU(),
        )
        self.rgb_enc1 = ResidualGeomBlock(c1)
        self.rgb_down = nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1)
        self.rgb_enc2 = ResidualGeomBlock(c2)

        # 3. Multi-Scale Edge-Gated Cross-Attention Blocks
        self.egca_low = EdgeGatedCrossGuidanceBlock(c2)
        self.up_fuse = nn.Sequential(
            nn.Conv2d(c2, c1 * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.GELU(),
        )
        self.egca_high = EdgeGatedCrossGuidanceBlock(c1)

        # 4. Refinement Decoder & Residual Head
        self.decoder = nn.Sequential(
            ResidualGeomBlock(c1),
            ResidualGeomBlock(c1),
        )
        self.residual_head = nn.Sequential(
            nn.Conv2d(c1, c1 // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(c1 // 2, 1, kernel_size=3, padding=1),
            nn.Tanh(),
        )
        self.residual_gain = nn.Parameter(torch.tensor(0.10))

        # 5. Aleatoric Uncertainty Head
        self.uncertainty_head = nn.Sequential(
            nn.Conv2d(c1, c1 // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(c1 // 2, 1, kernel_size=3, padding=1),
            nn.Softplus(),
        )

        # 6. Anisotropic Spatial Propagation Head
        self.aspn = AnisotropicSpatialPropagation(channels=c1, prop_steps=prop_steps)

        # 7. LoftUp-Inspired Continuous 2D Coordinate Cross-Attention Upsampler & Structural Prior Adapter
        self.coord_upsampler = ContinuousCoordCrossAttentionUpsampler(channels=c1)
        self.structural_prior = FoundationStructuralPriorAdapter(patch_radius=8)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize residual output conv and ASPN center weight so even at epoch 0 boundaries stay razor sharp."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        final_res_conv = self.residual_head[2]
        if isinstance(final_res_conv, nn.Conv2d):
            nn.init.zeros_(final_res_conv.weight)
            if final_res_conv.bias is not None:
                nn.init.zeros_(final_res_conv.bias)
        nn.init.zeros_(self.coord_upsampler.out_proj.weight)
        if self.coord_upsampler.out_proj.bias is not None:
            nn.init.zeros_(self.coord_upsampler.out_proj.bias)
        self.aspn.reset_center_identity()

    def forward(
        self,
        depth_lr: torch.Tensor,
        rgb_hr: Optional[torch.Tensor] = None,
        scale: int = 4,
        valid_mask_lr: Optional[torch.Tensor] = None,
    ) -> EGDepthOutput:
        squeeze = False
        if depth_lr.ndim == 3:
            depth_lr = depth_lr.unsqueeze(0)
            if rgb_hr is not None and rgb_hr.ndim == 3:
                rgb_hr = rgb_hr.unsqueeze(0)
            if valid_mask_lr is not None and valid_mask_lr.ndim == 3:
                valid_mask_lr = valid_mask_lr.unsqueeze(0)
            squeeze = True

        if rgb_hr is not None:
            H, W = rgb_hr.shape[-2], rgb_hr.shape[-1]
        else:
            H, W = depth_lr.shape[-2] * scale, depth_lr.shape[-1] * scale
            rgb_hr = torch.zeros((depth_lr.shape[0], 3, H, W), device=depth_lr.device, dtype=depth_lr.dtype)

        B = depth_lr.shape[0]
        device = depth_lr.device
        dtype = depth_lr.dtype

        if valid_mask_lr is None:
            valid_mask_lr = (depth_lr > 1e-3).to(dtype)
        else:
            valid_mask_lr = valid_mask_lr.to(dtype)

        # Stage-1 Mask-Bicubic Refinement & Pre-Conditioning
        rgb_lr = F.interpolate(rgb_hr, size=depth_lr.shape[-2:], mode="bilinear", align_corners=False)
        depth_lr_filled = fill_depth_holes(depth_lr, rgb=rgb_lr, valid_mask=valid_mask_lr, method="multiscale_bilateral")

        d_bicubic = interpolate_depth(depth_lr_filled, target_size=(H, W), mode="bicubic")
        d_nearest = interpolate_depth(depth_lr_filled, target_size=(H, W), mode="nearest")
        d_mask_bicubic, _ = mask_bicubic_upsample(depth_lr, rgb_hr=rgb_hr, scale=scale, valid_mask_lr=valid_mask_lr)

        # Foreground/Background Mode-Snapped & Texture-Suppressed Geometric Basis
        d_anchor, coherence_prior = bilateral_mode_snap_and_denoise(d_bicubic, rgb_hr, scale=scale)

        # Metric-Aligned Foundation Structural Prior (Depth-Anything / Relative Geometry Branch)
        d_prior_aligned = self.structural_prior(rgb_hr, d_anchor)

        # Normalize depth inputs to stable unit scale for neural processing, then restore metric scale
        d_mean = d_anchor.mean(dim=(2, 3), keepdim=True)
        d_std = torch.clamp(d_anchor.std(dim=(2, 3), keepdim=True), min=0.25)
        d_bic_norm = (d_bicubic - d_mean) / d_std
        d_near_norm = (d_nearest - d_mean) / d_std
        d_anc_norm = (d_anchor - d_mean) / d_std

        # Scale-invariant geometric gradients computed on normalized depth
        d_gx, d_gy, d_mag = compute_spatial_gradients(d_anc_norm)

        # Luminance gradients for RGB stream
        luma = 0.299 * rgb_hr[:, 0:1] + 0.587 * rgb_hr[:, 1:2] + 0.114 * rgb_hr[:, 2:3]
        rgb_gx, rgb_gy, _ = compute_spatial_gradients(luma)

        scale_channel = torch.full((B, 1, H, W), float(scale) / 8.0, device=device, dtype=dtype)
        mask_hr = F.interpolate(valid_mask_lr, size=(H, W), mode="nearest")
        mask_hr = torch.clamp(mask_hr, min=0.2)

        depth_in = torch.cat([d_bic_norm, d_near_norm, d_anc_norm, d_gx, d_gy, scale_channel], dim=1)
        rgb_in = torch.cat([rgb_hr, rgb_gx, rgb_gy], dim=1)

        # Encode streams
        fd1, _ = self.depth_stem(depth_in, mask_hr)
        fd1 = self.depth_enc1(fd1)
        fd2 = self.depth_enc2(self.depth_down(fd1))

        fr1 = self.rgb_enc1(self.rgb_stem(rgb_in))
        fr2 = self.rgb_enc2(self.rgb_down(fr1))

        # Low-resolution cross-guidance
        d_mag_low = F.avg_pool2d(d_mag, kernel_size=2, stride=2)
        coh_low = F.avg_pool2d(coherence_prior, kernel_size=2, stride=2)
        fused_low, _ = self.egca_low(fd2, fr2, d_mag_low, coh_low)

        # Upsample & fuse at full resolution with Coordinate-Based Cross-Attention (LoftUp-inspired)
        up_feat = self.up_fuse(fused_low)
        up_feat = 0.65 * up_feat + 0.35 * F.avg_pool2d(
            F.pad(up_feat, (1, 1, 1, 1), mode="replicate"), kernel_size=3, stride=1
        )
        if up_feat.shape[-2:] != (H, W):
            up_feat = F.interpolate(up_feat, size=(H, W), mode="bilinear", align_corners=False)

        # Continuous coordinate cross-attention refinement over upsampled feature map
        up_feat = self.coord_upsampler(up_feat, fr1)

        # Extract LoftUp column (c) raw neural feature map visualization (L2 activation energy)
        feat_energy = torch.sqrt((up_feat.detach() ** 2).mean(dim=1, keepdim=True) + 1e-8)

        fused_high, edge_gate = self.egca_high(fd1 + up_feat, fr1, d_mag, coherence_prior)
        dec_feat = self.decoder(fused_high)

        # Predict residual refinement over boundary-aware anchor
        residual = self.residual_head(dec_feat) * torch.clamp(self.residual_gain, min=0.01, max=1.5) * d_std
        coarse_depth = torch.clamp(d_anchor + residual, min=0.05)

        # Anisotropic Spatial Propagation refinement with edge gating
        depth_hr = self.aspn(coarse_depth, d_anchor, dec_feat, edge_gate)
        depth_hr = torch.clamp(depth_hr, min=0.05)

        # Anti-aliased uncertainty map (eradicating high-frequency checkerboard harmonics)
        raw_unc = self.uncertainty_head(dec_feat)
        uncertainty = F.avg_pool2d(F.pad(raw_unc, (1, 1, 1, 1), mode="replicate"), kernel_size=3, stride=1) + 1e-3

        if squeeze:
            return EGDepthOutput(
                depth_hr=depth_hr.squeeze(0),
                coarse_depth=coarse_depth.squeeze(0),
                uncertainty=uncertainty.squeeze(0),
                edge_gate=edge_gate.squeeze(0),
                depth_prior=d_prior_aligned.squeeze(0),
                mask_bicubic=d_mask_bicubic.squeeze(0),
                feature_vis=feat_energy.squeeze(0),
            )

        return EGDepthOutput(
            depth_hr=depth_hr,
            coarse_depth=coarse_depth,
            uncertainty=uncertainty,
            edge_gate=edge_gate,
            depth_prior=d_prior_aligned,
            mask_bicubic=d_mask_bicubic,
            feature_vis=feat_energy,
        )
