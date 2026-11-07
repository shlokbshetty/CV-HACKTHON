"""
Hybrid Depth Reconstruction Pipeline (ProbID - P4 Mandatory Requirement C).

Combines Classical Interpolation & Edge-Aware Filtering with Deep Learning (`EGDepthNet`)
via a 3-Stage Uncertainty-Guided Bilateral Energy & Sensor-Consistency Refinement Pipeline:
1. Stage 1 (Classical Pre-Conditioning): Outlier spike suppression + Multi-Scale Bilateral Hole Filling.
2. Stage 2 (Deep Prior Prediction): EG-DepthNet predicts high-res depth, Edge-Coherence Gate, and Uncertainty.
   - Supports FP32, BF16/FP16 Mixed Precision, INT8 Weight Quantization (~3.8x compression), and
     Uncertainty-Weighted Geometric Test-Time Augmentation (Flip-TTA).
3. Stage 3 (Uncertainty-Aware Bilateral Energy & Sensor Back-Projection Refinement):
   - Enforces consistency with valid low-res sensor measurements in smooth interior regions.
   - Applies edge-gated Joint Bilateral Upsampling (JBU) snapping at depth discontinuities to eliminate flying pixels.
   - Applies texture-suppressed Guided Filtering on planar regions to eliminate sensor noise & texture-copying.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import torch
import torch.nn.functional as F

from depth_refine.classical.filters import (
    bilateral_mode_snap_and_denoise,
    detect_edges_and_discontinuities,
    guided_filter,
    joint_bilateral_filter,
    mask_bicubic_upsample,
)
from depth_refine.classical.hole_filling import fill_depth_holes, remove_depth_outliers
from depth_refine.classical.interpolation import interpolate_depth
from depth_refine.models.eg_depth_net import EGDepthNet, EGDepthOutput
from depth_refine.models.optimization import (
    QuantizationStats,
    get_model_size_mb,
    optimize_model_for_hardware,
    predict_with_geometric_tta,
    quantize_model_int8,
)


@dataclass
class HybridReconstructionResult:
    """Complete output of the Two-Stage Self-Distilled Hybrid Depth Reconstruction Pipeline."""

    depth_hybrid: torch.Tensor  # (B, 1, H, W) or (1, H, W) final refined high-res depth (Stage 2)
    depth_deep: torch.Tensor  # (B, 1, H, W) or (1, H, W) raw deep network output
    depth_classical_init: torch.Tensor  # (B, 1, H, W) or (1, H, W) classical hole-filled init
    uncertainty: torch.Tensor  # (B, 1, H, W) or (1, H, W) aleatoric uncertainty map
    edge_gate: torch.Tensor  # (B, 1, H, W) or (1, H, W) geometric discontinuity gate in [0, 1]
    depth_mask_bicubic: Optional[torch.Tensor] = None  # Stage-1 Mask-Bicubic initialization (LoftUp (d))
    depth_prior: Optional[torch.Tensor] = None  # Metric-aligned Structural / Depth-Anything Prior
    feature_vis: Optional[torch.Tensor] = None  # Raw 2x/4x/8x Neural Feature Map (LoftUp (c))


def box_filter(x: torch.Tensor, r: int) -> torch.Tensor:
    k = 2 * r + 1
    return F.avg_pool2d(F.pad(x, (r, r, r, r), mode="reflect"), kernel_size=k, stride=1)


def fast_guided_filter(guide_rgb: torch.Tensor, src_depth: torch.Tensor, r: int = 4, eps: float = 1e-4) -> torch.Tensor:
    """Fast luminance-guided filter in PyTorch."""
    if guide_rgb.shape[1] == 3:
        I = 0.299 * guide_rgb[:, 0:1] + 0.587 * guide_rgb[:, 1:2] + 0.114 * guide_rgb[:, 2:3]
    else:
        I = guide_rgb
    p = src_depth
    mean_I = box_filter(I, r)
    mean_p = box_filter(p, r)
    mean_Ip = box_filter(I * p, r)
    cov_Ip = mean_Ip - mean_I * mean_p
    mean_II = box_filter(I * I, r)
    var_I = mean_II - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = box_filter(a, r)
    mean_b = box_filter(b, r)
    return mean_a * I + mean_b


@torch.no_grad()
def zero_training_hybrid_boost(
    d_pred_hr: torch.Tensor,
    d_lr: torch.Tensor,
    rgb_hr: torch.Tensor,
    scale: int,
    disc_thresh: float = 0.25,
    valid_mask_lr: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Zero-training post-processor for Epoch 59 checkpoint:
    1. Iterative Back-Projection (Hard LR Sensor Consistency)
    2. Edge-Guided Residual Propagation
    3. Morphological Min-Max Envelope Clamping
    4. RGB-Affinity Discontinuity Snapping (slashes Boundary RMSE)
    """
    H, W = d_pred_hr.shape[-2], d_pred_hr.shape[-1]
    if valid_mask_lr is not None:
        valid_lr = ((valid_mask_lr > 0.5) & (d_lr > 1e-3) & torch.isfinite(d_lr)).float()
    else:
        valid_lr = ((d_lr > 1e-3) & torch.isfinite(d_lr)).float()

    # Step 0: Fill any missing holes in LR input before anchoring using predicted depth
    d_pred_lr_init = F.interpolate(d_pred_hr, size=d_lr.shape[-2:], mode="area")
    if (valid_lr < 0.5).any():
        d_lr_filled = torch.where(valid_lr > 0.5, d_lr, d_pred_lr_init)
    else:
        d_lr_filled = d_lr

    # Step 1: 2-Iteration Edge-Guided Back-Projection (Fixes global scale/bias drift)
    d_refined = d_pred_hr.clone()
    for _ in range(2):
        d_pred_lr = F.interpolate(d_refined, size=d_lr_filled.shape[-2:], mode="area")
        err_lr = (d_lr_filled - d_pred_lr) * valid_lr
        err_hr = F.interpolate(err_lr, size=(H, W), mode="bicubic", align_corners=False)
        err_hr_guided = fast_guided_filter(rgb_hr, err_hr, r=max(2, scale), eps=1e-4)
        d_refined = d_refined + err_hr_guided

    # Step 2: Local 3x3 Morphological Min-Max Envelope Clamping (Eliminates overshoot halos)
    d_max_lr = F.max_pool2d(F.pad(d_lr_filled, (1, 1, 1, 1), mode="replicate"), kernel_size=3, stride=1)
    d_min_lr = -F.max_pool2d(F.pad(-d_lr_filled, (1, 1, 1, 1), mode="replicate"), kernel_size=3, stride=1)
    d_max_hr = F.interpolate(d_max_lr, size=(H, W), mode="nearest")
    d_min_hr = F.interpolate(d_min_lr, size=(H, W), mode="nearest")
    d_refined = torch.clamp(d_refined, min=d_min_hr, max=d_max_hr)

    # Step 3: Boundary Snapping / Shock Filtering at Depth Discontinuities
    # Where depth jumps > disc_thresh, snap flying ramp pixels to foreground (d_min) or background (d_max)
    is_boundary = ((d_max_hr - d_min_hr) > disc_thresh).float()
    mid_depth = 0.5 * (d_max_hr + d_min_hr)
    sharp_edge_depth = torch.where(d_refined < mid_depth, d_min_hr, d_max_hr)
    # Blend sharply right on high-gradient RGB edges inside the discontinuity band
    I = 0.299 * rgb_hr[:, 0:1] + 0.587 * rgb_hr[:, 1:2] + 0.114 * rgb_hr[:, 2:3]
    grad_I = torch.abs(I - box_filter(I, r=1))
    edge_weight = torch.clamp(grad_I * 8.0, 0.0, 1.0) * is_boundary
    d_refined = (1.0 - 0.55 * edge_weight) * d_refined + (0.55 * edge_weight) * sharp_edge_depth

    # Final subtle edge-preserving polish
    d_refined = fast_guided_filter(rgb_hr, d_refined, r=2, eps=1e-4)
    return torch.clamp(d_refined, min=d_min_hr, max=d_max_hr)


class HybridSensorAnchoredRefiner(torch.nn.Module):
    """
    Zero-Training Hybrid Sensor-Anchored Refinement Module for Epoch 59 Checkpoint.
    GPU-vectorized post-processor performing IBP sensor consistency, residual guided propagation,
    morphological min-max envelope clamping, and RGB-affinity boundary snapping.
    """

    def __init__(self, disc_thresh: float = 0.25) -> None:
        super().__init__()
        self.disc_thresh = disc_thresh

    @torch.no_grad()
    def forward(
        self,
        d_pred_hr: torch.Tensor,
        d_lr: torch.Tensor,
        rgb_hr: torch.Tensor,
        scale: int,
        valid_mask_lr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return zero_training_hybrid_boost(
            d_pred_hr=d_pred_hr,
            d_lr=d_lr,
            rgb_hr=rgb_hr,
            scale=scale,
            disc_thresh=self.disc_thresh,
            valid_mask_lr=valid_mask_lr,
        )


class HybridReconstructor:
    """
    Deep Module orchestrating Classical + Deep + Variational/Bilateral Energy Refinement,
    with built-in support for INT8 Quantization, FP16/BF16 Tensor-Core Inference, and Flip-TTA.
    """

    def __init__(
        self,
        model: Optional[EGDepthNet] = None,
        checkpoint_path: Optional[Path | str] = None,
        device: Optional[str | torch.device] = None,
        precision: Literal["fp32", "bf16", "fp16", "int8"] = "bf16",
    ) -> None:
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.base_model = model if model is not None else EGDepthNet(base_channels=48, prop_steps=4)
        self.base_model.to(self.device)

        if checkpoint_path is not None and Path(checkpoint_path).exists():
            self.load_checkpoint(checkpoint_path)
        else:
            default_ckpt = Path("checkpoints/eg_depthnet_best.pt")
            if default_ckpt.exists():
                self.load_checkpoint(default_ckpt)

        self.base_model.eval()
        self.int8_model: Optional[EGDepthNet] = None
        self.quant_stats: QuantizationStats = optimize_model_for_hardware(
            self.base_model, precision=precision, use_channels_last=False
        )
        self.precision = precision
        if precision == "int8":
            self.enable_int8_quantization()
        self.sensor_anchored_refiner = HybridSensorAnchoredRefiner().to(self.device)

    def load_checkpoint(self, path: Path | str) -> None:
        ckpt = torch.load(Path(path), map_location=self.device, weights_only=False)
        state = ckpt.get("ema_state_dict") or ckpt.get("model_state_dict") or ckpt
        self.base_model.load_state_dict(state, strict=False)
        self.base_model.eval()

    def enable_int8_quantization(self) -> QuantizationStats:
        """Build and cache the INT8 per-channel weight-quantized model (~3.8x memory reduction)."""
        if self.int8_model is None:
            q_model, stats = quantize_model_int8(self.base_model)
            self.int8_model = q_model.to(self.device)
            self.quant_stats = stats
        self.precision = "int8"
        return self.quant_stats

    def get_quantization_report(self) -> QuantizationStats:
        if self.int8_model is None:
            q_model, stats = quantize_model_int8(self.base_model)
            self.int8_model = q_model.to(self.device)
            self.quant_stats = stats
        return self.quant_stats

    @torch.no_grad()
    def reconstruct(
        self,
        depth_lr: torch.Tensor,
        rgb_hr: Optional[torch.Tensor] = None,
        scale: int = 4,
        valid_mask_lr: Optional[torch.Tensor] = None,
        precision: Optional[Literal["fp32", "bf16", "fp16", "int8"]] = None,
        use_tta: bool = False,
    ) -> HybridReconstructionResult:
        """
        Execute the 3-stage Hybrid Depth Reconstruction pipeline.
        """
        squeeze = False
        if depth_lr.ndim == 3:
            depth_lr = depth_lr.unsqueeze(0)
            if rgb_hr is not None and rgb_hr.ndim == 3:
                rgb_hr = rgb_hr.unsqueeze(0)
            if valid_mask_lr is not None and valid_mask_lr.ndim == 3:
                valid_mask_lr = valid_mask_lr.unsqueeze(0)
            squeeze = True

        orig_device = depth_lr.device
        depth_lr = depth_lr.to(self.device)
        if rgb_hr is not None:
            rgb_hr = rgb_hr.to(self.device)
            H, W = rgb_hr.shape[-2], rgb_hr.shape[-1]
        else:
            H, W = depth_lr.shape[-2] * scale, depth_lr.shape[-1] * scale
            rgb_hr = torch.zeros((depth_lr.shape[0], 3, H, W), device=self.device, dtype=depth_lr.dtype)

        if valid_mask_lr is None:
            valid_mask_lr = (depth_lr > 1e-3).to(depth_lr.dtype)
        else:
            valid_mask_lr = valid_mask_lr.to(self.device, dtype=depth_lr.dtype)

        eff_prec = precision or self.precision

        # --- STAGE 1: Classical Outlier Removal & Multi-Scale Hole Pre-Conditioning ---
        depth_clean_lr, mask_clean_lr = remove_depth_outliers(depth_lr, valid_mask_lr)
        rgb_lr = F.interpolate(rgb_hr, size=depth_lr.shape[-2:], mode="bilinear", align_corners=False)
        depth_filled_lr = fill_depth_holes(
            depth_clean_lr, rgb=rgb_lr, valid_mask=mask_clean_lr, method="multiscale_bilateral"
        )
        d_bicubic_hr = interpolate_depth(depth_filled_lr, target_size=(H, W), mode="bicubic")

        # --- STAGE 2: Deep Edge-Gated Network Forward Pass (with optional INT8 / AMP / Flip-TTA) ---
        if eff_prec == "int8":
            if self.int8_model is None:
                self.enable_int8_quantization()
            active_model = self.int8_model
        else:
            active_model = self.base_model

        assert active_model is not None

        def _forward_pass(
            d_in: torch.Tensor, r_in: torch.Tensor, sc: int, m_in: Optional[torch.Tensor]
        ) -> EGDepthOutput:
            if self.device.type == "cuda" and eff_prec in ("bf16", "fp16"):
                amp_dtype = torch.bfloat16 if (eff_prec == "bf16" and torch.cuda.is_bf16_supported()) else torch.float16
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    out = active_model(d_in, rgb_hr=r_in, scale=sc, valid_mask_lr=m_in)
                return EGDepthOutput(
                    depth_hr=out.depth_hr.float(),
                    coarse_depth=out.coarse_depth.float(),
                    uncertainty=out.uncertainty.float(),
                    edge_gate=out.edge_gate.float(),
                    depth_prior=out.depth_prior.float() if out.depth_prior is not None else None,
                    mask_bicubic=out.mask_bicubic.float() if out.mask_bicubic is not None else None,
                    feature_vis=out.feature_vis.float() if out.feature_vis is not None else None,
                )
            return active_model(d_in, rgb_hr=r_in, scale=sc, valid_mask_lr=m_in)

        if use_tta:
            model_out = predict_with_geometric_tta(
                _forward_pass, depth_clean_lr, rgb_hr, scale=scale, valid_mask_lr=mask_clean_lr
            )
        else:
            model_out = _forward_pass(depth_clean_lr, rgb_hr, scale, mask_clean_lr)

        d_deep = model_out.depth_hr
        uncertainty = model_out.uncertainty
        edge_gate = model_out.edge_gate
        d_mask_bic = (
            model_out.mask_bicubic
            if model_out.mask_bicubic is not None
            else mask_bicubic_upsample(depth_clean_lr, rgb_hr=rgb_hr, scale=scale, valid_mask_lr=mask_clean_lr)[0]
        )
        d_prior = model_out.depth_prior if model_out.depth_prior is not None else d_mask_bic
        f_vis = model_out.feature_vis if model_out.feature_vis is not None else edge_gate

        # --- STAGE 3: Boundary-Locked Anisotropic Diffusion & Iterative Back-Projection (IBP) ---
        disc_lr = detect_edges_and_discontinuities(depth_filled_lr, rgb_lr)
        interior_valid_lr = mask_clean_lr * (1.0 - disc_lr.coherence_gate)
        conf_weight = torch.clamp(1.0 / (1.0 + uncertainty * 2.0), min=0.15, max=0.55)
        bp_gain = 0.25 if scale <= 2 else 0.35

        # 3a. Dual-Stream Boundary-Preserving Planar Denoising:
        d_planar = guided_filter(
            d_deep,
            rgb_hr,
            radius=max(2, scale),
            eps=1e-4,
            texture_suppression=True,
        )
        planar_blend = 0.35 if scale <= 2 else 0.65
        d_fused = (1.0 - edge_gate) * (planar_blend * d_planar + (1.0 - planar_blend) * d_deep) + edge_gate * d_deep

        # 3b. 3-Step Boundary-Locked Anisotropic Diffusion restricted to dilated discontinuity mask
        # Eliminates the (b) Per-image optimized halo problem while locking step transitions
        dilated_disc = F.max_pool2d((edge_gate > 0.25).to(d_fused.dtype), kernel_size=5, stride=1, padding=2)
        d_diff = d_fused.clone()
        pad_rgb3 = F.pad(rgb_hr, (1, 1, 1, 1), mode="replicate")
        patches_rgb3 = F.unfold(pad_rgb3, kernel_size=3).view(rgb_hr.shape[0], 3, 9, H, W)
        c_dist3 = ((patches_rgb3 - rgb_hr.unsqueeze(2)) ** 2).mean(dim=1)
        w_rgb3 = torch.exp(-c_dist3 / (2.0 * (0.025**2)))
        diff_alpha = 0.35 if scale <= 2 else 0.65
        for _ in range(3):
            pad_d3 = F.pad(d_diff, (1, 1, 1, 1), mode="replicate")
            patches_d3 = F.unfold(pad_d3, kernel_size=3).view(d_diff.shape[0], 9, H, W)
            w_d3 = torch.exp(-((patches_d3 - d_diff) ** 2) / (2.0 * (0.12**2)))
            w_aniso = w_rgb3 * w_d3
            d_step = (patches_d3 * w_aniso).sum(dim=1, keepdim=True) / torch.clamp(
                w_aniso.sum(dim=1, keepdim=True), min=1e-6
            )
            d_diff = (dilated_disc * diff_alpha) * d_step + (1.0 - dilated_disc * diff_alpha) * d_diff

        # 3c. Multi-Scale Edge-Gated Shock Filter & Mode Snapping
        d_cur = d_diff
        shock_gain = 0.25 if scale <= 2 else 0.75
        shock_mask = torch.clamp((edge_gate - 0.35) / 0.45, min=0.0, max=1.0) * shock_gain
        for k in (3, 5):
            pad = k // 2
            d_erode = -F.max_pool2d(-d_cur, kernel_size=k, stride=1, padding=pad)
            d_dilate = F.max_pool2d(d_cur, kernel_size=k, stride=1, padding=pad)
            d_mid = 0.5 * (d_erode + d_dilate)
            target = torch.where(d_cur <= d_mid, d_erode, d_dilate)
            d_cur = (1.0 - shock_mask) * d_cur + shock_mask * target

        if scale >= 4:
            d_snapped, _ = bilateral_mode_snap_and_denoise(d_cur, rgb_hr, scale=scale)
        else:
            d_snapped = d_cur

        # 3d. Final 2-Step Iterative Back-Projection (IBP) to anchor valid low-res sensor measurements
        d_anchored = d_snapped
        for _ in range(2):
            pred_lr = F.adaptive_avg_pool2d(d_anchored, depth_lr.shape[-2:])
            residual_lr = (depth_clean_lr - pred_lr) * interior_valid_lr
            residual_hr = F.interpolate(residual_lr, size=(H, W), mode="bilinear", align_corners=False)
            d_anchored = d_anchored + bp_gain * conf_weight * (1.0 - edge_gate) * residual_hr

        d_hybrid = torch.clamp(d_anchored, min=0.05)
        d_hybrid = self.sensor_anchored_refiner(
            d_pred_hr=d_hybrid,
            d_lr=depth_clean_lr,
            rgb_hr=rgb_hr,
            scale=scale,
            valid_mask_lr=mask_clean_lr,
        )

        if squeeze:
            return HybridReconstructionResult(
                depth_hybrid=d_hybrid.squeeze(0).to(orig_device),
                depth_deep=d_deep.squeeze(0).to(orig_device),
                depth_classical_init=d_bicubic_hr.squeeze(0).to(orig_device),
                uncertainty=uncertainty.squeeze(0).to(orig_device),
                edge_gate=edge_gate.squeeze(0).to(orig_device),
                depth_mask_bicubic=d_mask_bic.squeeze(0).to(orig_device),
                depth_prior=d_prior.squeeze(0).to(orig_device),
                feature_vis=f_vis.squeeze(0).to(orig_device),
            )

        return HybridReconstructionResult(
            depth_hybrid=d_hybrid.to(orig_device),
            depth_deep=d_deep.to(orig_device),
            depth_classical_init=d_bicubic_hr.to(orig_device),
            uncertainty=uncertainty.to(orig_device),
            edge_gate=edge_gate.to(orig_device),
            depth_mask_bicubic=d_mask_bic.to(orig_device),
            depth_prior=d_prior.to(orig_device),
            feature_vis=f_vis.to(orig_device),
        )
