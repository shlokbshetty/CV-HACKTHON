"""
Depth Sensor Degradation and Discontinuity Simulation Module (ProbID - P4).

Accurately simulates real-world Time-of-Flight (ToF), Structured Light (Kinect v1/v2),
and LiDAR sensor characteristics:
1. Multi-scale sensor decimation (2x, 4x, 8x downsampling).
2. Heteroscedastic & axial Gaussian measurement noise.
3. Random sensor dropouts & structured IR parallax occlusion shadows at step edges.
4. Multipath reflection / specular outlier spikes.
5. High-precision geometric depth discontinuity detection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from depth_refine.classical.filters import compute_spatial_gradients


@dataclass
class DegradationConfig:
    """Configuration parameters for sensor degradation simulation."""

    scale: int = 4
    hole_ratio: float = 0.08
    structured_boundary_holes: bool = True
    noise_std: float = 0.015
    outlier_prob: float = 0.005
    seed: Optional[int] = None


class DegradedSample:
    """Container for degraded depth maps and corresponding high-resolution targets."""

    def __init__(
        self,
        depth_lr: torch.Tensor,
        valid_mask_lr: torch.Tensor,
        rgb_hr: torch.Tensor,
        depth_gt: torch.Tensor,
        valid_mask_gt: torch.Tensor,
        boundary_mask_gt: torch.Tensor,
    ) -> None:
        self.depth_lr = depth_lr
        self.valid_mask_lr = valid_mask_lr
        self.rgb_hr = rgb_hr
        self.depth_gt = depth_gt
        self.valid_mask_gt = valid_mask_gt
        self.boundary_mask_gt = boundary_mask_gt

    def __getitem__(self, item: str) -> torch.Tensor:
        if hasattr(self, item):
            return getattr(self, item)
        raise KeyError(f"DegradedSample has no attribute '{item}'")

    def __contains__(self, item: str) -> bool:
        return hasattr(self, item)

    def get(self, item: str, default: Any = None) -> Any:
        return getattr(self, item, default)

    def keys(self) -> Tuple[str, ...]:
        return (
            "depth_lr",
            "valid_mask_lr",
            "rgb_hr",
            "depth_gt",
            "valid_mask_gt",
            "boundary_mask_gt",
        )


def compute_discontinuity_mask(
    depth: torch.Tensor,
    threshold_rel: float = 0.025,
    dilation_radius: int = 1,
) -> torch.Tensor:
    """
    Compute binary depth boundary/discontinuity mask at depth step edges.

    Args:
        depth: (..., H, W) Depth tensor.
        threshold_rel: Relative depth difference threshold (e.g. 2.5% of depth).
        dilation_radius: Spatial morphological dilation radius to cover boundary transitions.

    Returns:
        Tensor of same shape with 1.0 at discontinuities and 0.0 elsewhere.
    """
    orig_shape = depth.shape
    if depth.ndim == 2:
        d = depth.unsqueeze(0).unsqueeze(0)
    elif depth.ndim == 3:
        d = depth.unsqueeze(0)
    else:
        d = depth

    B, C, H, W = d.shape
    device = d.device
    dtype = d.dtype

    # Compute spatial gradients
    gx, gy, grad_mag = compute_spatial_gradients(d.float())
    rel_grad = grad_mag / torch.clamp(d.float(), min=0.1)

    raw_mask = (rel_grad > threshold_rel).float()

    if dilation_radius > 0:
        kernel_size = 2 * dilation_radius + 1
        dilated = F.max_pool2d(raw_mask, kernel_size=kernel_size, stride=1, padding=dilation_radius)
    else:
        dilated = raw_mask

    dilated = dilated.to(dtype=dtype, device=device)
    return dilated.view(orig_shape)


def degrade_depth(
    depth_gt: torch.Tensor,
    rgb_hr: torch.Tensor,
    config: DegradationConfig,
    raw_depth_hr: Optional[torch.Tensor] = None,
) -> DegradedSample:
    """
    Simulate low-resolution sensor measurements with holes, noise, and outliers.

    Args:
        depth_gt: High-resolution ground truth depth (..., 1, H, W) or (..., H, W).
        rgb_hr: High-resolution RGB guidance (..., 3, H, W).
        config: Degradation configuration parameters.
        raw_depth_hr: Optional raw sensor measurement with physical hardware holes.

    Returns:
        DegradedSample containing depth_lr, valid_mask_lr, rgb_hr, depth_gt, etc.
    """
    if config.seed is not None:
        generator = torch.Generator(device=depth_gt.device if depth_gt.is_cuda else "cpu")
        generator.manual_seed(config.seed)
    else:
        generator = None

    is_3d = depth_gt.ndim == 3
    if is_3d:
        d_gt = depth_gt.unsqueeze(0)
        r_hr = rgb_hr.unsqueeze(0)
        raw_hr = raw_depth_hr.unsqueeze(0) if raw_depth_hr is not None else None
    else:
        d_gt = depth_gt
        r_hr = rgb_hr
        raw_hr = raw_depth_hr

    B, C, H, W = d_gt.shape
    scale = max(int(config.scale), 1)
    h_lr, w_lr = H // scale, W // scale

    # Valid mask for GT depth
    valid_mask_gt = ((d_gt > 0.05) & torch.isfinite(d_gt)).float()
    boundary_mask_gt = compute_discontinuity_mask(d_gt, threshold_rel=0.025, dilation_radius=2)

    # Downsample depth to low resolution using adaptive average pooling over valid pixels
    d_gt_valid = d_gt * valid_mask_gt
    sum_depth = F.adaptive_avg_pool2d(d_gt_valid, (h_lr, w_lr))
    sum_weight = F.adaptive_avg_pool2d(valid_mask_gt, (h_lr, w_lr))
    valid_mask_lr = (sum_weight > 1e-4).float()
    depth_lr = (sum_depth / torch.clamp(sum_weight, min=1e-6)) * valid_mask_lr

    # If raw sensor depth with real Kinect / LiDAR holes is provided, integrate hardware holes
    if raw_hr is not None:
        raw_valid = ((raw_hr > 0.05) & torch.isfinite(raw_hr)).float()
        # For sparse LiDAR points (KITTI) or Kinect shadows, max pool preserves existing valid beam hits
        raw_valid_lr = F.adaptive_max_pool2d(raw_valid, (h_lr, w_lr))
        valid_mask_lr = valid_mask_lr * raw_valid_lr
        depth_lr = depth_lr * valid_mask_lr

    # 1. Random sensor dropouts (missing ToF/Kinect pixels)
    if config.hole_ratio > 0.0:
        if generator is not None:
            rand_vals = torch.rand(valid_mask_lr.shape, generator=generator, device=valid_mask_lr.device)
        else:
            rand_vals = torch.rand_like(valid_mask_lr)
        drop = (rand_vals >= config.hole_ratio).float()
        valid_mask_lr = valid_mask_lr * drop

    # 2. Structured infrared parallax occlusion shadows at step discontinuities
    if config.structured_boundary_holes:
        disc_lr = compute_discontinuity_mask(depth_lr, threshold_rel=0.03, dilation_radius=1)
        # Shift mask horizontally by 1 pixel to emulate structured light baseline parallax
        shadow = torch.roll(disc_lr, shifts=1, dims=-1)
        valid_mask_lr = valid_mask_lr * (1.0 - shadow * 0.75)
        valid_mask_lr = (valid_mask_lr > 0.5).float()

    # 3. Additive Gaussian axial sensor noise
    if config.noise_std > 0.0:
        if generator is not None:
            noise = torch.randn(depth_lr.shape, generator=generator, device=depth_lr.device)
        else:
            noise = torch.randn_like(depth_lr)
        depth_lr = depth_lr + noise * config.noise_std * valid_mask_lr

    # 4. Outlier multipath reflection / specular spikes
    max_d = float(d_gt_valid.max().item()) if (valid_mask_gt > 0.5).any() else 10.0
    if config.outlier_prob > 0.0:
        if generator is not None:
            outlier_mask = (torch.rand(valid_mask_lr.shape, generator=generator, device=valid_mask_lr.device) < config.outlier_prob).float()
        else:
            outlier_mask = (torch.rand_like(valid_mask_lr) < config.outlier_prob).float()
        spike_val = max(8.5, max_d * 1.1)
        depth_lr = depth_lr * (1.0 - outlier_mask) + outlier_mask * spike_val

    depth_lr = torch.clamp(depth_lr, min=0.05, max=max(10.0, max_d * 1.05)) * valid_mask_lr

    if is_3d:
        depth_lr = depth_lr.squeeze(0)
        valid_mask_lr = valid_mask_lr.squeeze(0)
        r_hr = r_hr.squeeze(0)
        d_gt = d_gt.squeeze(0)
        valid_mask_gt = valid_mask_gt.squeeze(0)
        boundary_mask_gt = boundary_mask_gt.squeeze(0)

    return DegradedSample(
        depth_lr=depth_lr,
        valid_mask_lr=valid_mask_lr,
        rgb_hr=r_hr,
        depth_gt=d_gt,
        valid_mask_gt=valid_mask_gt,
        boundary_mask_gt=boundary_mask_gt,
    )

