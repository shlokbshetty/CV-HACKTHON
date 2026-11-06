"""
Comprehensive Evaluation Metrics for Edge-Preserving Depth Upsampling & Refinement:
- Global Depth Accuracy: RMSE (m), MAE (m), AbsRel, Delta1 (< 1.25)
- Image Reconstruction Quality: PSNR (dB), SSIM
- Boundary & Discontinuity Accuracy: Boundary-RMSE (m), Edge-F1 Score
- Texture-Copying Immunity: Texture-Copy Gradient Error (m/px on flat textured surfaces)
- Hardware Efficiency: Latency (ms), Peak Memory Consumption (MB)
"""

from __future__ import annotations

import time
from typing import Callable, Dict, Optional

import torch
import torch.nn.functional as F

from depth_refine.classical.filters import compute_spatial_gradients
from depth_refine.data.degradation import compute_discontinuity_mask


def compute_ssim(
    pred: torch.Tensor,
    gt: torch.Tensor,
    valid_mask: torch.Tensor,
    data_range: Optional[float] = None,
) -> float:
    """Compute Structural Similarity Index (SSIM) over valid depth pixels."""
    if pred.ndim == 3:
        pred = pred.unsqueeze(0)
        gt = gt.unsqueeze(0)
        valid_mask = valid_mask.unsqueeze(0)

    if data_range is None:
        valid_vals = gt[valid_mask > 0.5]
        data_range = float((valid_vals.max() - valid_vals.min()).item()) if valid_vals.numel() > 0 else 6.0
        data_range = max(data_range, 0.5)

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    mu_x = F.avg_pool2d(pred, kernel_size=7, stride=1, padding=3)
    mu_y = F.avg_pool2d(gt, kernel_size=7, stride=1, padding=3)

    sigma_x = torch.clamp(F.avg_pool2d(pred * pred, 7, 1, 3) - mu_x * mu_x, min=0.0)
    sigma_y = torch.clamp(F.avg_pool2d(gt * gt, 7, 1, 3) - mu_y * mu_y, min=0.0)
    sigma_xy = F.avg_pool2d(pred * gt, 7, 1, 3) - mu_x * mu_y

    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / (
        (mu_x * mu_x + mu_y * mu_y + C1) * (sigma_x + sigma_y + C2) + 1e-12
    )
    valid_ssim = (ssim_map * valid_mask).sum() / torch.clamp(valid_mask.sum(), min=1.0)
    return float(torch.clamp(valid_ssim, 0.0, 1.0).item())


def compute_depth_metrics(
    pred: torch.Tensor,
    gt: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    boundary_mask: Optional[torch.Tensor] = None,
    rgb_hr: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Compute all mandatory and advanced depth reconstruction metrics for a single sample or batch.
    """
    if pred.ndim == 3:
        pred = pred.unsqueeze(0)
    if gt.ndim == 3:
        gt = gt.unsqueeze(0)
    if valid_mask is not None and valid_mask.ndim == 3:
        valid_mask = valid_mask.unsqueeze(0)
    if boundary_mask is not None and boundary_mask.ndim == 3:
        boundary_mask = boundary_mask.unsqueeze(0)
    if rgb_hr is not None and rgb_hr.ndim == 3:
        rgb_hr = rgb_hr.unsqueeze(0)

    pred = pred.detach().float()
    gt = gt.detach().float()

    if valid_mask is None:
        valid_mask = ((gt > 0.05) & torch.isfinite(gt)).float()
    else:
        valid_mask = valid_mask.detach().float()

    if boundary_mask is None:
        boundary_mask = compute_discontinuity_mask(gt, threshold_rel=0.025, dilation_radius=2)
    else:
        boundary_mask = boundary_mask.detach().float()

    valid_bool = valid_mask > 0.5
    if valid_bool.sum() == 0:
        return {
            "RMSE": 0.0,
            "MAE": 0.0,
            "AbsRel": 0.0,
            "Delta1": 1.0,
            "PSNR": 50.0,
            "SSIM": 1.0,
            "Boundary_RMSE": 0.0,
            "Edge_F1": 1.0,
            "Texture_Copy_Err": 0.0,
        }

    p_val = torch.clamp(pred[valid_bool], min=1e-3)
    g_val = torch.clamp(gt[valid_bool], min=1e-3)

    diff = p_val - g_val
    mse = torch.mean(diff * diff)
    rmse = float(torch.sqrt(mse).item())
    mae = float(torch.mean(torch.abs(diff)).item())
    abs_rel = float(torch.mean(torch.abs(diff) / g_val).item())

    ratio = torch.maximum(p_val / g_val, g_val / p_val)
    delta1 = float((ratio < 1.25).float().mean().item())

    max_val = float(g_val.max().item())
    psnr = float(10.0 * torch.log10(torch.tensor(max_val * max_val) / torch.clamp(mse, min=1e-10)).item())
    ssim = compute_ssim(pred, gt, valid_mask, data_range=max_val)

    # Boundary-specific RMSE (within depth discontinuity band)
    b_mask = (boundary_mask > 0.5) & valid_bool
    if b_mask.sum() > 0:
        b_diff = pred[b_mask] - gt[b_mask]
        boundary_rmse = float(torch.sqrt(torch.mean(b_diff * b_diff)).item())
    else:
        boundary_rmse = rmse

    # Edge-F1 Score (Geometric Edge Preservation)
    pred_edges = compute_discontinuity_mask(pred, threshold_rel=0.025, dilation_radius=1) > 0.5
    gt_edges = compute_discontinuity_mask(gt, threshold_rel=0.025, dilation_radius=1) > 0.5
    pred_edges = pred_edges & valid_bool
    gt_edges = gt_edges & valid_bool

    tp = float((pred_edges & gt_edges).sum().item())
    fp = float((pred_edges & (~gt_edges)).sum().item())
    fn = float(((~pred_edges) & gt_edges).sum().item())
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    edge_f1 = (2.0 * precision * recall / max(precision + recall, 1e-6)) if (tp + fp + fn) > 0 else 1.0

    # Texture-Copy Gradient Error (spurious depth gradients on flat GT surfaces with strong RGB texture)
    _, _, pred_grad = compute_spatial_gradients(pred)
    _, _, gt_grad = compute_spatial_gradients(gt)
    flat_gt = (gt_grad < 0.015) & valid_bool
    if rgb_hr is not None:
        _, _, rgb_grad = compute_spatial_gradients(rgb_hr.detach().float())
        textured_flat = flat_gt & (rgb_grad > 0.04)
        if textured_flat.sum() > 10:
            tex_copy_err = float(pred_grad[textured_flat].mean().item())
        elif flat_gt.sum() > 0:
            tex_copy_err = float(pred_grad[flat_gt].mean().item())
        else:
            tex_copy_err = 0.0
    else:
        tex_copy_err = float(pred_grad[flat_gt].mean().item()) if flat_gt.sum() > 0 else 0.0

    return {
        "RMSE": rmse,
        "MAE": mae,
        "AbsRel": abs_rel,
        "Delta1": delta1,
        "PSNR": psnr,
        "SSIM": ssim,
        "Boundary_RMSE": boundary_rmse,
        "Edge_F1": edge_f1,
        "Texture_Copy_Err": tex_copy_err,
    }


def measure_runtime_and_memory(
    fn: Callable[[], torch.Tensor],
    device: Optional[torch.device] = None,
) -> tuple[torch.Tensor, float, float]:
    """
    Execute reconstruction callable `fn`, measuring latency in milliseconds and peak memory in MB.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        start_ev.record()
        out = fn()
        end_ev.record()
        torch.cuda.synchronize(device)
        latency_ms = float(start_ev.elapsed_time(end_ev))
        peak_mb = float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
        return out, latency_ms, peak_mb

    t0 = time.perf_counter()
    out = fn()
    latency_ms = (time.perf_counter() - t0) * 1000.0
    # Approximate tensor footprint on CPU
    peak_mb = float(out.numel() * out.element_size() * 8.0 / (1024.0 * 1024.0))
    return out, latency_ms, max(peak_mb, 1.2)
