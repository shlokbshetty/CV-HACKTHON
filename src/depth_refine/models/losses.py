"""
Composite Geometry-Aware Loss Suite for Edge-Preserving Depth Upsampling:
1. Boundary-Weighted Charbonnier + Aleatoric Uncertainty Loss
2. Multi-Scale Sobel Spatial Gradient Matching Loss (scales 1x, 2x, 4x)
3. 3D Surface Normal Cosine Similarity Loss (eliminates surface wrinkling & texture-copying)
4. Edge-Gated Planar Smoothness Loss (penalizes spurious depth gradients on flat textured walls)
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from depth_refine.classical.filters import compute_spatial_gradients
from depth_refine.models.eg_depth_net import EGDepthOutput


def compute_surface_normals(depth: torch.Tensor, focal_scale: float = 100.0) -> torch.Tensor:
    """
    Compute unit 3D surface normals (-dz/dx, -dz/dy, 1) / ||.|| from a depth map (B, 1, H, W).
    """
    gx, gy, _ = compute_spatial_gradients(depth)
    nx = -gx * focal_scale
    ny = -gy * focal_scale
    nz = torch.ones_like(depth)
    normals = torch.cat([nx, ny, nz], dim=1)
    return F.normalize(normals, p=2, dim=1, eps=1e-6)


class GeometryAwareDepthLoss(nn.Module):
    """
    Composite loss function designed to maximize RMSE, MAE, PSNR, SSIM, Boundary-RMSE, and Edge-F1
    while suppressing RGB texture copying.
    """

    def __init__(
        self,
        w_charbonnier: float = 1.0,
        w_grad: float = 0.6,
        w_normal: float = 0.35,
        w_smooth: float = 0.15,
        w_uncertainty: float = 0.05,
        boundary_weight_multiplier: float = 3.0,
        eps: float = 1e-3,
    ) -> None:
        super().__init__()
        self.w_charbonnier = w_charbonnier
        self.w_grad = w_grad
        self.w_normal = w_normal
        self.w_smooth = w_smooth
        self.w_uncertainty = w_uncertainty
        self.boundary_weight_multiplier = boundary_weight_multiplier
        self.eps = eps

    def _charbonnier(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        pixel_weights: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        diff = pred - gt
        loss_map = torch.sqrt(diff * diff + self.eps * self.eps) * pixel_weights * valid_mask
        return loss_map.sum() / torch.clamp(valid_mask.sum(), min=1.0)

    def _multiscale_gradient_loss(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        total_grad_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        for s in (1, 2, 4):
            if s > 1:
                p_s = F.avg_pool2d(pred, kernel_size=s, stride=s)
                g_s = F.avg_pool2d(gt, kernel_size=s, stride=s)
                m_s = (F.avg_pool2d(valid_mask, kernel_size=s, stride=s) > 0.9).to(pred.dtype)
            else:
                p_s, g_s, m_s = pred, gt, valid_mask

            pgx, pgy, _ = compute_spatial_gradients(p_s)
            ggx, ggy, _ = compute_spatial_gradients(g_s)
            diff_g = (torch.abs(pgx - ggx) + torch.abs(pgy - ggy)) * m_s
            total_grad_loss = total_grad_loss + diff_g.sum() / torch.clamp(m_s.sum(), min=1.0)
        return total_grad_loss / 3.0

    def _normal_loss(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        n_pred = compute_surface_normals(pred)
        n_gt = compute_surface_normals(gt)
        cos_sim = (n_pred * n_gt).sum(dim=1, keepdim=True)
        loss_map = (1.0 - cos_sim) * valid_mask
        return loss_map.sum() / torch.clamp(valid_mask.sum(), min=1.0)

    def forward(
        self,
        model_out: EGDepthOutput,
        depth_gt: torch.Tensor,
        valid_mask_gt: Optional[torch.Tensor] = None,
        boundary_mask_gt: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        pred = model_out.depth_hr
        coarse = model_out.coarse_depth
        uncertainty = model_out.uncertainty
        edge_gate = model_out.edge_gate

        if valid_mask_gt is None:
            valid_mask_gt = (depth_gt > 1e-3).to(pred.dtype)
        else:
            valid_mask_gt = valid_mask_gt.to(pred.dtype)

        if boundary_mask_gt is not None:
            pixel_weights = 1.0 + (self.boundary_weight_multiplier - 1.0) * boundary_mask_gt.to(pred.dtype)
        else:
            pixel_weights = torch.ones_like(valid_mask_gt)

        # 1. Charbonnier loss on both final ASPN output and coarse pre-propagation output
        loss_charb_final = self._charbonnier(pred, depth_gt, pixel_weights, valid_mask_gt)
        loss_charb_coarse = self._charbonnier(coarse, depth_gt, pixel_weights, valid_mask_gt)
        loss_charb = 0.75 * loss_charb_final + 0.25 * loss_charb_coarse

        # 2. Multi-scale Sobel gradient matching loss
        loss_grad = self._multiscale_gradient_loss(pred, depth_gt, valid_mask_gt)

        # 3. 3D Surface normal cosine loss
        loss_normal = self._normal_loss(pred, depth_gt, valid_mask_gt)

        # 4. Edge-Gated Texture-Copy Suppression & Gate Supervision
        _, _, gt_d_mag = compute_spatial_gradients(depth_gt)
        _, _, pred_d_mag = compute_spatial_gradients(pred)
        flat_gt_mask = (gt_d_mag < 0.015).to(pred.dtype) * valid_mask_gt
        # Penalize predicted depth gradients in regions where GT depth is flat (kills texture copying!)
        loss_smooth = (pred_d_mag * flat_gt_mask).sum() / torch.clamp(flat_gt_mask.sum(), min=1.0)
        if boundary_mask_gt is not None:
            with torch.amp.autocast("cuda", enabled=False):
                loss_gate = F.binary_cross_entropy(
                    torch.clamp(edge_gate.float(), 1e-4, 1.0 - 1e-4),
                    boundary_mask_gt.float(),
                )
            loss_smooth = loss_smooth + 0.2 * loss_gate.to(pred.dtype)

        # 5. Heteroscedastic aleatoric uncertainty calibration
        abs_err = torch.abs(pred.detach() - depth_gt)
        loss_unc = ((abs_err / uncertainty + torch.log(uncertainty)) * valid_mask_gt).sum() / torch.clamp(
            valid_mask_gt.sum(), min=1.0
        )

        total = (
            self.w_charbonnier * loss_charb
            + self.w_grad * loss_grad
            + self.w_normal * loss_normal
            + self.w_smooth * loss_smooth
            + self.w_uncertainty * loss_unc
        )

        return {
            "total": total,
            "charbonnier": loss_charb,
            "grad": loss_grad,
            "normal": loss_normal,
            "smooth": loss_smooth,
            "uncertainty": loss_unc,
        }
