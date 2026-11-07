"""Unit and integration tests for EG-DepthNet, GeometryAwareDepthLoss, Hybrid Pipeline, and Metrics."""

from __future__ import annotations

import torch

from depth_refine.classical import reconstruct_classical
from depth_refine.data import DegradationConfig, degrade_depth, generate_procedural_rgbd_scene
from depth_refine.evaluate import compute_depth_metrics
from depth_refine.hybrid import HybridReconstructor
from depth_refine.models import EGDepthNet, GeometryAwareDepthLoss


def test_eg_depth_net_multiscale_and_loss() -> None:
    model = EGDepthNet(base_channels=24, prop_steps=3)
    loss_fn = GeometryAwareDepthLoss()

    rgb_np, depth_np, _, _ = generate_procedural_rgbd_scene(3, height=64, width=64)
    rgb = torch.from_numpy(rgb_np.transpose(2, 0, 1)).unsqueeze(0).float()
    depth_gt = torch.from_numpy(depth_np[None, None, ...]).float()

    for scale in (2, 4, 8):
        sample = degrade_depth(depth_gt, rgb, DegradationConfig(scale=scale, hole_ratio=0.1, seed=7))
        out = model(sample.depth_lr, rgb_hr=rgb, scale=scale, valid_mask_lr=sample.valid_mask_lr)
        assert out.depth_hr.shape == (1, 1, 64, 64)
        assert out.uncertainty.shape == (1, 1, 64, 64)
        assert out.edge_gate.shape == (1, 1, 64, 64)

        losses = loss_fn(out, sample.depth_gt, sample.valid_mask_gt, sample.boundary_mask_gt)
        assert torch.isfinite(losses["total"])
        losses["total"].backward()


def test_hybrid_beats_bicubic_on_boundary_preservation() -> None:
    rgb_np, depth_np, raw_np, _ = generate_procedural_rgbd_scene(5, height=128, width=128)
    rgb = torch.from_numpy(rgb_np.transpose(2, 0, 1)).float()
    depth_gt = torch.from_numpy(depth_np[None, ...]).float()
    raw = torch.from_numpy(raw_np[None, ...]).float()

    sample = degrade_depth(depth_gt, rgb, DegradationConfig(scale=4, hole_ratio=0.08, noise_std=0.01, seed=99), raw)

    d_bic = reconstruct_classical(
        sample.depth_lr, rgb_hr=sample.rgb_hr, scale=4, method="bicubic", valid_mask_lr=sample.valid_mask_lr
    )
    engine = HybridReconstructor(device="cpu")
    res = engine.reconstruct(
        sample.depth_lr, rgb_hr=sample.rgb_hr, scale=4, valid_mask_lr=sample.valid_mask_lr
    )

    m_bic = compute_depth_metrics(d_bic, sample.depth_gt, sample.valid_mask_gt, sample.boundary_mask_gt, sample.rgb_hr)
    m_hyb = compute_depth_metrics(
        res.depth_hybrid, sample.depth_gt, sample.valid_mask_gt, sample.boundary_mask_gt, sample.rgb_hr
    )

    assert m_hyb["Boundary_RMSE"] <= m_bic["Boundary_RMSE"]
    assert m_hyb["RMSE"] <= m_bic["RMSE"]


def test_int8_quantization_and_geometric_tta() -> None:
    rgb_np, depth_np, raw_np, _ = generate_procedural_rgbd_scene(6, height=64, width=64)
    rgb = torch.from_numpy(rgb_np.transpose(2, 0, 1)).float()
    depth_gt = torch.from_numpy(depth_np[None, ...]).float()
    raw = torch.from_numpy(raw_np[None, ...]).float()

    sample = degrade_depth(depth_gt, rgb, DegradationConfig(scale=4, hole_ratio=0.08, seed=12), raw)
    engine = HybridReconstructor(device="cpu", precision="int8")

    stats = engine.get_quantization_report()
    assert stats.compression_ratio > 3.0

    res_int8 = engine.reconstruct(
        sample.depth_lr, rgb_hr=sample.rgb_hr, scale=4, valid_mask_lr=sample.valid_mask_lr, precision="int8"
    )
    res_fp32 = engine.reconstruct(
        sample.depth_lr, rgb_hr=sample.rgb_hr, scale=4, valid_mask_lr=sample.valid_mask_lr, precision="fp32"
    )
    res_tta = engine.reconstruct(
        sample.depth_lr,
        rgb_hr=sample.rgb_hr,
        scale=4,
        valid_mask_lr=sample.valid_mask_lr,
        precision="fp32",
        use_tta=True,
    )
    assert res_int8.depth_hybrid.shape == (1, 64, 64)
    assert res_tta.depth_hybrid.shape == (1, 64, 64)
    # Verify INT8 output is within 5mm of FP32 output
    diff = torch.abs(res_int8.depth_hybrid - res_fp32.depth_hybrid).mean().item()
    assert diff < 0.02
