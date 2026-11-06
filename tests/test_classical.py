"""Unit tests for Classical Computer Vision Depth Reconstruction & Degradation Modules."""

from __future__ import annotations

import torch

from depth_refine.classical import (
    detect_edges_and_discontinuities,
    fill_depth_holes,
    guided_filter,
    interpolate_depth,
    joint_bilateral_filter,
    reconstruct_classical,
    remove_depth_outliers,
)
from depth_refine.data import DegradationConfig, degrade_depth, generate_procedural_rgbd_scene


def test_degradation_and_interpolation_scales() -> None:
    rgb_np, depth_np, raw_np, _ = generate_procedural_rgbd_scene(0, height=128, width=128)
    rgb = torch.from_numpy(rgb_np.transpose(2, 0, 1)).float()
    depth = torch.from_numpy(depth_np[None, ...]).float()
    raw = torch.from_numpy(raw_np[None, ...]).float()

    for scale in (2, 4, 8):
        cfg = DegradationConfig(scale=scale, hole_ratio=0.1, noise_std=0.01, seed=42)
        sample = degrade_depth(depth, rgb, cfg, raw_depth_hr=raw)
        assert sample.depth_lr.shape == (1, 128 // scale, 128 // scale)

        for mode in ("nearest", "bilinear", "bicubic"):
            up = interpolate_depth(sample.depth_lr, scale=scale, mode=mode, valid_mask_lr=sample.valid_mask_lr)
            assert up.shape == (1, 128, 128)
            assert torch.isfinite(up).all()
            assert (up > 0.05).all()


def test_outlier_removal_and_hole_filling() -> None:
    rgb_np, depth_np, _, _ = generate_procedural_rgbd_scene(1, height=64, width=64)
    rgb = torch.from_numpy(rgb_np.transpose(2, 0, 1)).float()
    depth = torch.from_numpy(depth_np[None, ...]).float()

    # Inject holes and a single massive outlier spike
    noisy = depth.clone()
    noisy[:, 10:18, 10:18] = 0.0
    noisy[:, 32, 32] = 9.9
    valid_mask = (noisy > 0.05).float()

    cleaned, _ = remove_depth_outliers(noisy, valid_mask)
    assert abs(float(cleaned[0, 32, 32] - depth[0, 32, 32])) < 0.25

    for method in ("telea", "navier_stokes", "multiscale_bilateral", "colorization"):
        filled = fill_depth_holes(noisy, rgb=rgb, valid_mask=valid_mask, method=method)
        assert filled.shape == (1, 64, 64)
        assert (filled[:, 10:18, 10:18] > 0.3).all()


def test_edge_aware_filters_and_texture_suppression() -> None:
    rgb_np, depth_np, _, _ = generate_procedural_rgbd_scene(2, height=64, width=64)
    rgb = torch.from_numpy(rgb_np.transpose(2, 0, 1)).float()
    depth = torch.from_numpy(depth_np[None, ...]).float()

    disc = detect_edges_and_discontinuities(depth, rgb)
    assert disc.coherence_gate.shape == (1, 64, 64)
    assert float(disc.coherence_gate.min()) >= 0.0
    assert float(disc.coherence_gate.max()) <= 1.0

    out_gf = guided_filter(depth, rgb, radius=3, eps=1e-3, texture_suppression=True)
    out_jbf = joint_bilateral_filter(depth, rgb, kernel_size=5)
    assert out_gf.shape == (1, 64, 64)
    assert out_jbf.shape == (1, 64, 64)

    for m in ("nearest", "bilinear", "bicubic", "guided", "jbf", "boundary_aware_classical"):
        rec = reconstruct_classical(depth[:, ::4, ::4], rgb_hr=rgb, scale=4, method=m)
        assert rec.shape == (1, 64, 64)
