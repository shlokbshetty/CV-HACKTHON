"""
Tests for Data Degradation and Dataset Registry Pipelines.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from depth_refine.data.dataset_registry import (
    SyntheticRGBDDataset,
    generate_procedural_rgbd_scene,
    load_dataset,
)
from depth_refine.data.degradation import (
    DegradationConfig,
    compute_discontinuity_mask,
    degrade_depth,
)


def test_procedural_scene_generation():
    rgb, depth, valid_mask, boundary_mask = generate_procedural_rgbd_scene(seed=42, height=128, width=128)
    assert rgb.shape == (128, 128, 3)
    assert depth.shape == (128, 128)
    assert valid_mask.shape == (128, 128)
    assert boundary_mask.shape[:2] == (128, 128)
    assert depth.min() >= 0.5
    assert depth.max() <= 6.0
    assert rgb.min() >= 0.0 and rgb.max() <= 1.0


def test_degradation_pipeline_shapes_and_masks():
    H, W = 128, 128
    rgb_np, depth_np, _, _ = generate_procedural_rgbd_scene(seed=42, height=H, width=W)
    depth = torch.from_numpy(depth_np).float().unsqueeze(0)  # (1, H, W)
    rgb = torch.from_numpy(rgb_np).float().permute(2, 0, 1)  # (3, H, W)

    cfg = DegradationConfig(
        scale=4,
        hole_ratio=0.15,
        structured_boundary_holes=True,
        noise_std=0.01,
        outlier_prob=0.005,
        seed=123,
    )
    degraded = degrade_depth(depth, rgb, cfg)

    assert degraded.depth_lr.shape == (1, H // 4, W // 4)
    assert degraded.valid_mask_lr.shape == (1, H // 4, W // 4)
    assert degraded.rgb_hr.shape == (3, H, W)
    assert degraded.depth_gt.shape == (1, H, W)
    assert degraded.valid_mask_gt.shape == (1, H, W)
    assert degraded.boundary_mask_gt.shape == (1, H, W)

    # Verify that valid_mask_lr is a float mask in [0, 1] reflecting missing holes
    assert (degraded.valid_mask_lr < 0.5).any()
    assert (degraded.valid_mask_lr > 0.5).any()


def test_compute_discontinuity_mask():
    depth = torch.ones((1, 64, 64), dtype=torch.float32)
    depth[:, :, 32:] = 3.0  # Sharp step at col 32
    mask = compute_discontinuity_mask(depth, threshold_rel=0.05, dilation_radius=1)

    assert mask.shape == (1, 64, 64)
    assert mask.dtype == torch.float32
    # Discontinuity boundary must be flagged along column ~31-33
    assert (mask[:, :, 31:33] > 0.5).any()
    # Uniform regions must not be flagged
    assert not (mask[:, :, :20] > 0.5).any()
    assert not (mask[:, :, 45:] > 0.5).any()


def test_synthetic_dataset_loader():
    cfg = DegradationConfig(scale=2, hole_ratio=0.05)
    ds = load_dataset("synthetic", split="train", degradation=cfg, crop_size=64, max_samples=4)
    assert len(ds) == 4

    sample = ds[0]
    assert "depth_lr" in sample
    assert "rgb_hr" in sample
    assert "depth_gt" in sample
    assert "valid_mask_lr" in sample
    assert "boundary_mask_gt" in sample
    assert sample["scale"] == 2
