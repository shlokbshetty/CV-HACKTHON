"""
Dataset Registry and Sensor Degradation Subpackage for Depth Super-Resolution (ProbID - P4).
"""

from __future__ import annotations

from depth_refine.data.dataset_registry import (
    MiddleburyDataset,
    NYUv2Dataset,
    SyntheticRGBDDataset,
    generate_procedural_rgbd_scene,
    load_dataset,
)
from depth_refine.data.degradation import (
    DegradationConfig,
    DegradedSample,
    compute_discontinuity_mask,
    degrade_depth,
)

__all__ = [
    "DegradationConfig",
    "DegradedSample",
    "compute_discontinuity_mask",
    "degrade_depth",
    "generate_procedural_rgbd_scene",
    "load_dataset",
    "NYUv2Dataset",
    "MiddleburyDataset",
    "SyntheticRGBDDataset",
]

