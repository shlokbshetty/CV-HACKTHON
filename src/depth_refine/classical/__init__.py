"""Classical Computer Vision Deep Module."""

from depth_refine.classical.filters import (
    DiscontinuityAnalysis,
    bilateral_mode_snap_and_denoise,
    compute_spatial_gradients,
    detect_edges_and_discontinuities,
    guided_filter,
    joint_bilateral_filter,
    reconstruct_classical,
)
from depth_refine.classical.hole_filling import fill_depth_holes, remove_depth_outliers
from depth_refine.classical.interpolation import interpolate_depth

__all__ = [
    "DiscontinuityAnalysis",
    "bilateral_mode_snap_and_denoise",
    "compute_spatial_gradients",
    "detect_edges_and_discontinuities",
    "fill_depth_holes",
    "guided_filter",
    "interpolate_depth",
    "joint_bilateral_filter",
    "reconstruct_classical",
    "remove_depth_outliers",
]
