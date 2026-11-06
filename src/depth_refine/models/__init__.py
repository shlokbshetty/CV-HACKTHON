"""Deep Learning Models, Losses & Hardware Optimization Deep Module."""

from depth_refine.models.eg_depth_net import (
    AnisotropicSpatialPropagation,
    EdgeGatedCrossGuidanceBlock,
    EGDepthNet,
    EGDepthOutput,
    SparsityAwareConv,
)
from depth_refine.models.losses import GeometryAwareDepthLoss, compute_surface_normals
from depth_refine.models.optimization import (
    Int8QuantizedConv2d,
    QuantizationStats,
    get_model_size_mb,
    optimize_model_for_hardware,
    predict_with_geometric_tta,
    quantize_model_int8,
)

__all__ = [
    "AnisotropicSpatialPropagation",
    "EGDepthNet",
    "EGDepthOutput",
    "EdgeGatedCrossGuidanceBlock",
    "GeometryAwareDepthLoss",
    "Int8QuantizedConv2d",
    "QuantizationStats",
    "SparsityAwareConv",
    "compute_surface_normals",
    "get_model_size_mb",
    "optimize_model_for_hardware",
    "predict_with_geometric_tta",
    "quantize_model_int8",
]
