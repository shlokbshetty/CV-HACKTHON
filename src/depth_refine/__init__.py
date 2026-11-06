"""
Depth Refine: Edge-Preserving Depth Map Upsampling and Refinement (ProbID - P4).

Designed with Deep Modules:
1. `depth_refine.data`      - Multi-dataset loaders (NYUv2, Middlebury 2001-2021, KITTI, Synthetic) & Degradation Engine
2. `depth_refine.classical` - Interpolation, Hole-Filling, Joint Bilateral Filter, Guided Filter, Discontinuity Detection
3. `depth_refine.models`    - EG-DepthNet (Edge-Gated Cross-Attention + Anisotropic Spatial Propagation) & Geometry Losses
4. `depth_refine.hybrid`    - 3-Stage Hybrid Depth Reconstruction Pipeline
5. `depth_refine.evaluate`  - Comprehensive Accuracy, Boundary, Texture-Copy, Latency & Memory Benchmarking Suite
"""

__version__ = "1.0.0"

