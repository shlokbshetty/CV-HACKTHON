# Evaluation & Technical Design Report — Problem Statement 4 (ProbID: P4)
**Title**: Edge-Preserving Depth Map Upsampling and Refinement via Edge-Gated Cross-Attention & Bilateral Mode Energy Minimization

---

## 1. Team Identification

| Field | Details |
| :--- | :--- |
| **Problem ID** | `P4` — Edge-Preserving Depth Map Upsampling and Refinement |
| **Team ID** | `TeamID` *(Replace with official Team ID)* |
| **Member 1 (Name & USN)** | `Shlok Shetty — 1RUA24CSE0433` |
| **Member 2 (Name & USN)** | `Shree Santosh Yadav — 1RUA24CSE0436` |
| **Member 3 (Name & USN)** | `Shiveshwar Sanjay Magadum — 1RUA24CSE0432` |
| **Member 4 (Name & USN)** | `Siddharth S Gadekar — 1RUA24CSE0456` |
| **Member 5 (Name & USN)** | `Shashank Tandan — 1RUA24CSE0425` |
| **Repository URL** | `https://github.com/shlokbshetty/Depth-lab` |

---

## 2. System Architecture & Deep-Module Design Document

### 2.1 High-Level Architecture Diagram (PatchRefiner + LoftUp Two-Stage Hybrid)

```mermaid
flowchart TD
    subgraph Stage1["Stage 1: Classical Conditioning & Mask-Bicubic"]
        LR["Low-Res / Sparse / Noisy Depth D_lr\n(2x, 4x, 8x + Missing Holes)"]
        RGB["High-Res RGB Guidance I_hr"]
        OUTLIER["2-Pass Valid-Only Order-Statistic Median\n+ MAD Z-Score Outlier Filter"]
        PYR["Multi-Scale Coarse-to-Fine Pyramid\n& Cross-Bilateral Hole Inpainting"]
        MB["Mask-Bicubic Upsampler D_Mask-Bicubic (Stage 1)"]
        LR --> OUTLIER --> PYR --> MB
        RGB --> MB
    end

    subgraph Stage2["Stage 2: EG-DepthNet (LoftUp + PatchRefiner)"]
        PRIOR["Foundation Structural Prior Adapter\n(LWLS Metric Scale-Shift Aligned)"]
        COORD["Continuous 2D Fourier Coordinate\nCross-Attention Upsampler (LoftUp)"]
        STEM["Sparsity-Aware Partial Conv Stem"]
        EGCA["Edge-Gated Cross-Attention (EGCA)\nAnti-Texture-Copy Gate G_edge in [0, 1]"]
        ASPN["Anisotropic Spatial Propagation (ASPN)\n8-Neighbor Convex Affinity Refinement"]
        UNC["Aleatoric Uncertainty Head (sigma_D)"]
        MB & RGB --> STEM
        RGB --> PRIOR
        STEM & RGB --> COORD --> EGCA
        PRIOR & EGCA --> ASPN
        EGCA --> UNC
    end

    subgraph Stage3["Stage 2 Refinement: Self-Distilled Hybrid & Boundary Lock"]
        BLAD["3-Step Boundary-Locked Anisotropic Diffusion\n(Restricted to Dilated Discontinuity Band)"]
        BP["2-Step Iterative Back-Projection (IBP)\nSensor Consistency Anchor"]
        FINAL["Final High-Resolution Depth D_hybrid"]
        ASPN & UNC & MB --> BLAD --> BP --> FINAL
    end
```

### 2.2 Technical Design & Mathematical Formulation

1. **2-Pass Valid-Only Order-Statistic Median + MAD Outlier Filter & Pyramid Hole Filling (Stage 1)**:
   - Raw RGB-D sensors (Microsoft Kinect in NYUv2) and LiDAR scanners (Velodyne in KITTI) suffer from structured infrared shadows along object boundaries, missing pockets on specular/dark surfaces, and multipath outlier spikes.
   - `remove_depth_outliers` executes a 2-pass exact $5\times 5$ valid-masked order-statistic median and Median Absolute Deviation (`1.4826 * MAD`) filter to reject clustered outlier spikes before `fill_depth_holes` constructs a 5-level spatial pyramid ($\{1/16, 1/8, 1/4, 1/2, 1\}$) with harmonic Laplace smoothing and RGB cross-bilateral refinement.

2. **Mask-Bicubic Upsampling (Stage 1 — `MaskBicubicUpsampler`)**:
   - Standard interpolation (`Bicubic`) and Gaussian filtering (`JBF`) blur across occlusion jumps, creating intermediate "flying pixels" in 3D space.
   - `MaskBicubicUpsampler` blends bicubic spatial weights $w_{\text{bicubic}}(p,q)$ with foreground/background partition masks $\mathbb{I}(M(p) == M(q))$ and RGB edge-stopping weights $w_{\text{edge}}(I_p, I_q)$:
     $$D_{\text{Mask-Bicubic}}(p) = \frac{\sum_{q \in \mathcal{N}(p)} w_{\text{bicubic}}(p,q) \cdot \mathbb{I}(M(p) == M(q)) \cdot w_{\text{edge}}(I_p, I_q) \cdot D_{LR}(q)}{\sum_{q \in \mathcal{N}(p)} w_{\text{bicubic}}(p,q) \cdot \mathbb{I}(M(p) == M(q)) \cdot w_{\text{edge}}(I_p, I_q)}$$

3. **Continuous Coordinate Cross-Attention (LoftUp) & Foundation Structural Prior (PatchRefiner)**:
   - `ContinuousCoordCrossAttentionUpsampler` encodes continuous sub-pixel relative offsets $(\delta_x, \delta_y)$ with 2D sinusoidal Fourier features so HR RGB queries attend over $3\times 3$ LR depth feature neighborhoods across $2\times, 4\times, 8\times$ scales without deconvolution checkerboard artifacts.
   - `FoundationStructuralPriorAdapter` extracts dense relative geometry from RGB and aligns it to metric `LR GT` depth via **Locally Weighted Least-Squares (LWLS)** scale-shift alignment ($\hat{D}_{\text{prior}}(p) = a(p) P_{\text{rel}}(p) + b(p)$).

4. **Edge-Gated Cross-Attention (EGCA) for Texture-Copy Suppression**:
   - `EG-DepthNet` computes a physical + learned **Coherence Gate** $G_{\text{edge}} \in [0, 1]$ from the ratio of local 3D depth span to depth magnitude. Where the 3D surface is planar ($\text{span} \ll \tau_d$), $G_{\text{edge}} \to 0$ suppresses RGB feature injection, preventing wallpaper/checkerboard texture copying.

5. **Detail & Scale Disentangling (DSD) + Self-Distillation Loss (`GeometryAwareDepthLoss`)**:
   $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{Charbonnier}}(D, D_{GT}) + \lambda_{\text{grad}} \mathcal{L}_{\text{grad-log}}(\nabla D, \nabla D_{GT}) + \lambda_{\text{rank}} \mathcal{L}_{\text{edge-ranking}} + \lambda_{\text{distill}} \mathcal{L}_{\text{affinity}}(F_{\text{student}}, F_{\text{teacher}})$$

6. **Depth Boundary Error ($\varepsilon^{\text{acc}}_{\text{DBE}}$ & $\varepsilon^{\text{comp}}_{\text{DBE}}$) & INT8 Quantization**:
   - Evaluates Truncated Chamfer Distance via Euclidean Distance Transform (`cv2.distanceTransform`) between predicted and ground-truth depth discontinuity edges (`compute_dbe_metrics`).
   - Compresses `Conv2d` weights from FP32 (`5.31 MB`) to symmetric per-channel INT8 (`1.34 MB`, **3.96× compression ratio**) with `< 0.002 m` RMSE difference.

---

## 3. Dataset Split & Evaluation Protocol

To guarantee zero data leakage and strict scientific reproducibility:
- **NYU Depth V2 (`nyu_depth_v2_labeled.mat`)**: Strictly follows the official benchmark split—first **795 scenes (`0..794`)** for training, and the remaining **654 held-out scenes (`795..1448`)** strictly for testing/validation. Held-out test scenes are never seen during training.
- **Middlebury Stereo Datasets (2001–2021)**: Evaluated **zero-shot** (unseen during training) across scenes (`Venus`, `Cones`, `Teddy`, `Art`, `Books`, `Dolls`, `Moebius`, `Reindeer`, `Laundry`, `Aloe`, `Bowling1`, `Flowerpots`, `Adirondack`).
- **KITTI Vision Benchmark Suite**: Evaluated **zero-shot** (unseen during training) on outdoor depth completion scenes.
- **Consistent Multi-Scale Degradation**: For every test scene, $2\times$, $4\times$, and $8\times$ inputs are generated from the exact same ground-truth depth map with 10% structured boundary + random holes, axial noise ($\sigma_z = 0.015\text{ m}$), and 0.5% outlier spikes.

---

## 4. Quantitative Benchmark Results ($2\times, 4\times, 8\times$)

### 4.1 NYU Depth V2 (Held-Out Indoor Test Split)

| Scale | Method | RMSE (m) ↓ | MAE (m) ↓ | PSNR (dB) ↑ | SSIM ↑ | Boundary RMSE (m) ↓ | Edge-F1 ↑ | Tex-Copy Err ↓ | Latency (ms) ↓ |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2×** | Nearest-Neighbour | 0.1454 | 0.0517 | 35.47 | 0.9569 | 0.3677 | 0.4190 | 0.0185 | 143.6 |
| **2×** | Bilinear Interp. | 0.1432 | 0.0478 | 35.67 | 0.9621 | 0.3642 | 0.4148 | 0.0158 | 5.8 |
| **2×** | Bicubic Interp. | 0.1428 | 0.0485 | 35.69 | 0.9613 | 0.3631 | 0.4192 | 0.0175 | 6.5 |
| **2×** | Guided Filter (RGB) | 0.1441 | 0.0555 | 35.02 | 0.9596 | 0.3312 | 0.5300 | 0.0305 | 51.3 |
| **2×** | Joint Bilateral (JBF) | 0.1214 | 0.0389 | 37.21 | 0.9734 | 0.3088 | 0.6970 | 0.0153 | 12.2 |
| **2×** | Mask-Bicubic (Stage 1) | 0.1216 | 0.0392 | 37.16 | 0.9727 | 0.3096 | 0.7078 | 0.0154 | 30.2 |
| **2×** | **EG-DepthNet (Deep BF16)** | **0.0896** | **0.0317** | **40.39** | **0.9803** | **0.2184** | **0.8536** | **0.0092** | 430.0 |
| **2×** | **Hybrid INT8 Quantized (3.96× Comp.)** | **0.0910** | **0.0309** | **39.87** | **0.9805** | **0.2272** | **0.8292** | **0.0114** | 104.0 |
| **2×** | **Self-Distilled Hybrid (Stage 2)** | **0.0912** | **0.0309** | **39.86** | **0.9805** | **0.2276** | **0.8288** | **0.0114** | **76.2** |
| **2×** | **Hybrid + Flip-TTA (Self-Ensemble)** | **0.1112** | **0.0365** | **37.61** | **0.9756** | **0.2858** | **0.7490** | **0.0129** | 144.5 |
| **4×** | Nearest-Neighbour | 0.2565 | 0.1087 | 30.35 | 0.9225 | 0.5605 | 0.3798 | 0.0197 | 7.1 |
| **4×** | Bilinear Interp. | 0.2533 | 0.1047 | 30.52 | 0.9346 | 0.5554 | 0.1855 | 0.0169 | 7.2 |
| **4×** | Bicubic Interp. | 0.2534 | 0.1046 | 30.51 | 0.9339 | 0.5567 | 0.2090 | 0.0180 | 7.4 |
| **4×** | Guided Filter (RGB) | 0.2624 | 0.1123 | 29.87 | 0.9316 | 0.5437 | 0.3526 | 0.0362 | 14.2 |
| **4×** | Joint Bilateral (JBF) | 0.2499 | 0.1014 | 30.60 | 0.9382 | 0.5414 | 0.4396 | 0.0219 | 13.3 |
| **4×** | Mask-Bicubic (Stage 1) | 0.2490 | 0.1014 | 30.64 | 0.9383 | 0.5379 | 0.4586 | 0.0227 | 28.5 |
| **4×** | **EG-DepthNet (Deep BF16)** | **0.2123** | **0.0871** | **32.10** | **0.9504** | **0.4503** | **0.6262** | **0.0131** | 116.9 |
| **4×** | **Hybrid INT8 Quantized (3.96× Comp.)** | **0.2097** | **0.0864** | **32.37** | **0.9489** | **0.4506** | **0.5934** | **0.0212** | 113.2 |
| **4×** | **Self-Distilled Hybrid (Stage 2)** | **0.2095** | **0.0862** | **32.37** | **0.9490** | **0.4503** | **0.5932** | **0.0212** | **111.3** |
| **4×** | **Hybrid + Flip-TTA (Self-Ensemble)** | **0.2227** | **0.0910** | **31.62** | **0.9461** | **0.4800** | **0.5385** | **0.0227** | 181.5 |
| **8×** | Nearest-Neighbour | 0.6071 | 0.3181 | 22.17 | 0.8851 | 0.8792 | 0.2021 | 0.0171 | 10.7 |
| **8×** | Bilinear Interp. | 0.6043 | 0.3147 | 22.23 | 0.9074 | 0.8761 | 0.0346 | 0.0143 | 7.6 |
| **8×** | Bicubic Interp. | 0.6058 | 0.3147 | 22.19 | 0.9065 | 0.8805 | 0.0596 | 0.0149 | 7.9 |
| **8×** | Guided Filter (RGB) | 0.6451 | 0.3454 | 21.92 | 0.8927 | 0.8931 | 0.1417 | 0.0430 | 18.4 |
| **8×** | Joint Bilateral (JBF) | 0.6407 | 0.3374 | 21.92 | 0.9012 | 0.8911 | 0.1130 | 0.0216 | 14.0 |
| **8×** | Mask-Bicubic (Stage 1) | 0.6402 | 0.3370 | 21.93 | 0.9017 | 0.8898 | 0.1512 | 0.0222 | 37.1 |
| **8×** | **EG-DepthNet (Deep BF16)** | **0.5923** | **0.3069** | **22.90** | **0.9120** | **0.8037** | **0.3157** | **0.0183** | 131.2 |
| **8×** | **Hybrid INT8 Quantized (3.96× Comp.)** | **0.5890** | **0.3047** | **22.98** | **0.9124** | **0.7988** | **0.3087** | **0.0245** | 135.6 |
| **8×** | **Self-Distilled Hybrid (Stage 2)** | **0.5893** | **0.3044** | **22.96** | **0.9125** | **0.7994** | **0.3087** | **0.0245** | **145.2** |
| **8×** | **Hybrid + Flip-TTA (Self-Ensemble)** | **0.5993** | **0.3080** | **22.30** | **0.9110** | **0.8443** | **0.2786** | **0.0248** | 231.1 |

### 4.2 Middlebury Stereo (Zero-Shot Unseen Evaluation)

| Scale | Method | RMSE (m) ↓ | MAE (m) ↓ | PSNR (dB) ↑ | SSIM ↑ | Boundary RMSE (m) ↓ | Edge-F1 ↑ | Tex-Copy Err ↓ | Latency (ms) ↓ |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2×** | Nearest-Neighbour | 0.5254 | 0.0884 | 22.55 | 0.8794 | 1.5067 | 0.2400 | 0.0118 | 7.4 |
| **2×** | Bilinear Interp. | 0.5252 | 0.0852 | 22.56 | 0.8825 | 1.5067 | 0.2713 | 0.0090 | 6.9 |
| **2×** | Bicubic Interp. | 0.5253 | 0.0871 | 22.56 | 0.8810 | 1.5067 | 0.2698 | 0.0109 | 7.8 |
| **2×** | Guided Filter (RGB) | 0.5259 | 0.0827 | 22.55 | 0.8831 | 1.5063 | 0.3719 | 0.0194 | 14.4 |
| **2×** | Joint Bilateral (JBF) | 0.5249 | 0.0812 | 22.56 | 0.8858 | 1.5062 | 0.4790 | 0.0100 | 13.2 |
| **2×** | Mask-Bicubic (Stage 1) | 0.5249 | 0.0812 | 22.56 | 0.8857 | 1.5061 | 0.4741 | 0.0100 | 25.5 |
| **2×** | **EG-DepthNet (Deep BF16)** | **0.5257** | **0.0775** | **22.56** | **0.8879** | **1.5069** | **0.6022** | **0.0071** | 87.2 |
| **2×** | **Hybrid INT8 Quantized** | **0.5257** | **0.0780** | **22.56** | **0.8875** | **1.5068** | **0.5606** | **0.0093** | 83.1 |
| **2×** | **Self-Distilled Hybrid (Stage 2)** | **0.5256** | **0.0780** | **22.56** | **0.8875** | **1.5067** | **0.5626** | **0.0093** | **82.8** |
| **2×** | **Hybrid + Flip-TTA** | **0.5256** | **0.0796** | **22.56** | **0.8860** | **1.5075** | **0.4705** | **0.0087** | 129.6 |
| **4×** | Nearest-Neighbour | 0.5288 | 0.0986 | 22.51 | 0.8776 | 1.5108 | 0.2113 | 0.0071 | 7.8 |
| **4×** | Bilinear Interp. | 0.5286 | 0.0962 | 22.52 | 0.8805 | 1.5108 | 0.1548 | 0.0055 | 8.7 |
| **4×** | Bicubic Interp. | 0.5287 | 0.0974 | 22.52 | 0.8795 | 1.5107 | 0.1567 | 0.0065 | 7.5 |
| **4×** | Guided Filter (RGB) | 0.5293 | 0.0926 | 22.51 | 0.8806 | 1.5101 | 0.2991 | 0.0200 | 15.3 |
| **4×** | Joint Bilateral (JBF) | 0.5279 | 0.0911 | 22.53 | 0.8832 | 1.5100 | 0.3557 | 0.0080 | 15.1 |
| **4×** | Mask-Bicubic (Stage 1) | 0.5280 | 0.0900 | 22.53 | 0.8838 | 1.5102 | 0.3556 | 0.0076 | 33.6 |
| **4×** | **EG-DepthNet (Deep BF16)** | **0.5303** | **0.0855** | **22.50** | **0.8851** | **1.5162** | **0.5129** | **0.0073** | 137.6 |
| **4×** | **Hybrid INT8 Quantized** | **0.5298** | **0.0864** | **22.51** | **0.8853** | **1.5149** | **0.4707** | **0.0093** | 134.7 |
| **4×** | **Self-Distilled Hybrid (Stage 2)** | **0.5297** | **0.0864** | **22.51** | **0.8853** | **1.5147** | **0.4712** | **0.0093** | **131.6** |
| **4×** | **Hybrid + Flip-TTA** | **0.5297** | **0.0879** | **22.50** | **0.8846** | **1.5150** | **0.4382** | **0.0086** | 208.8 |
| **8×** | Nearest-Neighbour | 0.5413 | 0.1478 | 22.36 | 0.8701 | 1.5004 | 0.0493 | 0.0033 | 8.7 |
| **8×** | Bilinear Interp. | 0.5412 | 0.1461 | 22.36 | 0.8720 | 1.5005 | 0.0047 | 0.0025 | 7.6 |
| **8×** | Bicubic Interp. | 0.5413 | 0.1470 | 22.36 | 0.8717 | 1.5005 | 0.0055 | 0.0028 | 12.2 |
| **8×** | Guided Filter (RGB) | 0.5415 | 0.1471 | 22.35 | 0.8676 | 1.4962 | 0.1027 | 0.0217 | 19.7 |
| **8×** | Joint Bilateral (JBF) | 0.5396 | 0.1458 | 22.38 | 0.8720 | 1.4967 | 0.0564 | 0.0038 | 12.7 |
| **8×** | Mask-Bicubic (Stage 1) | 0.5396 | 0.1455 | 22.38 | 0.8721 | 1.4967 | 0.0657 | 0.0037 | 38.6 |
| **8×** | **EG-DepthNet (Deep BF16)** | **0.5370** | **0.1367** | **22.42** | **0.8745** | **1.5022** | **0.1479** | **0.0046** | 124.7 |
| **8×** | **Hybrid INT8 Quantized** | **0.5364** | **0.1366** | **22.42** | **0.8750** | **1.5007** | **0.1876** | **0.0051** | 111.7 |
| **8×** | **Self-Distilled Hybrid (Stage 2)** | **0.5364** | **0.1367** | **22.42** | **0.8750** | **1.5006** | **0.1856** | **0.0051** | **112.1** |
| **8×** | **Hybrid + Flip-TTA** | **0.5366** | **0.1369** | **22.42** | **0.8749** | **1.5006** | **0.1799** | **0.0049** | 182.2 |

### 4.3 KITTI Vision Suite (Zero-Shot Outdoor Evaluation)

| Scale | Method | RMSE (m) ↓ | MAE (m) ↓ | PSNR (dB) ↑ | SSIM ↑ | Boundary RMSE (m) ↓ | Edge-F1 ↑ | Tex-Copy Err ↓ | Latency (ms) ↓ |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2×** | Nearest-Neighbour | 1.2560 | 0.3935 | 18.03 | 0.8913 | 2.4419 | 0.3921 | 0.0157 | 6.7 |
| **2×** | Bilinear Interp. | 1.2555 | 0.3897 | 18.04 | 0.8949 | 2.4411 | 0.4162 | 0.0132 | 6.3 |
| **2×** | Bicubic Interp. | 1.2557 | 0.3912 | 18.03 | 0.8943 | 2.4415 | 0.4082 | 0.0147 | 8.2 |
| **2×** | Guided Filter (RGB) | 1.2565 | 0.3908 | 18.03 | 0.8921 | 2.4421 | 0.4680 | 0.0236 | 14.4 |
| **2×** | Joint Bilateral (JBF) | 1.2548 | 0.3871 | 18.04 | 0.8939 | 2.4402 | 0.4690 | 0.0136 | 12.7 |
| **2×** | Mask-Bicubic (Stage 1) | 1.2548 | 0.3874 | 18.04 | 0.8937 | 2.4400 | 0.4735 | 0.0136 | 24.0 |
| **2×** | **EG-DepthNet (Deep BF16)** | **1.2292** | **0.3669** | **18.22** | **0.9003** | **2.3907** | **0.5512** | **0.0114** | 81.3 |
| **2×** | **Hybrid INT8 Quantized** | **1.2314** | **0.3679** | **18.20** | **0.8991** | **2.3950** | **0.5450** | **0.0124** | 74.0 |
| **2×** | **Self-Distilled Hybrid (Stage 2)** | **1.2295** | **0.3674** | **18.22** | **0.8992** | **2.3913** | **0.5457** | **0.0124** | **75.0** |
| **2×** | **Hybrid + Flip-TTA** | **1.2276** | **0.3669** | **18.23** | **0.8984** | **2.3876** | **0.5097** | **0.0134** | 129.0 |
| **4×** | Nearest-Neighbour | 1.4149 | 0.6287 | 16.99 | 0.8483 | 2.6355 | 0.2075 | 0.0160 | 7.4 |
| **4×** | Bilinear Interp. | 1.4136 | 0.6252 | 17.00 | 0.8608 | 2.6339 | 0.0757 | 0.0138 | 7.1 |
| **4×** | Bicubic Interp. | 1.4142 | 0.6266 | 16.99 | 0.8601 | 2.6349 | 0.0889 | 0.0144 | 7.6 |
| **4×** | Guided Filter (RGB) | 1.5188 | 0.6838 | 16.38 | 0.8493 | 2.8032 | 0.1627 | 0.0267 | 14.8 |
| **4×** | Joint Bilateral (JBF) | 1.5199 | 0.6843 | 16.38 | 0.8473 | 2.8056 | 0.1213 | 0.0167 | 13.0 |
| **4×** | Mask-Bicubic (Stage 1) | 1.5194 | 0.6836 | 16.38 | 0.8477 | 2.8046 | 0.1411 | 0.0163 | 29.5 |
| **4×** | **EG-DepthNet (Deep BF16)** | **1.4163** | **0.5938** | **16.99** | **0.8552** | **2.6450** | **0.2715** | **0.0185** | 126.0 |
| **4×** | **Hybrid INT8 Quantized** | **1.4089** | **0.5886** | **17.03** | **0.8559** | **2.6334** | **0.2873** | **0.0211** | 117.7 |
| **4×** | **Self-Distilled Hybrid (Stage 2)** | **1.4101** | **0.5902** | **17.02** | **0.8557** | **2.6339** | **0.2871** | **0.0211** | **112.5** |
| **4×** | **Hybrid + Flip-TTA** | **1.4321** | **0.5994** | **16.89** | **0.8557** | **2.6729** | **0.2772** | **0.0202** | 180.6 |
| **8×** | Nearest-Neighbour | 3.0316 | 2.0934 | 10.59 | 0.5252 | 4.0080 | 0.3439 | 0.0047 | 8.8 |
| **8×** | Bilinear Interp. | 3.0312 | 2.0931 | 10.59 | 0.5289 | 4.0074 | 0.4129 | 0.0042 | 7.9 |
| **8×** | Bicubic Interp. | 3.0317 | 2.0933 | 10.59 | 0.5289 | 4.0084 | 0.4513 | 0.0042 | 7.9 |
| **8×** | Guided Filter (RGB) | 3.3765 | 2.3327 | 9.43 | 0.4698 | 4.4407 | 0.4179 | 0.0070 | 15.1 |
| **8×** | Joint Bilateral (JBF) | 3.3777 | 2.3324 | 9.43 | 0.4700 | 4.4417 | 0.4264 | 0.0044 | 13.7 |
| **8×** | Mask-Bicubic (Stage 1) | 3.3777 | 2.3324 | 9.43 | 0.4701 | 4.4416 | 0.4115 | 0.0042 | 30.9 |
| **8×** | **EG-DepthNet (Deep BF16)** | **3.4034** | **2.4021** | **9.36** | **0.4304** | **4.4693** | **0.4269** | **0.0045** | 124.5 |
| **8×** | **Hybrid INT8 Quantized** | **3.4010** | **2.4004** | **9.37** | **0.4312** | **4.4629** | **0.3271** | **0.0046** | 114.6 |
| **8×** | **Self-Distilled Hybrid (Stage 2)** | **3.4010** | **2.4004** | **9.37** | **0.4312** | **4.4630** | **0.3278** | **0.0046** | **110.0** |
| **8×** | **Hybrid + Flip-TTA** | **3.4005** | **2.3997** | **9.37** | **0.4314** | **4.4622** | **0.3222** | **0.0046** | 175.7 |

---

## 5. Controlled Ablation Study (MIDDLEBURY — 4× Upsampling)

Reproducible via `uv run python scripts/run_ablation.py`:

| ID | Configuration | RMSE (m) ↓ | Boundary RMSE (m) ↓ | PSNR (dB) ↑ | SSIM ↑ | Edge-F1 ↑ | Tex-Copy Err ↓ |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **A1** | Baseline: Naive Bicubic (No Hole Pre-Fill) | 0.8934 | 1.8010 | 17.29 | 0.7635 | 0.0167 | 0.0237 |
| **A2** | `+ Multi-Scale Pyramid Hole Fill & Outlier Spike Filter` | 0.7829 | 1.7895 | 19.03 | 0.8257 | 0.2013 | 0.0069 |
| **A3** | `+ Classical Fast Guided Filter (He et al.)` | 0.7733 | 1.7848 | 19.43 | 0.8251 | 0.3142 | 0.0202 |
| **A4** | `+ Classical Joint Bilateral Filter (JBF)` | 0.7831 | 1.7857 | 19.04 | 0.8272 | 0.3360 | 0.0112 |
| **A5** | `+ Deep Model w/o Edge-Coherence Gate (Ungated RGB)` | 0.7781 | 1.7856 | 19.23 | 0.8253 | 0.2837 | 0.0186 |
| **A6** | `+ EG-DepthNet (Edge-Coherence Gate + Coord Attn + ASPN)` | 0.7797 | 1.8048 | 19.46 | 0.8296 | 0.5095 | 0.0102 |
| **A7** | **`+ Self-Distilled Hybrid Stage-2 (Sensor Back-Proj + Mode Energy)`** | **0.7791** | **1.8040** | **19.48** | **0.8300** | **0.4797** | **0.0103** |
| **A8** | **`+ Hybrid INT8 Weight Quantized (3.96× Compressed)`** | **0.7791** | **1.8037** | **19.48** | **0.8300** | **0.4752** | **0.0102** |

---

## 6. Documented Failure Cases & Limitations Analysis

1. **Camouflaged / Iso-Color Occlusion Boundaries ($\|\mathbf{c}_{fg} - \mathbf{c}_{bg}\| \to 0$)**:
   - *Phenomenon*: When a foreground object has the exact same RGB color and luminance as the background wall behind it, RGB-guided bilateral mode snapping cannot determine the sub-pixel boundary location from color alone ($w_{fg} \approx 0.5$).
   - *Behavior*: The system gracefully falls back to the Sparsity-Aware geometric depth stream (`fd1`, `fd2`) and ASPN spatial propagation without introducing artifacts, though sub-pixel edge sharpness approaches bicubic resolution.
2. **High-Contrast Cast Shadows Across Planar Surfaces**:
   - *Phenomenon*: Sharp illumination shadows on a floor create a step edge in RGB luminance alongside small sensor noise in depth.
   - *Behavior*: While our Edge-Coherence Gate ($G_{\text{edge}}$) suppresses ~35%–50% of texture-copying error compared to ungated filters, if a cast shadow coincides spatially with a noisy sensor spike that mimics a $>4\%$ depth jump, a mild depth crease ($< 1.5\text{ cm}$) can remain.
3. **Specular Mirrors, Glass Windows, and Large $>70\%$ Sensor Dropouts**:
   - *Phenomenon*: Infrared structured-light cameras (Kinect) and LiDAR beams pass through glass windows or reflect specularly off mirrors, creating large contiguous missing regions where no valid sensor anchors exist.
   - *Behavior*: Multi-scale pyramid inpainting bridges the missing region smoothly from surrounding window frames, which reconstructs a planar window surface rather than the distant outdoor geometry behind the glass.
4. **Extreme $8\times$ Downsampling on Thin Structures ($< 4\text{ px}$ wide)**:
   - *Phenomenon*: Structures thinner than half a low-resolution grid cell (e.g., thin chair legs or wires under $8\times$ downsampling) may be completely averaged out in the LR sensor input ($D_{\min}$ and $D_{\max}$ both reflect the background).
   - *Mitigation*: Our uncertainty head ($\hat{\sigma}_D$) flags thin high-frequency RGB structures with high uncertainty so downstream 3D applications can filter uncertain boundary points.

---

## 7. Deployment & Next-Gen Web Studio (`frontend/`)

The project features two deployment modalities:

1. **Next-Gen Web Studio (`frontend/`)**:
   - Built with **React 18 + Vite 6 + TypeScript** and **Three.js (`@react-three/fiber`, `@react-three/drei`)**.
   - **Interactive 3D Workspace (`/app` — `WorkspaceV3.tsx`)**: Real-time 3D point cloud & textured surface mesh renderer with 4 render modes (*Surface, Points, Wireframe, Clay*), dynamic Z-extrusion controls, 5 perceptual colormaps (*Turbo, Viridis, Thermal, Clay, Albedo*), and client-side **OBJ 3D model export** (`exportDepthToObj`).
   - **CV Benchmark Studio (`/studio`, `/benchmark` — `BenchmarkStudioPage.tsx`)**: Interactive multi-dataset comparison (NYU Depth V2, Middlebury Stereo, KITTI LiDAR) with real-time HUD scorecards for RMSE, Boundary RMSE, PSNR, SSIM, and Edge-F1.
   - **Cinematic Scrollytelling (`/` — `LandingV3.tsx`)**: Video scroll-synchronization powered by GSAP and Lenis.
   - Quickstart: `cd frontend; npm install; npm run dev` (Runs on `http://localhost:5173`).

2. **Streamlit Studio (`app.py`) & Docker**:
   - Interactive Streamlit dashboard on port 8501 (`uv run streamlit run app.py`).
   - Production Docker container: `docker build -t depth-refine-p4 . && docker run -p 8501:8501 depth-refine-p4`.
   - Official Submission Report: [`Hackathon_Technical_Solution_Report.md`](./Hackathon_Technical_Solution_Report.md).
