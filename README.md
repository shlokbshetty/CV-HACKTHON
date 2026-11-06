 # Edge-Preserving Depth Map Upsampling and Refinement (ProbID - P4)

An intelligent, edge-aware depth reconstruction and super-resolution framework built for **Problem Statement 4 (ProbID - P4)**. Transforms sparse, noisy, or low-resolution depth maps ($2\times, 4\times, 8\times$) into dense, high-resolution depth representations while sharply preserving geometric object boundaries and suppressing RGB texture-copying artifacts.

---

## Key Architectural Innovations

1. **EG-DepthNet (Edge-Gated Cross-Guidance Network)**:
   - **Sparsity-Aware Partial Convolution Stem**: Normalizes activations strictly over valid sensor pixels so missing Kinect shadows or sparse LiDAR holes never corrupt feature maps.
   - **Decoupled Dual-Stream Pyramid**: Separates 3D geometric surface gradients ($\nabla D$) from 2D RGB appearance features ($\nabla I$).
   - **Texture-Copy Suppressing Edge-Gated Cross-Attention (EGCA)**: Computes a learned per-pixel **Edge Coherence Gate** $G_{edge} \in [0, 1]$. On flat walls/floors with high-contrast RGB patterns (checkerboards, posters, wood grain), $G_{edge} \to 0$ blocks texture bleeding into depth. At true 3D occlusion boundaries, $G_{edge} \to 1$ locks the transition to sub-pixel RGB edges.
   - **Anisotropic Spatial Propagation Network (ASPN)**: Iterative 8-neighbor convex affinity propagation eliminates "flying pixels" across step discontinuities.
2. **3-Stage Hybrid Depth Reconstruction Pipeline**:
   - **Stage 1 (Classical Pre-Conditioning)**: Statistical outlier spike removal + Multi-Scale Cross-Bilateral hole filling (`fill_depth_cross_bf` equivalent).
   - **Stage 2 (Deep Prior & Aleatoric Uncertainty)**: Predicts high-resolution depth $\hat{D}_{deep}$, learned edge gate $G_{edge}$, and uncertainty $\hat{\sigma}_D$.
   - **Stage 3 (Uncertainty-Guided Bilateral Energy & Sensor Back-Projection)**: Anchors interior smooth regions to low-res sensor measurements, snaps step edges via trilateral filtering, and polishes planar surfaces with texture-suppressed Guided Filtering.

---

## Supported Datasets (NYUv2, Middlebury 2001–2021, KITTI)

| Dataset | Editions / Subsets Supported | Size | Role |
| :--- | :--- | :---: | :--- |
| **NYU Depth V2** | Official Labeled `.mat` (`nyu_depth_v2_labeled.mat`: 1,449 aligned RGB + inpainted GT + raw Kinect depth with holes) | ~2.8 GB | Primary Multi-Scale Training ($2\times, 4\times, 8\times$) & Indoor Benchmark |
| **Middlebury Stereo** | **2001** (`Venus`, `Sawtooth`, `Bull`), **2003** (`Cones`, `Teddy`), **2005** (`Art`, `Books`, `Dolls`, `Laundry`, `Moebius`, `Reindeer`), **2006** (`Aloe`, `Bowling1`, `Flowerpots`), **2014** (`Adirondack`, `Jadeplant` `.pfm` sub-pixel GT), **2021 Mobile** (`artroom1` `.pfm`) | ~150 MB | Zero-Shot Sub-Pixel Discontinuity & Boundary Benchmark |
| **KITTI Vision Suite** | Depth Completion Validation Selection (Sparse 64-beam Velodyne LiDAR + Semi-Dense GT + RGB) | ~650 MB | Zero-Shot Outdoor Sparse-to-Dense Benchmark |

---

## Teammate's RTX 5060 (8GB VRAM, 24GB RAM) Training Guide

Your friend's laptop (**RTX 5060 8GB VRAM + 24GB System RAM**) can train the full multi-scale model at maximum speed using native Blackwell `bfloat16` Mixed Precision and full in-RAM dataset caching:

### Step 1: Install Prerequisites on the RTX 5060 Laptop
1. Ensure **NVIDIA Driver >= 572.xx** is installed (`nvidia-smi`).
2. Install **uv** (fast Python environment manager):
   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

### Step 2: Clone Repo & Set Up Environment
```powershell
git clone <YOUR_GITHUB_REPO_URL>
cd CV-HACKTHON
uv venv --python 3.12
uv pip install --extra-index-url https://download.pytorch.org/whl/cu128 -e .
```
*(Note: Use `cu128` on RTX 5060 Blackwell GPUs, or `cu124` on RTX 40-series GPUs).*

### Step 3: Download the Datasets (~3.0 GB total)
```powershell
# Downloads Middlebury (2005/2006/2014/2021), KITTI validation cache, and full 2.8GB NYU Depth V2 .mat:
uv run python scripts/download_datasets.py --dataset all --full-nyuv2-mat
```

### Step 4: Train the SOTA Model on RTX 5060
```powershell
uv run python scripts/train.py --config configs/rtx5060_nyuv2.yaml
```
- Automatically caches NYUv2 in the **24GB System RAM** for zero disk I/O wait.
- Uses `bfloat16` AMP across $2\times, 4\times, 8\times$ scales with **Exponential Moving Average (EMA)** weights.
- Saves the best weights to `checkpoints/eg_depthnet_best.pt`.

### Step 5: Run Full Evaluation & Launch the Interactive Web App
```powershell
# Generate Markdown & JSON Benchmark Reports across 2x, 4x, 8x:
uv run python scripts/evaluate.py --datasets nyuv2 middlebury kitti --scales 2 4 8

# Launch Interactive Streamlit Dashboard (2D/3D Point Cloud + Boundary Inspector):
uv run streamlit run app.py
```

---

## Benchmark Results (Quantitative Highlights)

Evaluated across **Middlebury (2001–2021)**, **NYU Depth V2**, and **KITTI** on an **NVIDIA GeForce RTX 5060 Laptop GPU**:

| Dataset | Scale | Method | RMSE (m) ↓ | Boundary RMSE (m) ↓ | PSNR (dB) ↑ | SSIM ↑ | Edge F1 ↑ | Latency (ms) ↓ |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **NYU Depth V2** | 2× | Bicubic Baseline | 1.0417 | 1.1705 | 17.64 | 0.8047 | 0.3921 | 3.4 ms |
| | 2× | **Hybrid (Ours)** | **1.0067** *(−3.5 cm)* | **1.1273** *(−4.3 cm)* | **18.11** | **0.8152** | **0.3971** | 38.6 ms |
| | 4× | Bicubic Baseline | 0.7443 | 0.9999 | 21.11 | 0.8525 | 0.3184 | 2.8 ms |
| | 4× | **Hybrid (Ours)** | **0.7282** *(−1.6 cm)* | **0.9743** *(−2.6 cm)* | **21.34** | **0.8650** | **0.3613** | 39.7 ms |
| **Middlebury** | 4× | Bicubic Baseline | 0.4112 | 0.8666 | 23.42 | 0.9092 | 0.3797 | 2.9 ms |
| | 4× | **Hybrid (Ours)** | **0.4020** *(−0.9 cm)* | **0.8530** *(−1.4 cm)* | **23.63** | **0.9176** | **0.4676** | 39.9 ms |
| **KITTI LiDAR** | 2× | Bicubic Baseline | 0.9799 | 0.7805 | 10.68 | 0.7143 | 0.2284 | 4.1 ms |
| | 2× | **Hybrid (Ours)** | **0.9228** *(−5.7 cm)* | **0.7355** *(−4.5 cm)* | **11.21** | **0.7360** | **0.2485** | 42.3 ms |
| | 4× | Bicubic Baseline | 1.0263 | 0.8738 | 10.29 | 0.7286 | 0.2006 | 2.8 ms |
| | 4× | **Hybrid (Ours)** | **1.0114** *(−1.5 cm)* | **0.8499** *(−2.4 cm)* | **10.42** | **0.7550** | **0.2403** | 39.2 ms |

Full multi-dataset report: [outputs/benchmark_report.md](outputs/benchmark_report.md)  
Comprehensive Technical Hackathon Defense: [`hackathon_technical_report.md`](file:///C:/Users/siddh/.gemini/antigravity/brain/a1f5090e-c099-4ddf-875c-7b35d12ba2ba/hackathon_technical_report.md)

---

## Codebase Structure (`codebase-design` Deep Modules)

```text
CV-HACKTHON/
├── configs/
│   └── rtx5060_nyuv2.yaml            # Tuned training config for RTX 5060 (8GB VRAM, 24GB RAM)
├── src/depth_refine/
│   ├── data/                         # Deep Module 1: Dataset Registry (NYUv2, Middlebury, KITTI) & Sensor Degradation
│   ├── classical/                    # Deep Module 2: Nearest/Bilinear/Bicubic, Telea/Colorization, JBF, Guided Filter
│   ├── models/                       # Deep Module 3: EG-DepthNet (EGCA + ASPN) & GeometryAwareDepthLoss
│   ├── hybrid/                       # Deep Module 4: 3-Stage Hybrid Depth Reconstruction Pipeline
│   └── evaluate/                     # Deep Module 5: RMSE, MAE, PSNR, SSIM, Boundary-RMSE, Edge-F1, Latency, VRAM
├── scripts/
│   ├── download_datasets.py          # Automated resumable dataset downloader
│   ├── train.py                      # Multi-scale AMP + EMA training script
│   └── evaluate.py                   # Quantitative benchmark runner
├── tests/                            # Automated unit & integration test suite
└── app.py                            # Deployed Streamlit application with 3D Point Cloud Viewer
```
