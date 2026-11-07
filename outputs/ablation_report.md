# Controlled Ablation Study (MIDDLEBURY — 4× Upsampling)

| ID | Configuration | RMSE (m) ↓ | Boundary RMSE (m) ↓ | PSNR (dB) ↑ | SSIM ↑ | Edge-F1 ↑ | Texture-Copy Err ↓ |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| A1 | Baseline: Naive Bicubic (No Hole Pre-Fill) | 0.8934 | 1.8010 | 17.29 | 0.7635 | 0.0167 | 0.0237 |
| A2 | + Multi-Scale Pyramid Hole Fill & Outlier Filter | 0.7829 | 1.7895 | 19.03 | 0.8257 | 0.2013 | 0.0069 |
| A3 | + Classical Fast Guided Filter (He et al.) | 0.7733 | 1.7848 | 19.43 | 0.8251 | 0.3142 | 0.0202 |
| A4 | + Classical Joint Bilateral Filter (JBF) | 0.7831 | 1.7857 | 19.04 | 0.8272 | 0.3360 | 0.0112 |
| A5 | + Deep Model w/o Edge-Coherence Gate (Ungated RGB) | 0.7781 | 1.7856 | 19.23 | 0.8253 | 0.2837 | 0.0186 |
| A6 | + EG-DepthNet (Edge-Coherence Gate + ASPN) | 0.7797 | 1.8048 | 19.46 | 0.8296 | 0.5095 | 0.0102 |
| A7 | **+ Hybrid Stage-3 (Sensor Back-Proj + Mode Energy)** | **0.7791** | **1.8040** | **19.48** | **0.8300** | **0.4797** | **0.0103** |
| A8 | **+ Hybrid INT8 Weight Quantized (3.96x Compressed)** | **0.7791** | **1.8037** | **19.48** | **0.8300** | **0.4752** | **0.0102** |
