# Edge-Preserving Depth Upsampling & Refinement Benchmark Report (ProbID: P4)

**Repository**: [https://github.com/shlokbshetty/Depth-lab](https://github.com/shlokbshetty/Depth-lab)  
**Evaluation Protocol**: Multi-scale ($2\times, 4\times, 8\times$) evaluation across NYU Depth V2 (held-out indoor test split), Middlebury Stereo (zero-shot sub-pixel discontinuities), and KITTI Vision Suite (zero-shot outdoor sparse LiDAR).

---

## 1. Full Multi-Scale Quantitative Benchmark Results

| Dataset | Scale | Method | RMSE (m) ↓ | MAE (m) ↓ | PSNR (dB) ↑ | SSIM ↑ | Boundary RMSE ↓ | Edge F1 ↑ | Tex-Copy Err ↓ | Latency (ms) ↓ | Memory (MB) ↓ |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| NYUV2 | 2× | Nearest-Neighbour | 0.1454 | 0.0517 | 35.47 | 0.9569 | 0.3677 | 0.4190 | 0.0185 | 143.56 | 29.6 |
| NYUV2 | 2× | Bilinear Interp. | 0.1432 | 0.0478 | 35.67 | 0.9621 | 0.3642 | 0.4148 | 0.0158 | 5.77 | 29.7 |
| NYUV2 | 2× | Bicubic Interp. | 0.1428 | 0.0485 | 35.69 | 0.9613 | 0.3631 | 0.4192 | 0.0175 | 6.45 | 29.7 |
| NYUV2 | 2× | Guided Filter (RGB) | 0.1441 | 0.0555 | 35.02 | 0.9596 | 0.3312 | 0.5300 | 0.0305 | 51.30 | 29.8 |
| NYUV2 | 2× | Joint Bilateral (JBF) | 0.1214 | 0.0389 | 37.21 | 0.9734 | 0.3088 | 0.6970 | 0.0153 | 12.22 | 80.9 |
| NYUV2 | 2× | Mask-Bicubic (Stage 1) | 0.1216 | 0.0392 | 37.16 | 0.9727 | 0.3096 | 0.7078 | 0.0154 | 30.15 | 102.6 |
| NYUV2 | 2× | EG-DepthNet (Deep BF16) | 0.0896 | 0.0317 | 40.39 | 0.9803 | 0.2184 | 0.8536 | 0.0092 | 429.97 | 549.1 |
| NYUV2 | 2× | Hybrid INT8 Quantized (4x Comp.) | 0.0910 | 0.0309 | 39.87 | 0.9805 | 0.2272 | 0.8292 | 0.0114 | 103.99 | 473.6 |
| NYUV2 | 2× | **Self-Distilled Hybrid (Stage 2)** | **0.0912** | **0.0309** | **39.86** | **0.9805** | **0.2276** | **0.8288** | **0.0114** | 76.16 | 549.1 |
| NYUV2 | 2× | **Hybrid + Flip-TTA (Self-Ensemble)** | **0.1112** | **0.0365** | **37.61** | **0.9756** | **0.2858** | **0.7490** | **0.0129** | 144.54 | 551.7 |
| NYUV2 | 4× | Nearest-Neighbour | 0.2565 | 0.1087 | 30.35 | 0.9225 | 0.5605 | 0.3798 | 0.0197 | 7.09 | 13.8 |
| NYUV2 | 4× | Bilinear Interp. | 0.2533 | 0.1047 | 30.52 | 0.9346 | 0.5554 | 0.1855 | 0.0169 | 7.22 | 13.8 |
| NYUV2 | 4× | Bicubic Interp. | 0.2534 | 0.1046 | 30.51 | 0.9339 | 0.5567 | 0.2090 | 0.0180 | 7.40 | 13.8 |
| NYUV2 | 4× | Guided Filter (RGB) | 0.2624 | 0.1123 | 29.87 | 0.9316 | 0.5437 | 0.3526 | 0.0362 | 14.24 | 14.0 |
| NYUV2 | 4× | Joint Bilateral (JBF) | 0.2499 | 0.1014 | 30.60 | 0.9382 | 0.5414 | 0.4396 | 0.0219 | 13.26 | 80.5 |
| NYUV2 | 4× | Mask-Bicubic (Stage 1) | 0.2490 | 0.1014 | 30.64 | 0.9383 | 0.5379 | 0.4586 | 0.0227 | 28.46 | 297.9 |
| NYUV2 | 4× | EG-DepthNet (Deep BF16) | 0.2123 | 0.0871 | 32.10 | 0.9504 | 0.4503 | 0.6262 | 0.0131 | 116.89 | 548.6 |
| NYUV2 | 4× | Hybrid INT8 Quantized (4x Comp.) | 0.2097 | 0.0864 | 32.37 | 0.9489 | 0.4506 | 0.5934 | 0.0212 | 113.23 | 473.0 |
| NYUV2 | 4× | **Self-Distilled Hybrid (Stage 2)** | **0.2095** | **0.0862** | **32.37** | **0.9490** | **0.4503** | **0.5932** | **0.0212** | 111.34 | 548.6 |
| NYUV2 | 4× | **Hybrid + Flip-TTA (Self-Ensemble)** | **0.2227** | **0.0910** | **31.62** | **0.9461** | **0.4800** | **0.5385** | **0.0227** | 181.45 | 551.1 |
| NYUV2 | 8× | Nearest-Neighbour | 0.6071 | 0.3181 | 22.17 | 0.8851 | 0.8792 | 0.2021 | 0.0171 | 10.68 | 9.9 |
| NYUV2 | 8× | Bilinear Interp. | 0.6043 | 0.3147 | 22.23 | 0.9074 | 0.8761 | 0.0346 | 0.0143 | 7.62 | 9.9 |
| NYUV2 | 8× | Bicubic Interp. | 0.6058 | 0.3147 | 22.19 | 0.9065 | 0.8805 | 0.0596 | 0.0149 | 7.85 | 9.9 |
| NYUV2 | 8× | Guided Filter (RGB) | 0.6451 | 0.3454 | 21.92 | 0.8927 | 0.8931 | 0.1417 | 0.0430 | 18.39 | 14.0 |
| NYUV2 | 8× | Joint Bilateral (JBF) | 0.6407 | 0.3374 | 21.92 | 0.9012 | 0.8911 | 0.1130 | 0.0216 | 14.04 | 80.4 |
| NYUV2 | 8× | Mask-Bicubic (Stage 1) | 0.6402 | 0.3370 | 21.93 | 0.9017 | 0.8898 | 0.1512 | 0.0222 | 37.06 | 297.7 |
| NYUV2 | 8× | EG-DepthNet (Deep BF16) | 0.5923 | 0.3069 | 22.90 | 0.9120 | 0.8037 | 0.3157 | 0.0183 | 131.22 | 548.4 |
| NYUV2 | 8× | Hybrid INT8 Quantized (4x Comp.) | 0.5890 | 0.3047 | 22.98 | 0.9124 | 0.7988 | 0.3087 | 0.0245 | 135.56 | 472.9 |
| NYUV2 | 8× | **Self-Distilled Hybrid (Stage 2)** | **0.5893** | **0.3044** | **22.96** | **0.9125** | **0.7994** | **0.3087** | **0.0245** | 145.20 | 548.4 |
| NYUV2 | 8× | **Hybrid + Flip-TTA (Self-Ensemble)** | **0.5993** | **0.3080** | **22.30** | **0.9110** | **0.8443** | **0.2786** | **0.0248** | 231.11 | 550.9 |
| MIDDLEBURY | 2× | Nearest-Neighbour | 0.5254 | 0.0884 | 22.55 | 0.8794 | 1.5067 | 0.2400 | 0.0118 | 7.36 | 29.6 |
| MIDDLEBURY | 2× | Bilinear Interp. | 0.5252 | 0.0852 | 22.56 | 0.8825 | 1.5067 | 0.2713 | 0.0090 | 6.93 | 29.7 |
| MIDDLEBURY | 2× | Bicubic Interp. | 0.5253 | 0.0871 | 22.56 | 0.8810 | 1.5067 | 0.2698 | 0.0109 | 7.83 | 29.7 |
| MIDDLEBURY | 2× | Guided Filter (RGB) | 0.5259 | 0.0827 | 22.55 | 0.8831 | 1.5063 | 0.3719 | 0.0194 | 14.38 | 29.8 |
| MIDDLEBURY | 2× | Joint Bilateral (JBF) | 0.5249 | 0.0812 | 22.56 | 0.8858 | 1.5062 | 0.4790 | 0.0100 | 13.24 | 80.9 |
| MIDDLEBURY | 2× | Mask-Bicubic (Stage 1) | 0.5249 | 0.0812 | 22.56 | 0.8857 | 1.5061 | 0.4741 | 0.0100 | 25.54 | 102.6 |
| MIDDLEBURY | 2× | EG-DepthNet (Deep BF16) | 0.5257 | 0.0775 | 22.56 | 0.8879 | 1.5069 | 0.6022 | 0.0071 | 87.16 | 549.1 |
| MIDDLEBURY | 2× | Hybrid INT8 Quantized (4x Comp.) | 0.5257 | 0.0780 | 22.56 | 0.8875 | 1.5068 | 0.5606 | 0.0093 | 83.06 | 473.6 |
| MIDDLEBURY | 2× | **Self-Distilled Hybrid (Stage 2)** | **0.5256** | **0.0780** | **22.56** | **0.8875** | **1.5067** | **0.5626** | **0.0093** | 82.78 | 549.1 |
| MIDDLEBURY | 2× | **Hybrid + Flip-TTA (Self-Ensemble)** | **0.5256** | **0.0796** | **22.56** | **0.8860** | **1.5075** | **0.4705** | **0.0087** | 129.60 | 551.7 |
| MIDDLEBURY | 4× | Nearest-Neighbour | 0.5288 | 0.0986 | 22.51 | 0.8776 | 1.5108 | 0.2113 | 0.0071 | 7.76 | 13.8 |
| MIDDLEBURY | 4× | Bilinear Interp. | 0.5286 | 0.0962 | 22.52 | 0.8805 | 1.5108 | 0.1548 | 0.0055 | 8.65 | 13.8 |
| MIDDLEBURY | 4× | Bicubic Interp. | 0.5287 | 0.0974 | 22.52 | 0.8795 | 1.5107 | 0.1567 | 0.0065 | 7.51 | 13.8 |
| MIDDLEBURY | 4× | Guided Filter (RGB) | 0.5293 | 0.0926 | 22.51 | 0.8806 | 1.5101 | 0.2991 | 0.0200 | 15.29 | 14.0 |
| MIDDLEBURY | 4× | Joint Bilateral (JBF) | 0.5279 | 0.0911 | 22.53 | 0.8832 | 1.5100 | 0.3557 | 0.0080 | 15.08 | 80.5 |
| MIDDLEBURY | 4× | Mask-Bicubic (Stage 1) | 0.5280 | 0.0900 | 22.53 | 0.8838 | 1.5102 | 0.3556 | 0.0076 | 33.64 | 297.9 |
| MIDDLEBURY | 4× | EG-DepthNet (Deep BF16) | 0.5303 | 0.0855 | 22.50 | 0.8851 | 1.5162 | 0.5129 | 0.0073 | 137.55 | 548.6 |
| MIDDLEBURY | 4× | Hybrid INT8 Quantized (4x Comp.) | 0.5298 | 0.0864 | 22.51 | 0.8853 | 1.5149 | 0.4707 | 0.0093 | 134.73 | 473.0 |
| MIDDLEBURY | 4× | **Self-Distilled Hybrid (Stage 2)** | **0.5297** | **0.0864** | **22.51** | **0.8853** | **1.5147** | **0.4712** | **0.0093** | 131.60 | 548.6 |
| MIDDLEBURY | 4× | **Hybrid + Flip-TTA (Self-Ensemble)** | **0.5297** | **0.0879** | **22.50** | **0.8846** | **1.5150** | **0.4382** | **0.0086** | 208.79 | 551.1 |
| MIDDLEBURY | 8× | Nearest-Neighbour | 0.5413 | 0.1478 | 22.36 | 0.8701 | 1.5004 | 0.0493 | 0.0033 | 8.66 | 9.9 |
| MIDDLEBURY | 8× | Bilinear Interp. | 0.5412 | 0.1461 | 22.36 | 0.8720 | 1.5005 | 0.0047 | 0.0025 | 7.63 | 9.9 |
| MIDDLEBURY | 8× | Bicubic Interp. | 0.5413 | 0.1470 | 22.36 | 0.8717 | 1.5005 | 0.0055 | 0.0028 | 12.16 | 9.9 |
| MIDDLEBURY | 8× | Guided Filter (RGB) | 0.5415 | 0.1471 | 22.35 | 0.8676 | 1.4962 | 0.1027 | 0.0217 | 19.66 | 14.0 |
| MIDDLEBURY | 8× | Joint Bilateral (JBF) | 0.5396 | 0.1458 | 22.38 | 0.8720 | 1.4967 | 0.0564 | 0.0038 | 12.68 | 80.4 |
| MIDDLEBURY | 8× | Mask-Bicubic (Stage 1) | 0.5396 | 0.1455 | 22.38 | 0.8721 | 1.4967 | 0.0657 | 0.0037 | 38.64 | 297.7 |
| MIDDLEBURY | 8× | EG-DepthNet (Deep BF16) | 0.5370 | 0.1367 | 22.42 | 0.8745 | 1.5022 | 0.1479 | 0.0046 | 124.72 | 548.4 |
| MIDDLEBURY | 8× | Hybrid INT8 Quantized (4x Comp.) | 0.5364 | 0.1366 | 22.42 | 0.8750 | 1.5007 | 0.1876 | 0.0051 | 111.74 | 472.9 |
| MIDDLEBURY | 8× | **Self-Distilled Hybrid (Stage 2)** | **0.5364** | **0.1367** | **22.42** | **0.8750** | **1.5006** | **0.1856** | **0.0051** | 112.14 | 548.4 |
| MIDDLEBURY | 8× | **Hybrid + Flip-TTA (Self-Ensemble)** | **0.5366** | **0.1369** | **22.42** | **0.8749** | **1.5006** | **0.1799** | **0.0049** | 182.20 | 550.9 |
| KITTI | 2× | Nearest-Neighbour | 1.2560 | 0.3935 | 18.03 | 0.8913 | 2.4419 | 0.3921 | 0.0157 | 6.72 | 29.6 |
| KITTI | 2× | Bilinear Interp. | 1.2555 | 0.3897 | 18.04 | 0.8949 | 2.4411 | 0.4162 | 0.0132 | 6.34 | 29.7 |
| KITTI | 2× | Bicubic Interp. | 1.2557 | 0.3912 | 18.03 | 0.8943 | 2.4415 | 0.4082 | 0.0147 | 8.16 | 29.7 |
| KITTI | 2× | Guided Filter (RGB) | 1.2565 | 0.3908 | 18.03 | 0.8921 | 2.4421 | 0.4680 | 0.0236 | 14.40 | 29.8 |
| KITTI | 2× | Joint Bilateral (JBF) | 1.2548 | 0.3871 | 18.04 | 0.8939 | 2.4402 | 0.4690 | 0.0136 | 12.65 | 80.9 |
| KITTI | 2× | Mask-Bicubic (Stage 1) | 1.2548 | 0.3874 | 18.04 | 0.8937 | 2.4400 | 0.4735 | 0.0136 | 24.02 | 102.6 |
| KITTI | 2× | EG-DepthNet (Deep BF16) | 1.2292 | 0.3669 | 18.22 | 0.9003 | 2.3907 | 0.5512 | 0.0114 | 81.31 | 549.1 |
| KITTI | 2× | Hybrid INT8 Quantized (4x Comp.) | 1.2314 | 0.3679 | 18.20 | 0.8991 | 2.3950 | 0.5450 | 0.0124 | 74.02 | 473.6 |
| KITTI | 2× | **Self-Distilled Hybrid (Stage 2)** | **1.2295** | **0.3674** | **18.22** | **0.8992** | **2.3913** | **0.5457** | **0.0124** | 74.97 | 549.1 |
| KITTI | 2× | **Hybrid + Flip-TTA (Self-Ensemble)** | **1.2276** | **0.3669** | **18.23** | **0.8984** | **2.3876** | **0.5097** | **0.0134** | 129.02 | 551.7 |
| KITTI | 4× | Nearest-Neighbour | 1.4149 | 0.6287 | 16.99 | 0.8483 | 2.6355 | 0.2075 | 0.0160 | 7.44 | 13.8 |
| KITTI | 4× | Bilinear Interp. | 1.4136 | 0.6252 | 17.00 | 0.8608 | 2.6339 | 0.0757 | 0.0138 | 7.10 | 13.8 |
| KITTI | 4× | Bicubic Interp. | 1.4142 | 0.6266 | 16.99 | 0.8601 | 2.6349 | 0.0889 | 0.0144 | 7.64 | 13.8 |
| KITTI | 4× | Guided Filter (RGB) | 1.5188 | 0.6838 | 16.38 | 0.8493 | 2.8032 | 0.1627 | 0.0267 | 14.76 | 14.0 |
| KITTI | 4× | Joint Bilateral (JBF) | 1.5199 | 0.6843 | 16.38 | 0.8473 | 2.8056 | 0.1213 | 0.0167 | 12.95 | 80.5 |
| KITTI | 4× | Mask-Bicubic (Stage 1) | 1.5194 | 0.6836 | 16.38 | 0.8477 | 2.8046 | 0.1411 | 0.0163 | 29.53 | 297.9 |
| KITTI | 4× | EG-DepthNet (Deep BF16) | 1.4163 | 0.5938 | 16.99 | 0.8552 | 2.6450 | 0.2715 | 0.0185 | 125.95 | 548.6 |
| KITTI | 4× | Hybrid INT8 Quantized (4x Comp.) | 1.4089 | 0.5886 | 17.03 | 0.8559 | 2.6334 | 0.2873 | 0.0211 | 117.65 | 473.0 |
| KITTI | 4× | **Self-Distilled Hybrid (Stage 2)** | **1.4101** | **0.5902** | **17.02** | **0.8557** | **2.6339** | **0.2871** | **0.0211** | 112.47 | 548.6 |
| KITTI | 4× | **Hybrid + Flip-TTA (Self-Ensemble)** | **1.4321** | **0.5994** | **16.89** | **0.8557** | **2.6729** | **0.2772** | **0.0202** | 180.64 | 551.1 |
| KITTI | 8× | Nearest-Neighbour | 3.0316 | 2.0934 | 10.59 | 0.5252 | 4.0080 | 0.3439 | 0.0047 | 8.77 | 9.9 |
| KITTI | 8× | Bilinear Interp. | 3.0312 | 2.0931 | 10.59 | 0.5289 | 4.0074 | 0.4129 | 0.0042 | 7.90 | 9.9 |
| KITTI | 8× | Bicubic Interp. | 3.0317 | 2.0933 | 10.59 | 0.5289 | 4.0084 | 0.4513 | 0.0042 | 7.89 | 9.9 |
| KITTI | 8× | Guided Filter (RGB) | 3.3765 | 2.3327 | 9.43 | 0.4698 | 4.4407 | 0.4179 | 0.0070 | 15.14 | 14.0 |
| KITTI | 8× | Joint Bilateral (JBF) | 3.3777 | 2.3324 | 9.43 | 0.4700 | 4.4417 | 0.4264 | 0.0044 | 13.68 | 80.4 |
| KITTI | 8× | Mask-Bicubic (Stage 1) | 3.3777 | 2.3324 | 9.43 | 0.4701 | 4.4416 | 0.4115 | 0.0042 | 30.88 | 297.7 |
| KITTI | 8× | EG-DepthNet (Deep BF16) | 3.4034 | 2.4021 | 9.36 | 0.4304 | 4.4693 | 0.4269 | 0.0045 | 124.48 | 548.4 |
| KITTI | 8× | Hybrid INT8 Quantized (4x Comp.) | 3.4010 | 2.4004 | 9.37 | 0.4312 | 4.4629 | 0.3271 | 0.0046 | 114.59 | 472.9 |
| KITTI | 8× | **Self-Distilled Hybrid (Stage 2)** | **3.4010** | **2.4004** | **9.37** | **0.4312** | **4.4630** | **0.3278** | **0.0046** | 110.04 | 548.4 |
| KITTI | 8× | **Hybrid + Flip-TTA (Self-Ensemble)** | **3.4005** | **2.3997** | **9.37** | **0.4314** | **4.4622** | **0.3222** | **0.0046** | 175.65 | 550.9 |

---

## 2. Controlled Ablation Study (MIDDLEBURY — 4× Upsampling)

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

---

## 3. Visual & Qualitative Discontinuity Refinement Showcase

- **Training Curves**: [`outputs/training_curves.png`](outputs/training_curves.png)
- **NYU Depth V2**: [`outputs/qualitative_nyuv2.png`](outputs/qualitative_nyuv2.png)
- **Middlebury Stereo**: [`outputs/qualitative_middlebury.png`](outputs/qualitative_middlebury.png)
- **KITTI Vision Suite**: [`outputs/qualitative_kitti.png`](outputs/qualitative_kitti.png)
