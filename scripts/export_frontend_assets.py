import os
import json
import sys
from pathlib import Path

PROJECT_SRC = Path(__file__).resolve().parent.parent / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

import cv2
import numpy as np
import torch
import matplotlib as mpl

from depth_refine.classical.filters import reconstruct_classical
from depth_refine.data.degradation import DegradationConfig, degrade_depth
from depth_refine.data.dataset_registry import load_dataset, MiddleburyDataset
from depth_refine.evaluate.metrics import compute_depth_metrics
from depth_refine.hybrid.hybrid_pipeline import HybridReconstructor

def colorize_depth(
    depth: np.ndarray,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap_name: str = "turbo",
    invalid_mask: np.ndarray | None = None,
    mode: str = "linear",
    overlay_edges: bool = False,
    rgb_guidance: np.ndarray | None = None,
    near_is_warm: bool = True,
) -> np.ndarray:
    if mode == "disparity":
        d_safe = np.maximum(depth, 0.05)
        disp = 1.0 / d_safe
        if invalid_mask is not None:
            disp_valid = disp[~invalid_mask & (depth > 0.05) & np.isfinite(depth)]
        else:
            disp_valid = disp[(depth > 0.05) & np.isfinite(depth)]
        disp_min = float(np.percentile(disp_valid, 2)) if disp_valid.size > 0 else float(disp.min())
        disp_max = float(np.percentile(disp_valid, 98)) if disp_valid.size > 0 else float(disp.max())
        norm = np.clip((disp - disp_min) / max(disp_max - disp_min, 1e-6), 0.0, 1.0)
        if not near_is_warm:
            norm = 1.0 - norm
    else:  # Linear
        if vmin is None or vmax is None:
            valid = depth[(depth > 0.05) & np.isfinite(depth)]
            if invalid_mask is not None:
                valid = valid[~invalid_mask[(depth > 0.05) & np.isfinite(depth)]]
            v_min = float(np.percentile(valid, 2)) if (valid.size > 0 and vmin is None) else (vmin if vmin is not None else 0.5)
            v_max = float(np.percentile(valid, 98)) if (valid.size > 0 and vmax is None) else (vmax if vmax is not None else 5.0)
        else:
            v_min, v_max = vmin, vmax
        norm = np.clip((depth - v_min) / max(v_max - v_min, 1e-6), 0.0, 1.0)
        if near_is_warm:
            norm = 1.0 - norm

    try:
        cmap = mpl.colormaps[cmap_name]
    except Exception:
        cmap = mpl.cm.get_cmap(cmap_name)

    rgb = (cmap(norm)[..., :3] * 255.0).astype(np.uint8)

    if overlay_edges and rgb_guidance is not None:
        try:
            gray_u8 = (np.clip(rgb_guidance, 0.0, 1.0) * 255.0).astype(np.uint8)
            if gray_u8.ndim == 3:
                gray_u8 = cv2.cvtColor(gray_u8, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray_u8, 60, 160) > 0
            rgb[edges] = np.array([0, 240, 255], dtype=np.uint8)
        except Exception:
            pass

    if invalid_mask is not None:
        rgb[invalid_mask] = np.array([12, 14, 18], dtype=np.uint8)

    return rgb

def main():
    out_dir = Path("frontend/public/assets/benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = Path("checkpoints/eg_depthnet_best.pt")
    engine = HybridReconstructor(checkpoint_path=ckpt if ckpt.exists() else None, device=device)

    deg_cfg = DegradationConfig(
        scale=4,
        hole_ratio=0.36,
        structured_boundary_holes=True,
        noise_std=0.015,
        outlier_prob=0.005,
        seed=42,
    )

    targets = [
        {"key": "nyuv2", "type": "dataset", "name": "nyuv2", "crop": 256, "label": "NYU Depth V2 (Indoor RGB-D)"},
        {"key": "middlebury", "type": "preset", "dir": Path("data/middleburry/data/artroom1"), "label": "Middlebury Stereo (artroom1)"},
        {"key": "artroom1", "type": "preset", "dir": Path("data/middleburry/data/artroom1"), "label": "Art Studio & Easel (artroom1)"},
        {"key": "bandsaw1", "type": "preset", "dir": Path("data/middleburry/data/bandsaw1"), "label": "Bandsaw Industrial Workshop (bandsaw1)"},
        {"key": "chess1", "type": "preset", "dir": Path("data/middleburry/data/chess1"), "label": "Chessboard & Pieces (chess1)"},
        {"key": "ladder1", "type": "preset", "dir": Path("data/middleburry/data/ladder1"), "label": "Ladder & Scaffolding (ladder1)"},
        {"key": "skates1", "type": "preset", "dir": Path("data/middleburry/data/skates1"), "label": "Roller Skates (skates1)"},
        {"key": "kitti", "type": "dataset", "name": "kitti", "crop": 256, "label": "KITTI Vision Suite (Outdoor LiDAR)"},
        {"key": "synthetic", "type": "dataset", "name": "synthetic", "crop": 256, "label": "3D Texture-Copy Stress Scene"},
    ]

    manifest = {}

    for item in targets:
        k = item["key"]
        print(f"Processing target: {k} ({item['label']})...")
        rgb_hr = None
        depth_gt = None
        valid_gt = None
        boundary_gt = None

        if item["type"] == "dataset":
            try:
                ds = load_dataset(name=item["name"], split="test", degradation=deg_cfg, crop_size=item["crop"], max_samples=4)
                sample = ds[0]
                depth_lr = sample["depth_lr"].to(device)
                valid_lr = sample["valid_mask_lr"].to(device)
                rgb_hr = sample["rgb_hr"].to(device)
                depth_gt = sample["depth_gt"].to(device)
                valid_gt = sample["valid_mask_gt"].to(device)
                boundary_gt = sample["boundary_mask_gt"].to(device)
            except Exception as e:
                print(f"  Error loading dataset {item['name']}: {e}, falling back to synthetic")
                ds = load_dataset(name="synthetic", split="test", degradation=deg_cfg, crop_size=256, max_samples=4)
                sample = ds[0]
                depth_lr = sample["depth_lr"].to(device)
                valid_lr = sample["valid_mask_lr"].to(device)
                rgb_hr = sample["rgb_hr"].to(device)
                depth_gt = sample["depth_gt"].to(device)
                valid_gt = sample["valid_mask_gt"].to(device)
                boundary_gt = sample["boundary_mask_gt"].to(device)
        else: # preset
            p_dir = item["dir"]
            rgb_path = next(p_dir.glob("im*.png"), None) or next(p_dir.glob("view*.png"), None)
            disp_path = next(p_dir.glob("disp*.pfm"), None) or next(p_dir.glob("disp*.png"), None)
            bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            rgb_np_orig = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

            if disp_path.suffix == ".pfm":
                disp_np = MiddleburyDataset(scene_dirs=[])._read_disp(disp_path)
                disp_valid = np.nan_to_num(disp_np, nan=0.0, posinf=0.0, neginf=0.0)
                depth_np_orig = 100.0 / np.maximum(disp_valid, 5.0)
                depth_np_orig = np.clip(depth_np_orig, 0.5, 6.0).astype(np.float32)
            else:
                gray = cv2.imread(str(disp_path), cv2.IMREAD_UNCHANGED)
                if gray.ndim == 3:
                    gray = gray[:, :, 0]
                disp_valid = np.nan_to_num(gray.astype(np.float32), nan=0.0)
                depth_np_orig = 100.0 / np.maximum(disp_valid, 5.0)
                depth_np_orig = np.clip(depth_np_orig, 0.5, 6.0).astype(np.float32)

            h, w = rgb_np_orig.shape[:2]
            cdim = min(h, w)
            cy, cx = h // 2, w // 2
            rgb_cr = rgb_np_orig[cy - cdim//2:cy + cdim//2, cx - cdim//2:cx + cdim//2]
            d_cr = depth_np_orig[cy - cdim//2:cy + cdim//2, cx - cdim//2:cx + cdim//2]
            rgb_256 = cv2.resize(rgb_cr, (256, 256))
            d_256 = cv2.resize(d_cr, (256, 256), interpolation=cv2.INTER_NEAREST)

            rgb_t = torch.from_numpy(rgb_256.transpose(2, 0, 1)).float().to(device)
            depth_t = torch.from_numpy(d_256[None, ...]).float().to(device)
            degraded = degrade_depth(depth_t, rgb_t, deg_cfg)

            depth_lr = degraded.depth_lr.to(device)
            valid_lr = degraded.valid_mask_lr.to(device)
            rgb_hr = degraded.rgb_hr.to(device)
            depth_gt = degraded.depth_gt.to(device)
            valid_gt = degraded.valid_gt.to(device) if hasattr(degraded, "valid_gt") else degraded.valid_mask_gt.to(device)
            boundary_gt = degraded.boundary_gt.to(device) if hasattr(degraded, "boundary_gt") else degraded.boundary_mask_gt.to(device)

        preds = {}
        for m in ["nearest", "guided", "jbf", "mask_bicubic", "bicubic"]:
            out_t = reconstruct_classical(depth_lr, rgb_hr, scale=4, method=m, valid_mask_lr=valid_lr)
            preds[m] = out_t[0].detach().cpu().numpy()

        res_obj = engine.reconstruct(depth_lr, rgb_hr, scale=4, valid_mask_lr=valid_lr, precision="bf16", use_tta=False)
        preds["deep"] = res_obj.depth_deep[0].detach().cpu().numpy()
        preds["hybrid"] = res_obj.depth_hybrid[0].detach().cpu().numpy()

        edge_gate_np = res_obj.edge_gate[0].detach().cpu().numpy()
        uncertainty_np = res_obj.uncertainty[0].detach().cpu().numpy()
        depth_prior_np = (
            res_obj.depth_prior[0].detach().cpu().numpy()
            if res_obj.depth_prior is not None
            else preds["hybrid"]
        )
        feature_vis_np = (
            res_obj.feature_vis[0].detach().cpu().numpy()
            if res_obj.feature_vis is not None
            else edge_gate_np
        )

        rgb_np = rgb_hr.detach().cpu().numpy().transpose(1, 2, 0)
        gt_np = depth_gt[0].detach().cpu().numpy()
        valid_gt_np = valid_gt[0].detach().cpu().numpy() > 0.5
        lr_np = depth_lr[0].detach().cpu().numpy()
        lr_valid_np = valid_lr[0].detach().cpu().numpy() > 0.5

        valid_vals = gt_np[valid_gt_np]
        vmin = float(np.percentile(valid_vals, 2)) if valid_vals.size > 0 else float(np.percentile(gt_np, 2))
        vmax = float(np.percentile(valid_vals, 98)) if valid_vals.size > 0 else float(np.percentile(gt_np, 98))

        # Save Images
        # 1. RGB
        cv2.imwrite(str(out_dir / f"{k}_rgb.png"), cv2.cvtColor((rgb_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

        # 2. Prior
        cv2.imwrite(str(out_dir / f"{k}_prior.png"), cv2.cvtColor(colorize_depth(depth_prior_np, vmin, vmax, cmap_name="turbo"), cv2.COLOR_RGB2BGR))

        # 3. LR
        lr_vis = cv2.resize(
            colorize_depth(lr_np, vmin, vmax, cmap_name="turbo", invalid_mask=~lr_valid_np),
            (gt_np.shape[1], gt_np.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        cv2.imwrite(str(out_dir / f"{k}_lr.png"), cv2.cvtColor(lr_vis, cv2.COLOR_RGB2BGR))

        # 4. GT
        cv2.imwrite(str(out_dir / f"{k}_gt.png"), cv2.cvtColor(colorize_depth(gt_np, vmin, vmax, cmap_name="turbo", invalid_mask=~valid_gt_np), cv2.COLOR_RGB2BGR))

        # 5. Hybrid
        cv2.imwrite(str(out_dir / f"{k}_hybrid.png"), cv2.cvtColor(colorize_depth(preds["hybrid"], vmin, vmax, cmap_name="turbo"), cv2.COLOR_RGB2BGR))

        # Panel 2 Images
        cv2.imwrite(str(out_dir / f"{k}_jbf.png"), cv2.cvtColor(colorize_depth(preds["jbf"], vmin, vmax, cmap_name="turbo"), cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_dir / f"{k}_feat.png"), cv2.cvtColor(colorize_depth(feature_vis_np, None, None, cmap_name="viridis", near_is_warm=False), cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_dir / f"{k}_mask_bicubic.png"), cv2.cvtColor(colorize_depth(preds["mask_bicubic"], vmin, vmax, cmap_name="turbo"), cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_dir / f"{k}_bicubic.png"), cv2.cvtColor(colorize_depth(preds["bicubic"], vmin, vmax, cmap_name="turbo"), cv2.COLOR_RGB2BGR))

        # Panel 3 Comparisons
        cv2.imwrite(str(out_dir / f"{k}_comp_nearest.png"), cv2.cvtColor(colorize_depth(preds["nearest"], vmin, vmax, cmap_name="turbo"), cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_dir / f"{k}_comp_guided.png"), cv2.cvtColor(colorize_depth(preds["guided"], vmin, vmax, cmap_name="turbo"), cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_dir / f"{k}_comp_mask_bicubic.png"), cv2.cvtColor(colorize_depth(preds["mask_bicubic"], vmin, vmax, cmap_name="turbo"), cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_dir / f"{k}_comp_deep.png"), cv2.cvtColor(colorize_depth(preds["deep"], vmin, vmax, cmap_name="turbo"), cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_dir / f"{k}_comp_hybrid.png"), cv2.cvtColor(colorize_depth(preds["hybrid"], vmin, vmax, cmap_name="turbo"), cv2.COLOR_RGB2BGR))

        # Panel 4 Explainability
        cv2.imwrite(str(out_dir / f"{k}_edge_gate.png"), cv2.cvtColor(colorize_depth(edge_gate_np, 0.0, 1.0, cmap_name="magma", near_is_warm=False), cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_dir / f"{k}_uncertainty.png"), cv2.cvtColor(colorize_depth(uncertainty_np, None, None, cmap_name="inferno", near_is_warm=False), cv2.COLOR_RGB2BGR))
        err_bic = np.abs(preds["bicubic"] - gt_np)
        err_hyb = np.abs(preds["hybrid"] - gt_np)
        gain_map = err_bic - err_hyb
        cv2.imwrite(str(out_dir / f"{k}_gain_map.png"), cv2.cvtColor(colorize_depth(gain_map, -0.15, 0.40, cmap_name="coolwarm", near_is_warm=False), cv2.COLOR_RGB2BGR))

        # Tab 2 Zoom Patches
        H, W = gt_np.shape
        crop_y, crop_x = H // 2, W // 2
        r_crop = 36
        y0, y1 = max(0, crop_y - r_crop), min(H, crop_y + r_crop)
        x0, x1 = max(0, crop_x - r_crop), min(W, crop_x + r_crop)

        patch_rgb = cv2.resize((rgb_np[y0:y1, x0:x1] * 255).astype(np.uint8), (220, 220), interpolation=cv2.INTER_NEAREST)
        patch_nr = cv2.resize(colorize_depth(preds["nearest"][y0:y1, x0:x1], vmin, vmax, cmap_name="turbo"), (220, 220), interpolation=cv2.INTER_NEAREST)
        patch_bic = cv2.resize(colorize_depth(preds["bicubic"][y0:y1, x0:x1], vmin, vmax, cmap_name="turbo"), (220, 220), interpolation=cv2.INTER_NEAREST)
        patch_hyb = cv2.resize(colorize_depth(preds["hybrid"][y0:y1, x0:x1], vmin, vmax, cmap_name="turbo"), (220, 220), interpolation=cv2.INTER_NEAREST)
        patch_gt = cv2.resize(colorize_depth(gt_np[y0:y1, x0:x1], vmin, vmax, cmap_name="turbo"), (220, 220), interpolation=cv2.INTER_NEAREST)

        cv2.imwrite(str(out_dir / f"{k}_patch_rgb.png"), cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_dir / f"{k}_patch_nearest.png"), cv2.cvtColor(patch_nr, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_dir / f"{k}_patch_bicubic.png"), cv2.cvtColor(patch_bic, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_dir / f"{k}_patch_hybrid.png"), cv2.cvtColor(patch_hyb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_dir / f"{k}_patch_gt.png"), cv2.cvtColor(patch_gt, cv2.COLOR_RGB2BGR))

        # Tab 2 Error Heatmaps
        err_max = max(0.35, float(np.percentile(np.abs(preds["bicubic"] - gt_np), 97)))
        for em in ["bicubic", "guided", "mask_bicubic", "hybrid"]:
            err_m = np.abs(preds[em] - gt_np)
            cv2.imwrite(
                str(out_dir / f"{k}_err_{em}.png"),
                cv2.cvtColor(colorize_depth(err_m, 0.0, err_max, cmap_name="magma", near_is_warm=False), cv2.COLOR_RGB2BGR),
            )

        # 1D Depth Profile array
        profile_data = {
            "crop_y": crop_y,
            "gt": gt_np[crop_y, :].tolist(),
            "bicubic": preds["bicubic"][crop_y, :].tolist(),
            "jbf": preds["jbf"][crop_y, :].tolist(),
            "hybrid": preds["hybrid"][crop_y, :].tolist(),
        }
        with open(out_dir / f"{k}_profile.json", "w", encoding="utf-8") as f:
            json.dump(profile_data, f)

        # Compute real metrics
        m_hyb = compute_depth_metrics(torch.from_numpy(preds["hybrid"][None, None, ...]).float().to(device), depth_gt, valid_gt, boundary_gt, rgb_hr)
        m_bic = compute_depth_metrics(torch.from_numpy(preds["bicubic"][None, None, ...]).float().to(device), depth_gt, valid_gt, boundary_gt, rgb_hr)
        m_mb = compute_depth_metrics(torch.from_numpy(preds["mask_bicubic"][None, None, ...]).float().to(device), depth_gt, valid_gt, boundary_gt, rgb_hr)
        m_jbf = compute_depth_metrics(torch.from_numpy(preds["jbf"][None, None, ...]).float().to(device), depth_gt, valid_gt, boundary_gt, rgb_hr)
        m_guided = compute_depth_metrics(torch.from_numpy(preds["guided"][None, None, ...]).float().to(device), depth_gt, valid_gt, boundary_gt, rgb_hr)
        m_nr = compute_depth_metrics(torch.from_numpy(preds["nearest"][None, None, ...]).float().to(device), depth_gt, valid_gt, boundary_gt, rgb_hr)
        m_deep = compute_depth_metrics(torch.from_numpy(preds["deep"][None, None, ...]).float().to(device), depth_gt, valid_gt, boundary_gt, rgb_hr)

        manifest[k] = {
            "label": item["label"],
            "metrics": {
                "hybrid": {k_m: float(v_m) for k_m, v_m in m_hyb.items()},
                "bicubic": {k_m: float(v_m) for k_m, v_m in m_bic.items()},
                "mask_bicubic": {k_m: float(v_m) for k_m, v_m in m_mb.items()},
                "jbf": {k_m: float(v_m) for k_m, v_m in m_jbf.items()},
                "guided": {k_m: float(v_m) for k_m, v_m in m_guided.items()},
                "nearest": {k_m: float(v_m) for k_m, v_m in m_nr.items()},
                "deep": {k_m: float(v_m) for k_m, v_m in m_deep.items()},
            }
        }
        print(f"  Done {k}: Hybrid RMSE = {m_hyb['RMSE']:.4f}m, Bicubic RMSE = {m_bic['RMSE']:.4f}m")

    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print("All assets generated successfully!")

if __name__ == "__main__":
    main()
