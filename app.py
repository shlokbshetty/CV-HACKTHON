"""
Interactive Edge-Preserving Depth Map Upsampling & Refinement Application (ProbID - P4).

Launch with:
    uv run streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import cv2
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch

from depth_refine.classical.filters import compute_spatial_gradients, reconstruct_classical
from depth_refine.data.dataset_registry import load_dataset
from depth_refine.data.degradation import DegradationConfig, degrade_depth
from depth_refine.evaluate.benchmark_runner import ALL_METHODS, METHOD_LABELS
from depth_refine.evaluate.metrics import compute_depth_metrics, measure_runtime_and_memory
from depth_refine.hybrid.hybrid_pipeline import HybridReconstructor

st.set_page_config(
    page_title="Edge-Preserving Depth Upsampling & Refinement (P4)",
    page_icon="📐",
    layout="wide",
)


@st.cache_resource
def get_hybrid_engine() -> HybridReconstructor:
    ckpt = Path("checkpoints/eg_depthnet_best.pt")
    return HybridReconstructor(checkpoint_path=ckpt if ckpt.exists() else None)


def colorize_depth(
    depth: np.ndarray,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap_name: str = "turbo",
    invalid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Convert a 2D depth or error array into an RGB uint8 visualization."""
    if vmin is None:
        valid = depth[depth > 1e-3]
        vmin = float(np.percentile(valid, 2)) if valid.size > 0 else 0.5
    if vmax is None:
        valid = depth[depth > 1e-3]
        vmax = float(np.percentile(valid, 98)) if valid.size > 0 else 6.0
    norm = np.clip((depth - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
    try:
        cmap = mpl.colormaps[cmap_name]
    except Exception:
        cmap = cm.get_cmap(cmap_name)
    rgb = (cmap(norm)[..., :3] * 255.0).astype(np.uint8)
    if invalid_mask is not None:
        rgb[invalid_mask] = np.array([25, 25, 30], dtype=np.uint8)
    return rgb


def build_3d_point_cloud_figure(
    depth: np.ndarray,
    rgb: np.ndarray,
    title: str,
    stride: int = 3,
) -> go.Figure:
    """Create an interactive 3D RGB-colored point cloud in Plotly."""
    H, W = depth.shape
    yy, xx = np.meshgrid(np.arange(0, H, stride), np.arange(0, W, stride), indexing="ij")
    z = depth[::stride, ::stride]
    c = rgb[::stride, ::stride]

    valid = (z > 0.1) & (z < 10.0) & np.isfinite(z)
    u = xx[valid]
    v = yy[valid]
    z_val = z[valid]
    c_val = (np.clip(c[valid], 0.0, 1.0) * 255).astype(int)

    # Back-project (u, v, z) using pinhole camera model
    fx = fy = max(H, W) * 0.9
    cx, cy = W / 2.0, H / 2.0
    X = (u - cx) * z_val / fx
    Y = -(v - cy) * z_val / fy
    Z = -z_val

    colors_str = [f"rgb({r},{g},{b})" for r, g, b in c_val]

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=X,
                y=Z,
                z=Y,
                mode="markers",
                marker=dict(size=2.4, color=colors_str, opacity=0.92),
            )
        ]
    )
    fig.update_layout(
        title=title,
        margin=dict(l=0, r=0, b=0, t=35),
        height=460,
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Depth Z (m)",
            zaxis_title="Y (m)",
            aspectmode="data",
            camera=dict(eye=dict(x=1.25, y=1.35, z=0.65)),
        ),
    )
    return fig


def generate_camera_depth_prior(rgb_np: np.ndarray) -> np.ndarray:
    """
    Generate a realistic geometric depth prior for a standalone RGB camera photo.
    Combines vertical perspective ground slope with luminance & edge cues.
    Range: [0.6m, 5.0m] (standard indoor/room scale).
    """
    H, W = rgb_np.shape[:2]
    y_coords, _ = np.mgrid[0:H, 0:W]
    y_norm = y_coords / float(max(H, 1))
    depth_base = 3.8 - 2.4 * y_norm
    gray = cv2.cvtColor((rgb_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    depth_prior = depth_base + 0.4 * (1.0 - gray)
    return np.clip(depth_prior, 0.6, 6.0).astype(np.float32)


def main() -> None:
    st.title("📐 Edge-Preserving Depth Map Upsampling & Refinement (ProbID - P4)")
    st.markdown(
        "**Architecture**: `EG-DepthNet` (Edge-Gated Cross-Attention + Anisotropic Spatial Propagation) "
        "+ **Stage-3 Uncertainty-Guided Bilateral Energy Refinement**  \n"
        "*Benchmarked on NYU Depth V2, Middlebury Stereo (2001–2021), and KITTI Vision Suite.*"
    )

    engine = get_hybrid_engine()
    device = engine.device

    # --- Sidebar Controls ---
    st.sidebar.header("⚙️ Sensor & Reconstruction Controls")
    dataset_choice = st.sidebar.selectbox(
        "Dataset / Source",
        options=["middlebury", "nyuv2", "kitti", "synthetic", "custom_upload"],
        format_func=lambda x: {
            "middlebury": "Middlebury Stereo (2001–2021)",
            "nyuv2": "NYU Depth V2 (Indoor RGB-D)",
            "kitti": "KITTI Vision Suite (Outdoor LiDAR)",
            "synthetic": "3D Texture-Copy Stress Scene",
            "custom_upload": "Upload Custom RGB + Depth",
        }[x],
    )

    scale = st.sidebar.select_slider("Upsampling Factor", options=[2, 4, 8], value=4)
    hole_ratio = st.sidebar.slider("Missing Depth Ratio (Sparsity)", 0.0, 0.50, 0.10, 0.02)
    structured_holes = st.sidebar.checkbox("Structured Boundary Infrared Shadows", value=True)
    noise_std = st.sidebar.slider("Sensor Axial Noise (m)", 0.0, 0.06, 0.015, 0.005)
    outlier_prob = st.sidebar.slider("Outlier Spike Probability", 0.0, 0.03, 0.005, 0.005)
    kinect_hardware_holes = False
    if dataset_choice == "nyuv2":
        kinect_hardware_holes = st.sidebar.checkbox(
            "Hardware Kinect Absorption Shadows (Stress Test)",
            value=False,
            help="When checked, includes raw 2011 Kinect v1 hardware absorption blackouts in addition to multi-scale decimation.",
        )

    st.sidebar.markdown("---")
    st.sidebar.subheader("⚡ Quantization & Acceleration")
    quant_precision = st.sidebar.selectbox(
        "Inference Precision / Quantization",
        options=["bf16", "int8", "fp16", "fp32"],
        format_func=lambda x: {
            "bf16": "BF16 Tensor-Core AMP (2× Compression)",
            "int8": "INT8 Weight Quantized (~3.8× Compression)",
            "fp16": "FP16 Half-Precision (2× Compression)",
            "fp32": "FP32 Full Precision",
        }[x],
    )
    use_tta = st.sidebar.checkbox("Geometric Self-Ensemble (Flip-TTA)", value=False)
    q_stats = engine.get_quantization_report()
    st.sidebar.caption(
        f"**Model Params**: {q_stats.num_parameters:,}  \n"
        f"**FP32 Weights**: `{q_stats.fp32_size_mb:.2f} MB`  \n"
        f"**INT8 Quantized**: `{q_stats.compressed_size_mb:.2f} MB` (**{q_stats.compression_ratio:.2f}× smaller**)"
    )

    deg_cfg = DegradationConfig(
        scale=scale,
        hole_ratio=hole_ratio,
        structured_boundary_holes=structured_holes,
        noise_std=noise_std,
        outlier_prob=outlier_prob,
        seed=42,
    )

    if dataset_choice == "custom_upload":
        st.markdown("### 📸 Custom Camera & RGB-D Testing Studio")
        st.markdown(
            "Test the trained **EG-DepthNet + Hybrid Reconstructor** on your own camera photos, phone captures, or custom RGB-D datasets."
        )

        upload_mode = st.radio(
            "Select Input Source Mode",
            options=["photo_only", "paired_rgbd", "preset_samples"],
            format_func=lambda x: {
                "photo_only": "📷 Phone / Camera Photo (RGB Only — Instant Depth Prior & Upsampling)",
                "paired_rgbd": "🛰️ Paired RGB + Depth Map (.png, .npy, 16-bit millimeter depth)",
                "preset_samples": "🖼️ Try Built-in Real-World Camera Presets",
            }[x],
            horizontal=True,
        )

        target_res = st.selectbox(
            "Target Processing Resolution",
            options=[256, 384, 512],
            index=0,
            format_func=lambda x: f"{x} × {x} pixels ({'Fast' if x==256 else 'Balanced' if x==384 else 'High-Definition'})",
        )

        rgb_np = None
        depth_np = None
        sample_name = "custom_camera"

        PRESET_MAP = {
            "Art Studio (Sculpture & Easel)": Path("data/middlebury/2005_Art"),
            "Office Desk (Books & Complex Shapes)": Path("data/middlebury/2005_Books"),
            "Living Room Dolls (Fine Foreground Details)": Path("data/middlebury/2005_Dolls"),
            "Indoor Laundry (Cloth Folds & Occlusions)": Path("data/middlebury/2005_Laundry"),
            "Botanical Aloe (Thin Object Edges)": Path("data/middlebury/2006_Aloe"),
        }

        if upload_mode == "preset_samples":
            preset_choice = st.selectbox("Choose a Built-in Real Camera Scene", options=list(PRESET_MAP.keys()))
            p_dir = PRESET_MAP[preset_choice]
            rgb_path = next(p_dir.glob("view*.png"), None) or next(p_dir.glob("im*.png"), None)
            disp_path = next(p_dir.glob("disp*.png"), None) or next(p_dir.glob("disp*.pfm"), None)
            if rgb_path and rgb_path.exists():
                bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
                rgb_np = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            if disp_path and disp_path.exists():
                gray = cv2.imread(str(disp_path), cv2.IMREAD_UNCHANGED)
                if gray is not None:
                    if gray.ndim == 3:
                        gray = gray[:, :, 0]
                    disp_valid = np.nan_to_num(gray.astype(np.float32), nan=0.0)
                    depth_np = 100.0 / np.maximum(disp_valid, 5.0)
                    depth_np = np.clip(depth_np, 0.5, 6.0).astype(np.float32)
            sample_name = p_dir.name

        elif upload_mode == "photo_only":
            st.info("💡 **Camera Photo Testing**: Upload any photo from your phone or camera. A geometric depth prior will be synthesized and degraded at your selected factor (2×, 4×, 8×) to demonstrate how EG-DepthNet sharpens boundaries!")
            up_col1, up_col2 = st.columns([2, 1])
            with up_col1:
                rgb_file = st.file_uploader(
                    "Upload Camera Photo (.jpg, .png, .jpeg, .webp)",
                    type=["png", "jpg", "jpeg", "webp"],
                    key="camera_photo_uploader",
                )
            with up_col2:
                st.markdown("**Or Test with Sample Photos:**")
                if st.button("🛋️ Living Room Statue"):
                    st.session_state["camera_preset"] = "Art Studio (Sculpture & Easel)"
                if st.button("📚 Office Desk"):
                    st.session_state["camera_preset"] = "Office Desk (Books & Complex Shapes)"
                if st.button("🌿 Botanical Plant"):
                    st.session_state["camera_preset"] = "Botanical Aloe (Thin Object Edges)"

            if rgb_file is not None:
                file_bytes = np.asarray(bytearray(rgb_file.read()), dtype=np.uint8)
                bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if bgr is not None:
                    rgb_np = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                    sample_name = Path(rgb_file.name).stem
            elif "camera_preset" in st.session_state:
                p_dir = PRESET_MAP[st.session_state["camera_preset"]]
                rgb_path = next(p_dir.glob("view*.png"), None)
                if rgb_path and rgb_path.exists():
                    bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
                    rgb_np = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                    sample_name = p_dir.name
            else:
                p_dir = PRESET_MAP["Art Studio (Sculpture & Easel)"]
                rgb_path = next(p_dir.glob("view*.png"), None)
                if rgb_path and rgb_path.exists():
                    bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
                    rgb_np = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                    sample_name = "camera_demo_art"

        elif upload_mode == "paired_rgbd":
            st.info("💡 **Paired RGB-D Testing**: Upload both an RGB guidance image and its corresponding depth map (supports 8-bit, 16-bit millimeter PNG, or .npy).")
            p_col1, p_col2 = st.columns(2)
            with p_col1:
                rgb_file = st.file_uploader("Upload High-Res RGB Image (.png, .jpg)", type=["png", "jpg", "jpeg", "webp"], key="paired_rgb")
            with p_col2:
                depth_file = st.file_uploader("Upload Aligned Depth Map (.png, .npy)", type=["png", "npy"], key="paired_depth")

            if rgb_file is not None:
                file_bytes = np.asarray(bytearray(rgb_file.read()), dtype=np.uint8)
                bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if bgr is not None:
                    rgb_np = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                    sample_name = Path(rgb_file.name).stem

            if depth_file is not None:
                if depth_file.name.endswith(".npy"):
                    depth_np = np.load(depth_file).astype(np.float32)
                else:
                    d_bytes = np.asarray(bytearray(depth_file.read()), dtype=np.uint8)
                    d_raw = cv2.imdecode(d_bytes, cv2.IMREAD_UNCHANGED)
                    if d_raw is not None:
                        if d_raw.ndim == 3:
                            d_raw = d_raw[:, :, 0]
                        d_raw_f = d_raw.astype(np.float32)
                        if d_raw_f.max() > 20.0:  # 16-bit millimeter depth
                            depth_np = np.clip(d_raw_f / 1000.0, 0.2, 10.0)
                        else:  # 8-bit grayscale normalized
                            depth_np = 0.8 + 4.2 * (d_raw_f - d_raw_f.min()) / max(float(d_raw_f.max() - d_raw_f.min()), 1e-5)

            if rgb_np is None or depth_np is None:
                st.warning("Please upload both an RGB image and a Depth map to run paired evaluation. Or click below to load a sample:")
                if st.button("🖼️ Load Demo Paired Scene (Office Desk)"):
                    p_dir = PRESET_MAP["Office Desk (Books & Complex Shapes)"]
                    bgr = cv2.imread(str(next(p_dir.glob("view*.png"))), cv2.IMREAD_COLOR)
                    rgb_np = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                    gray = cv2.imread(str(next(p_dir.glob("disp*.png"))), cv2.IMREAD_GRAYSCALE).astype(np.float32)
                    depth_np = np.clip(100.0 / np.maximum(gray, 5.0), 0.5, 6.0)
                    sample_name = "demo_office_desk"

        if rgb_np is None:
            st.error("No image available. Please upload an RGB photo.")
            return

        # Resize / Center-crop cleanly to (target_res, target_res)
        h_orig, w_orig = rgb_np.shape[:2]
        crop_dim = min(h_orig, w_orig)
        cy, cx = h_orig // 2, w_orig // 2
        rgb_cropped = rgb_np[cy - crop_dim // 2 : cy + crop_dim // 2, cx - crop_dim // 2 : cx + crop_dim // 2]
        rgb_res = cv2.resize(rgb_cropped, (target_res, target_res))

        if depth_np is not None:
            dh_orig, dw_orig = depth_np.shape[:2]
            d_dim = min(dh_orig, dw_orig)
            dcy, dcx = dh_orig // 2, dw_orig // 2
            d_cropped = depth_np[dcy - d_dim // 2 : dcy + d_dim // 2, dcx - d_dim // 2 : dcx + d_dim // 2]
            depth_res = cv2.resize(d_cropped, (target_res, target_res), interpolation=cv2.INTER_NEAREST)
        else:
            depth_res = generate_camera_depth_prior(rgb_res)

        rgb_t = torch.from_numpy(rgb_res.transpose(2, 0, 1)).float()
        depth_t = torch.from_numpy(depth_res[None, ...]).float()
        degraded = degrade_depth(depth_t, rgb_t, deg_cfg)

        sample = {
            "name": sample_name,
            "scale": scale,
            "depth_lr": degraded.depth_lr,
            "valid_mask_lr": degraded.valid_mask_lr,
            "rgb_hr": degraded.rgb_hr,
            "depth_gt": degraded.depth_gt,
            "valid_mask_gt": degraded.valid_mask_gt,
            "boundary_mask_gt": degraded.boundary_mask_gt,
        }
    else:
        ds = load_dataset(
            name=dataset_choice,  # type: ignore[arg-type]
            split="test",
            degradation=deg_cfg,
            crop_size=256,
            max_samples=16,
            fallback_to_synthetic=True,
            use_raw_sensor_holes=kinect_hardware_holes,
        )
        scene_idx = st.sidebar.number_input("Scene Index", min_value=0, max_value=max(0, len(ds) - 1), value=0)
        sample = ds[int(scene_idx)]

    depth_lr = sample["depth_lr"].to(device)  # type: ignore[union-attr]
    valid_lr = sample["valid_mask_lr"].to(device)  # type: ignore[union-attr]
    rgb_hr = sample["rgb_hr"].to(device)  # type: ignore[union-attr]
    depth_gt = sample["depth_gt"].to(device)  # type: ignore[union-attr]
    valid_gt = sample["valid_mask_gt"].to(device)  # type: ignore[union-attr]
    boundary_gt = sample["boundary_mask_gt"].to(device)  # type: ignore[union-attr]

    # Run all methods and collect outputs + metrics
    preds: Dict[str, np.ndarray] = {}
    metrics_rows = []

    for m in ALL_METHODS:
        if m in ("nearest", "bilinear", "bicubic", "guided", "jbf"):
            out_t, lat_ms, mem_mb = measure_runtime_and_memory(
                lambda m_name=m: reconstruct_classical(
                    depth_lr, rgb_hr, scale=scale, method=m_name, valid_mask_lr=valid_lr  # type: ignore[arg-type]
                ),
                device=device,
            )
        elif m == "deep":
            res_obj, lat_ms, mem_mb = measure_runtime_and_memory(
                lambda: engine.reconstruct(
                    depth_lr, rgb_hr, scale=scale, valid_mask_lr=valid_lr, precision=quant_precision, use_tta=use_tta  # type: ignore[arg-type]
                ),
                device=device,
            )
            out_t = res_obj.depth_deep
        elif m == "hybrid_int8":
            res_obj, lat_ms, mem_mb = measure_runtime_and_memory(
                lambda: engine.reconstruct(
                    depth_lr, rgb_hr, scale=scale, valid_mask_lr=valid_lr, precision="int8", use_tta=False
                ),
                device=device,
            )
            out_t = res_obj.depth_hybrid
        elif m == "hybrid_tta":
            res_obj, lat_ms, mem_mb = measure_runtime_and_memory(
                lambda: engine.reconstruct(
                    depth_lr, rgb_hr, scale=scale, valid_mask_lr=valid_lr, precision=quant_precision, use_tta=True  # type: ignore[arg-type]
                ),
                device=device,
            )
            out_t = res_obj.depth_hybrid
        else:
            res_obj, lat_ms, mem_mb = measure_runtime_and_memory(
                lambda: engine.reconstruct(
                    depth_lr, rgb_hr, scale=scale, valid_mask_lr=valid_lr, precision=quant_precision, use_tta=use_tta  # type: ignore[arg-type]
                ),
                device=device,
            )
            out_t = res_obj.depth_hybrid
            edge_gate_np = res_obj.edge_gate[0].detach().cpu().numpy()
            uncertainty_np = res_obj.uncertainty[0].detach().cpu().numpy()

        preds[m] = out_t[0].detach().cpu().numpy()
        m_dict = compute_depth_metrics(out_t, depth_gt, valid_gt, boundary_gt, rgb_hr)
        m_dict["Method"] = METHOD_LABELS[m]
        m_dict["Key"] = m
        m_dict["Latency (ms)"] = lat_ms
        m_dict["Memory (MB)"] = mem_mb
        metrics_rows.append(m_dict)

    rgb_np = rgb_hr.detach().cpu().numpy().transpose(1, 2, 0)
    gt_np = depth_gt[0].detach().cpu().numpy()
    lr_np = depth_lr[0].detach().cpu().numpy()
    lr_valid_np = valid_lr[0].detach().cpu().numpy() > 0.5
    boundary_np = boundary_gt[0].detach().cpu().numpy()

    vmin, vmax = float(np.percentile(gt_np, 2)), float(np.percentile(gt_np, 98))

    # Top KPI Banner
    best_row = next(r for r in metrics_rows if r["Key"] == "hybrid")
    bic_row = next(r for r in metrics_rows if r["Key"] == "bicubic")
    rmse_gain = (bic_row["RMSE"] - best_row["RMSE"]) / max(bic_row["RMSE"], 1e-6) * 100.0
    b_rmse_gain = (bic_row["Boundary_RMSE"] - best_row["Boundary_RMSE"]) / max(bic_row["Boundary_RMSE"], 1e-6) * 100.0

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Hybrid RMSE (m) ↓", f"{best_row['RMSE']:.4f} m", f"-{rmse_gain:.1f}% vs Bicubic")
    k2.metric("Boundary RMSE (m) ↓", f"{best_row['Boundary_RMSE']:.4f} m", f"-{b_rmse_gain:.1f}% vs Bicubic")
    k3.metric("Hybrid PSNR ↑", f"{best_row['PSNR']:.2f} dB", f"+{best_row['PSNR'] - bic_row['PSNR']:.2f} dB")
    k4.metric("Edge-F1 Score ↑", f"{best_row['Edge_F1']:.4f}", f"SSIM: {best_row['SSIM']:.4f}")
    k5.metric("Inference Latency", f"{best_row['Latency (ms)']:.1f} ms", f"{best_row['Memory (MB)']:.1f} MB VRAM")

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "🖼️ Visual Reconstruction Studio",
            "🔍 Boundary & Discontinuity Inspector",
            "🌐 Interactive 3D Point Cloud",
            "📊 Quantitative Benchmark Leaderboard",
        ]
    )

    with tab1:
        st.subheader(f"Scene `{sample['name']}` — {scale}× Depth Upsampling & Refinement")
        c1, c2, c3, c4 = st.columns(4)
        c1.image((rgb_np * 255).astype(np.uint8), caption="High-Res RGB Guidance", use_container_width=True)
        lr_up_vis = cv2.resize(
            colorize_depth(lr_np, vmin, vmax, invalid_mask=~lr_valid_np),
            (gt_np.shape[1], gt_np.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        c2.image(
            lr_up_vis,
            caption=f"Degraded LR Input ({lr_np.shape[1]}×{lr_np.shape[0]}, Holes & Noise)",
            use_container_width=True,
        )
        c3.image(
            colorize_depth(preds["bicubic"], vmin, vmax),
            caption=f"Bicubic Baseline (RMSE: {bic_row['RMSE']:.4f}m)",
            use_container_width=True,
        )
        c4.image(
            colorize_depth(preds["hybrid"], vmin, vmax),
            caption=f"Hybrid EG-DepthNet (RMSE: {best_row['RMSE']:.4f}m)",
            use_container_width=True,
        )

        st.markdown("#### Full Method Comparison Matrix (Classical vs. Deep vs. Hybrid)")
        m_cols = st.columns(5)
        for idx, m_key in enumerate(["nearest", "guided", "jbf", "deep", "hybrid"]):
            r_info = next(r for r in metrics_rows if r["Key"] == m_key)
            m_cols[idx].image(
                colorize_depth(preds[m_key], vmin, vmax),
                caption=f"{METHOD_LABELS[m_key]}\nRMSE: {r_info['RMSE']:.4f}m | Edge-F1: {r_info['Edge_F1']:.3f}",
                use_container_width=True,
            )

        st.markdown("#### Internal Explainability: Learned Edge-Coherence Gate ($G_{edge}$) & Uncertainty Map")
        e1, e2, e3 = st.columns(3)
        e1.image(
            colorize_depth(edge_gate_np, 0.0, 1.0, cmap_name="magma"),
            caption="Learned Edge-Coherence Gate G_edge (Suppresses Texture-Copying on Flat Walls)",
            use_container_width=True,
        )
        e2.image(
            colorize_depth(uncertainty_np, None, None, cmap_name="inferno"),
            caption="Predicted Aleatoric Uncertainty (Drives Stage-3 Bilateral Energy Fusion)",
            use_container_width=True,
        )
        e3.image(
            colorize_depth(gt_np, vmin, vmax),
            caption="Ground Truth High-Resolution Depth",
            use_container_width=True,
        )

    with tab2:
        st.subheader("Object Boundary & Depth Discontinuity Analysis")
        H, W = gt_np.shape
        col_ctrl1, col_ctrl2 = st.columns(2)
        crop_y = col_ctrl1.slider("Zoom Crop Center Y", 40, H - 40, H // 2)
        crop_x = col_ctrl2.slider("Zoom Crop Center X", 40, W - 40, W // 2)
        r_crop = 36
        y0, y1 = max(0, crop_y - r_crop), min(H, crop_y + r_crop)
        x0, x1 = max(0, crop_x - r_crop), min(W, crop_x + r_crop)

        zc1, zc2, zc3, zc4, zc5 = st.columns(5)
        zc1.image(
            cv2.resize((rgb_np[y0:y1, x0:x1] * 255).astype(np.uint8), (220, 220), interpolation=cv2.INTER_NEAREST),
            caption="RGB Patch",
            use_container_width=True,
        )
        zc2.image(
            cv2.resize(
                colorize_depth(preds["nearest"][y0:y1, x0:x1], vmin, vmax), (220, 220), interpolation=cv2.INTER_NEAREST
            ),
            caption="Nearest (Blocky Artifacts)",
            use_container_width=True,
        )
        zc3.image(
            cv2.resize(
                colorize_depth(preds["bicubic"][y0:y1, x0:x1], vmin, vmax), (220, 220), interpolation=cv2.INTER_NEAREST
            ),
            caption="Bicubic (Blurred Boundary)",
            use_container_width=True,
        )
        zc4.image(
            cv2.resize(
                colorize_depth(preds["hybrid"][y0:y1, x0:x1], vmin, vmax), (220, 220), interpolation=cv2.INTER_NEAREST
            ),
            caption="Hybrid (Sharp Discontinuity)",
            use_container_width=True,
        )
        zc5.image(
            cv2.resize(colorize_depth(gt_np[y0:y1, x0:x1], vmin, vmax), (220, 220), interpolation=cv2.INTER_NEAREST),
            caption="Ground Truth Patch",
            use_container_width=True,
        )

        # 1D Cross-Section Depth Profile along row `crop_y`
        st.markdown(f"#### 1D Horizontal Depth Profile across Occlusion Discontinuity (Row Y = {crop_y})")
        fig_prof = go.Figure()
        xs = np.arange(W)
        fig_prof.add_trace(go.Scatter(x=xs, y=gt_np[crop_y, :], name="Ground Truth", line=dict(color="black", width=3)))
        fig_prof.add_trace(
            go.Scatter(
                x=xs,
                y=preds["bicubic"][crop_y, :],
                name="Bicubic (Oversmoothed Step)",
                line=dict(color="#ef553b", dash="dot"),
            )
        )
        fig_prof.add_trace(
            go.Scatter(x=xs, y=preds["jbf"][crop_y, :], name="Classical JBF", line=dict(color="#ffa15a", dash="dash"))
        )
        fig_prof.add_trace(
            go.Scatter(
                x=xs,
                y=preds["hybrid"][crop_y, :],
                name="Hybrid (Ours - Sharp Step)",
                line=dict(color="#00cc96", width=2.5),
            )
        )
        fig_prof.update_layout(
            xaxis_title="Pixel Column X",
            yaxis_title="Depth (meters)",
            height=340,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig_prof, use_container_width=True)

        st.markdown("#### Absolute Depth Error Heatmaps ($|\\hat{D} - D_{gt}|$, darker = lower error)")
        err_max = max(0.35, float(np.percentile(np.abs(preds["bicubic"] - gt_np), 97)))
        ec1, ec2, ec3, ec4 = st.columns(4)
        for col_ui, m_key in zip([ec1, ec2, ec3, ec4], ["bicubic", "guided", "jbf", "hybrid"]):
            err_map = np.abs(preds[m_key] - gt_np)
            col_ui.image(
                colorize_depth(err_map, 0.0, err_max, cmap_name="magma"),
                caption=f"{METHOD_LABELS[m_key]} Error (Mean MAE: {err_map.mean():.4f}m)",
                use_container_width=True,
            )

    with tab3:
        st.subheader("Interactive 3D RGB-D Point Cloud Comparison")
        st.markdown(
            "Conventional interpolation (`Bicubic`) produces **flying pixels** bridging foreground and background objects at occlusion boundaries. "
            "Our **Hybrid Edge-Gated Reconstruction** preserves clean 3D geometric separation."
        )
        pc1, pc2 = st.columns(2)
        with pc1:
            st.plotly_chart(
                build_3d_point_cloud_figure(preds["bicubic"], rgb_np, "Bicubic Baseline (Flying Pixels at Edges)"),
                use_container_width=True,
            )
        with pc2:
            st.plotly_chart(
                build_3d_point_cloud_figure(
                    preds["hybrid"], rgb_np, "Hybrid EG-DepthNet (Crisp 3D Object Boundaries)"
                ),
                use_container_width=True,
            )

    with tab4:
        st.subheader(f"Live Quantitative Evaluation Table ({scale}× Upsampling)")
        df = pd.DataFrame(metrics_rows)[
            [
                "Method",
                "RMSE",
                "MAE",
                "PSNR",
                "SSIM",
                "Boundary_RMSE",
                "Edge_F1",
                "Texture_Copy_Err",
                "Delta1",
                "Latency (ms)",
                "Memory (MB)",
            ]
        ]
        st.dataframe(
            df.style.highlight_min(
                subset=["RMSE", "MAE", "Boundary_RMSE", "Texture_Copy_Err"], color="#1b4332"
            ).highlight_max(subset=["PSNR", "SSIM", "Edge_F1", "Delta1"], color="#1b4332"),
            use_container_width=True,
        )

        bc1, bc2 = st.columns(2)
        with bc1:
            fig_bar = px.bar(
                df,
                x="Method",
                y=["RMSE", "Boundary_RMSE", "MAE"],
                barmode="group",
                title="Global vs. Boundary Error Comparison (Lower is Better ↓)",
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        with bc2:
            fig_f1 = px.bar(
                df,
                x="Method",
                y=["Edge_F1", "SSIM"],
                barmode="group",
                title="Boundary F1 & Structural Similarity (Higher is Better ↑)",
            )
            st.plotly_chart(fig_f1, use_container_width=True)

        report_json = Path("outputs/benchmark_report.json")
        if report_json.exists():
            st.markdown("#### Saved Multi-Dataset & Multi-Scale Benchmark Report (2×, 4×, 8×)")
            saved_data = json.loads(report_json.read_text(encoding="utf-8"))
            flat_rows = []
            for item in saved_data:
                row = {"Dataset": item["dataset"].upper(), "Scale": f"{item['scale']}×", "Method": item["label"]}
                row.update(item["metrics"])
                flat_rows.append(row)
            st.dataframe(pd.DataFrame(flat_rows), use_container_width=True)

    # --- Export Reconstructed Depth Map ---
    st.markdown("---")
    st.markdown("### 💾 Export Reconstructed Depth Map")
    exp_col1, exp_col2 = st.columns(2)
    depth_mm = (preds["hybrid"] * 1000.0).clip(0, 65535).astype(np.uint16)
    _, png_buf = cv2.imencode(".png", depth_mm)
    exp_col1.download_button(
        "📥 Download High-Res Depth (16-bit PNG, Millimeters)",
        data=png_buf.tobytes(),
        file_name=f"{sample['name']}_hybrid_depth_16bit.png",
        mime="image/png",
        use_container_width=True,
    )
    import io
    npy_buf = io.BytesIO()
    np.save(npy_buf, preds["hybrid"].astype(np.float32))
    exp_col2.download_button(
        "📥 Download Depth Map as Numpy Array (.npy)",
        data=npy_buf.getvalue(),
        file_name=f"{sample['name']}_hybrid_depth.npy",
        mime="application/octet-stream",
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
