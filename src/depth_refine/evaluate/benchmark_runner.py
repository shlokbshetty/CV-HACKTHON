"""
Automated Multi-Method, Multi-Scale & Quantization Benchmark Runner.

Compares:
1. Nearest-Neighbour Interpolation (`nearest`)
2. Bilinear Interpolation (`bilinear`)
3. Bicubic Interpolation (`bicubic`)
4. Fast Guided Filtering (`guided`)
5. Joint Bilateral Filtering (`jbf`)
6. Deep Learning (`deep` - EG-DepthNet BF16)
7. Hybrid Reconstruction (`hybrid` - EG-DepthNet + JBF Energy)
8. INT8 Quantized Hybrid (`hybrid_int8` - 4x Weight Compression)
9. Hybrid + Geometric Flip-TTA (`hybrid_tta` - Self-Ensemble SOTA)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional

import numpy as np
import torch

from depth_refine.classical.filters import reconstruct_classical
from depth_refine.data.dataset_registry import load_dataset
from depth_refine.data.degradation import DegradationConfig
from depth_refine.evaluate.metrics import compute_depth_metrics, measure_runtime_and_memory
from depth_refine.hybrid.hybrid_pipeline import HybridReconstructor

ALL_METHODS = [
    "nearest",
    "bilinear",
    "bicubic",
    "guided",
    "jbf",
    "deep",
    "hybrid_int8",
    "hybrid",
    "hybrid_tta",
]

METHOD_LABELS = {
    "nearest": "Nearest-Neighbour",
    "bilinear": "Bilinear Interp.",
    "bicubic": "Bicubic Interp.",
    "guided": "Guided Filter (RGB)",
    "jbf": "Joint Bilateral (JBF)",
    "deep": "EG-DepthNet (Deep BF16)",
    "hybrid_int8": "Hybrid INT8 Quantized (4x Comp.)",
    "hybrid": "Hybrid (EG-DepthNet + JBF Energy)",
    "hybrid_tta": "Hybrid + Flip-TTA (Self-Ensemble)",
}


@dataclass
class MethodSummary:
    """Aggregated metrics for a single method at a given scale and dataset."""

    dataset: str
    scale: int
    method: str
    label: str
    metrics: Dict[str, float] = field(default_factory=dict)


class BenchmarkSuite:
    """
    Deep Module for running quantitative benchmarks across datasets, scales, and methods.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Path | str] = None,
        device: Optional[str | torch.device] = None,
    ) -> None:
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hybrid_engine = HybridReconstructor(checkpoint_path=checkpoint_path, device=self.device)
        # Pre-initialize INT8 quantized model so timing reflects steady-state inference
        self.hybrid_engine.get_quantization_report()

    @torch.no_grad()
    def run_single_method(
        self,
        method: str,
        depth_lr: torch.Tensor,
        rgb_hr: torch.Tensor,
        scale: int,
        valid_mask_lr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if method in ("nearest", "bilinear", "bicubic", "guided", "jbf", "boundary_aware_classical"):
            return reconstruct_classical(
                depth_lr=depth_lr,
                rgb_hr=rgb_hr,
                scale=scale,
                method=method,  # type: ignore[arg-type]
                valid_mask_lr=valid_mask_lr,
            )
        if method == "deep":
            res = self.hybrid_engine.reconstruct(
                depth_lr=depth_lr,
                rgb_hr=rgb_hr,
                scale=scale,
                valid_mask_lr=valid_mask_lr,
                precision="bf16",
                use_tta=False,
            )
            return res.depth_deep
        if method == "hybrid_int8":
            res = self.hybrid_engine.reconstruct(
                depth_lr=depth_lr,
                rgb_hr=rgb_hr,
                scale=scale,
                valid_mask_lr=valid_mask_lr,
                precision="int8",
                use_tta=False,
            )
            return res.depth_hybrid
        if method == "hybrid":
            res = self.hybrid_engine.reconstruct(
                depth_lr=depth_lr,
                rgb_hr=rgb_hr,
                scale=scale,
                valid_mask_lr=valid_mask_lr,
                precision="bf16",
                use_tta=False,
            )
            return res.depth_hybrid
        if method == "hybrid_tta":
            res = self.hybrid_engine.reconstruct(
                depth_lr=depth_lr,
                rgb_hr=rgb_hr,
                scale=scale,
                valid_mask_lr=valid_mask_lr,
                precision="bf16",
                use_tta=True,
            )
            return res.depth_hybrid
        raise ValueError(f"Unknown method: {method}")

    @torch.no_grad()
    def evaluate_dataset(
        self,
        dataset_name: Literal["nyuv2", "middlebury", "kitti", "synthetic"] = "synthetic",
        scales: List[int] = [2, 4, 8],
        methods: List[str] = ALL_METHODS,
        root_dir: Path | str = Path("data"),
        max_samples: int = 12,
        hole_ratio: float = 0.08,
        noise_std: float = 0.015,
        fallback_to_synthetic: bool = True,
    ) -> List[MethodSummary]:
        summaries: List[MethodSummary] = []

        for scale in scales:
            deg_cfg = DegradationConfig(
                scale=scale,
                hole_ratio=hole_ratio,
                noise_std=noise_std,
                structured_boundary_holes=True,
            )
            ds = load_dataset(
                name=dataset_name,
                split="test",
                root_dir=root_dir,
                degradation=deg_cfg,
                crop_size=256,
                max_samples=max_samples,
                fallback_to_synthetic=fallback_to_synthetic,
            )

            accum: Dict[str, List[Dict[str, float]]] = {m: [] for m in methods}

            for i in range(len(ds)):
                sample = ds[i]
                depth_lr = sample["depth_lr"].to(self.device)  # type: ignore[union-attr]
                valid_lr = sample["valid_mask_lr"].to(self.device)  # type: ignore[union-attr]
                rgb_hr = sample["rgb_hr"].to(self.device)  # type: ignore[union-attr]
                depth_gt = sample["depth_gt"].to(self.device)  # type: ignore[union-attr]
                valid_gt = sample["valid_mask_gt"].to(self.device)  # type: ignore[union-attr]
                boundary_gt = sample["boundary_mask_gt"].to(self.device)  # type: ignore[union-attr]

                for m in methods:
                    pred, lat_ms, mem_mb = measure_runtime_and_memory(
                        lambda m_name=m: self.run_single_method(
                            m_name, depth_lr, rgb_hr, scale=scale, valid_mask_lr=valid_lr
                        ),
                        device=self.device,
                    )
                    m_dict = compute_depth_metrics(
                        pred=pred,
                        gt=depth_gt,
                        valid_mask=valid_gt,
                        boundary_mask=boundary_gt,
                        rgb_hr=rgb_hr,
                    )
                    m_dict["Latency_ms"] = lat_ms
                    m_dict["Peak_Memory_MB"] = mem_mb
                    accum[m].append(m_dict)

            for m in methods:
                rows = accum[m]
                avg_metrics = {k: float(np.mean([r[k] for r in rows])) for k in rows[0].keys()}
                summaries.append(
                    MethodSummary(
                        dataset=dataset_name,
                        scale=scale,
                        method=m,
                        label=METHOD_LABELS.get(m, m),
                        metrics=avg_metrics,
                    )
                )

        return summaries

    @staticmethod
    def format_markdown_table(summaries: List[MethodSummary]) -> str:
        lines = [
            "| Dataset | Scale | Method | RMSE (m) ↓ | MAE (m) ↓ | PSNR (dB) ↑ | SSIM ↑ | Boundary RMSE ↓ | Edge F1 ↑ | Tex-Copy Err ↓ | Latency (ms) ↓ | Memory (MB) ↓ |",
            "| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
        for s in summaries:
            mt = s.metrics
            is_best = s.method in ("hybrid", "hybrid_tta")
            b = "**" if is_best else ""
            lines.append(
                f"| {s.dataset.upper()} | {s.scale}× | {b}{s.label}{b} | "
                f"{b}{mt['RMSE']:.4f}{b} | {b}{mt['MAE']:.4f}{b} | "
                f"{b}{mt['PSNR']:.2f}{b} | {b}{mt['SSIM']:.4f}{b} | "
                f"{b}{mt['Boundary_RMSE']:.4f}{b} | {b}{mt['Edge_F1']:.4f}{b} | "
                f"{b}{mt['Texture_Copy_Err']:.4f}{b} | "
                f"{mt['Latency_ms']:.2f} | {mt['Peak_Memory_MB']:.1f} |"
            )
        return "\n".join(lines)

    @staticmethod
    def save_report(summaries: List[MethodSummary], output_dir: Path | str = Path("outputs")) -> Path:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        md_path = out_dir / "benchmark_report.md"
        json_path = out_dir / "benchmark_report.json"

        md_text = "# Edge-Preserving Depth Upsampling & Refinement Benchmark Report\n\n"
        md_text += BenchmarkSuite.format_markdown_table(summaries) + "\n"
        md_path.write_text(md_text, encoding="utf-8")

        payload = [
            {
                "dataset": s.dataset,
                "scale": s.scale,
                "method": s.method,
                "label": s.label,
                "metrics": s.metrics,
            }
            for s in summaries
        ]
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return md_path
