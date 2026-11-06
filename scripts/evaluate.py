"""
CLI Benchmark Runner for ProbID - P4.

Evaluates Nearest, Bilinear, Bicubic, Guided Filter, Joint Bilateral Filter,
EG-DepthNet (Deep), and Hybrid across 2x, 4x, and 8x upsampling factors.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console
from rich.table import Table

from depth_refine.evaluate.benchmark_runner import ALL_METHODS, BenchmarkSuite

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Depth Upsampling & Refinement Benchmark Suite.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["nyuv2", "middlebury", "kitti"],
        help="Datasets to evaluate (nyuv2, middlebury, kitti, synthetic)",
    )
    parser.add_argument("--scales", nargs="+", type=int, default=[2, 4, 8], help="Upsampling scales (e.g. 2 4 8)")
    parser.add_argument("--max-samples", type=int, default=6, help="Number of test samples per dataset")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/eg_depthnet_best.pt")
    parser.add_argument("--output-dir", type=str, default="outputs")
    args = parser.parse_args()

    suite = BenchmarkSuite(checkpoint_path=args.checkpoint)
    all_summaries = []

    for ds_name in args.datasets:
        console.print(f"\n[bold cyan]Evaluating Dataset: {ds_name.upper()} across scales {args.scales}...[/bold cyan]")
        summaries = suite.evaluate_dataset(
            dataset_name=ds_name,  # type: ignore[arg-type]
            scales=args.scales,
            methods=ALL_METHODS,
            max_samples=args.max_samples,
            fallback_to_synthetic=True,
        )
        all_summaries.extend(summaries)

        table = Table(title=f"{ds_name.upper()} Depth Reconstruction Benchmark")
        table.add_column("Scale", justify="center")
        table.add_column("Method", style="cyan")
        table.add_column("RMSE (m) ↓", justify="right")
        table.add_column("MAE (m) ↓", justify="right")
        table.add_column("PSNR (dB) ↑", justify="right")
        table.add_column("SSIM ↑", justify="right")
        table.add_column("Boundary RMSE ↓", justify="right")
        table.add_column("Edge F1 ↑", justify="right")
        table.add_column("Tex-Copy ↓", justify="right")
        table.add_column("Latency (ms)", justify="right")

        for s in summaries:
            mt = s.metrics
            style = "bold green" if s.method in ("hybrid", "deep") else None
            table.add_row(
                f"{s.scale}×",
                s.label,
                f"{mt['RMSE']:.4f}",
                f"{mt['MAE']:.4f}",
                f"{mt['PSNR']:.2f}",
                f"{mt['SSIM']:.4f}",
                f"{mt['Boundary_RMSE']:.4f}",
                f"{mt['Edge_F1']:.4f}",
                f"{mt['Texture_Copy_Err']:.4f}",
                f"{mt['Latency_ms']:.1f}",
                style=style,
            )
        console.print(table)

    md_path = BenchmarkSuite.save_report(all_summaries, output_dir=Path(args.output_dir))
    console.print(f"\n[bold green]✓ Full benchmark report saved to {md_path}[/bold green]")


if __name__ == "__main__":
    main()

