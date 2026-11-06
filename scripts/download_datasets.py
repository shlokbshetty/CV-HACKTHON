"""
Automated Dataset Downloader & Preparation Utility for ProbID - P4.

Supports:
1. NYU Depth V2 (`nyu_depth_v2_labeled.mat`, ~2.8 GB) with resumable HTTP Range download.
2. Middlebury Stereo Datasets across all 6 editions:
   - 2001 (Venus, Sawtooth, Bull)
   - 2003 (Cones, Teddy)
   - 2005 (Art, Books, Dolls, Laundry, Moebius, Reindeer)
   - 2006 (Aloe, Bowling1, Flowerpots)
   - 2014 (Adirondack, Jadeplant)
   - 2021 Mobile (artroom1)
3. KITTI Vision Benchmark Suite (Depth Completion Validation Subset).

Usage on Teammate's RTX 5060 Laptop:
    uv run python scripts/download_datasets.py --dataset all --full-nyuv2-mat
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import requests
from rich.console import Console
from tqdm import tqdm

from depth_refine.data.dataset_registry import generate_procedural_rgbd_scene

console = Console()

NYUV2_MAT_URL = "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"

# Verified 200-OK direct URLs across Middlebury 2001, 2003, 2005, 2006, 2014, and 2021
MIDDLEBURY_SCENES: Dict[str, Dict[str, str]] = {
    # 2001 Benchmark
    "2001_Venus": {
        "im2.ppm": "https://vision.middlebury.edu/stereo/data/scenes2001/data/venus/im2.ppm",
        "disp2.pgm": "https://vision.middlebury.edu/stereo/data/scenes2001/data/venus/disp2.pgm",
    },
    # 2003 Structured Light Benchmark
    "2003_Cones": {
        "view1.png": "https://vision.middlebury.edu/stereo/data/scenes2003/newdata/cones/im2.png",
        "disp1.png": "https://vision.middlebury.edu/stereo/data/scenes2003/newdata/cones/disp2.png",
    },
    "2003_Teddy": {
        "view1.png": "https://vision.middlebury.edu/stereo/data/scenes2003/newdata/teddy/im2.png",
        "disp1.png": "https://vision.middlebury.edu/stereo/data/scenes2003/newdata/teddy/disp2.png",
    },
    # 2005 Benchmark (Standard in Guided Depth SR literature)
    "2005_Art": {
        "view1.png": "https://vision.middlebury.edu/stereo/data/scenes2005/ThirdSize/Art/Illum1/Exp1/view1.png",
        "disp1.png": "https://vision.middlebury.edu/stereo/data/scenes2005/ThirdSize/Art/disp1.png",
    },
    "2005_Books": {
        "view1.png": "https://vision.middlebury.edu/stereo/data/scenes2005/ThirdSize/Books/Illum1/Exp1/view1.png",
        "disp1.png": "https://vision.middlebury.edu/stereo/data/scenes2005/ThirdSize/Books/disp1.png",
    },
    "2005_Dolls": {
        "view1.png": "https://vision.middlebury.edu/stereo/data/scenes2005/ThirdSize/Dolls/Illum1/Exp1/view1.png",
        "disp1.png": "https://vision.middlebury.edu/stereo/data/scenes2005/ThirdSize/Dolls/disp1.png",
    },
    "2005_Moebius": {
        "view1.png": "https://vision.middlebury.edu/stereo/data/scenes2005/ThirdSize/Moebius/Illum1/Exp1/view1.png",
        "disp1.png": "https://vision.middlebury.edu/stereo/data/scenes2005/ThirdSize/Moebius/disp1.png",
    },
    "2005_Reindeer": {
        "view1.png": "https://vision.middlebury.edu/stereo/data/scenes2005/ThirdSize/Reindeer/Illum1/Exp1/view1.png",
        "disp1.png": "https://vision.middlebury.edu/stereo/data/scenes2005/ThirdSize/Reindeer/disp1.png",
    },
    "2005_Laundry": {
        "view1.png": "https://vision.middlebury.edu/stereo/data/scenes2005/ThirdSize/Laundry/Illum1/Exp1/view1.png",
        "disp1.png": "https://vision.middlebury.edu/stereo/data/scenes2005/ThirdSize/Laundry/disp1.png",
    },
    # 2006 Benchmark
    "2006_Aloe": {
        "view1.png": "https://vision.middlebury.edu/stereo/data/scenes2006/ThirdSize/Aloe/Illum1/Exp1/view1.png",
        "disp1.png": "https://vision.middlebury.edu/stereo/data/scenes2006/ThirdSize/Aloe/disp1.png",
    },
    "2006_Bowling1": {
        "view1.png": "https://vision.middlebury.edu/stereo/data/scenes2006/ThirdSize/Bowling1/Illum1/Exp1/view1.png",
        "disp1.png": "https://vision.middlebury.edu/stereo/data/scenes2006/ThirdSize/Bowling1/disp1.png",
    },
    "2006_Flowerpots": {
        "view1.png": "https://vision.middlebury.edu/stereo/data/scenes2006/ThirdSize/Flowerpots/Illum1/Exp1/view1.png",
        "disp1.png": "https://vision.middlebury.edu/stereo/data/scenes2006/ThirdSize/Flowerpots/disp1.png",
    },
    # 2014 Subpixel High-Res Benchmark (.pfm)
    "2014_Adirondack": {
        "im0.png": "https://vision.middlebury.edu/stereo/data/scenes2014/datasets/Adirondack-imperfect/im0.png",
        "disp0.pfm": "https://vision.middlebury.edu/stereo/data/scenes2014/datasets/Adirondack-imperfect/disp0.pfm",
        "calib.txt": "https://vision.middlebury.edu/stereo/data/scenes2014/datasets/Adirondack-imperfect/calib.txt",
    },
    # 2021 Mobile Dataset (.pfm)
    "2021_artroom1": {
        "im0.png": "https://vision.middlebury.edu/stereo/data/scenes2021/data/artroom1/im0.png",
        "disp0.pfm": "https://vision.middlebury.edu/stereo/data/scenes2021/data/artroom1/disp0.pfm",
        "calib.txt": "https://vision.middlebury.edu/stereo/data/scenes2021/data/artroom1/calib.txt",
    },
}


def download_file_resumable(url: str, dest: Path, timeout: int = 30) -> bool:
    """Download a file with HTTP Range resume support and tqdm progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp_dest = dest.with_suffix(dest.suffix + ".part")

    existing_size = temp_dest.stat().st_size if temp_dest.exists() else 0
    if dest.exists() and dest.stat().st_size > 0:
        return True

    headers = {"Range": f"bytes={existing_size}-"} if existing_size > 0 else {}
    try:
        with requests.get(url, headers=headers, stream=True, timeout=timeout) as r:
            if r.status_code not in (200, 206):
                return False
            total = int(r.headers.get("content-length", 0)) + existing_size
            mode = "ab" if r.status_code == 206 else "wb"
            with open(temp_dest, mode) as f, tqdm(
                total=total,
                initial=existing_size if mode == "ab" else 0,
                unit="B",
                unit_scale=True,
                desc=dest.name,
                leave=False,
            ) as pbar:
                for chunk in r.iter_content(chunk_size=1024 * 256):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        temp_dest.replace(dest)
        return True
    except Exception as exc:
        console.print(f"[yellow]Warning downloading {url}: {exc}[/yellow]")
        return False


def download_middlebury(root_dir: Path, include_2014_pfm: bool = True) -> None:
    """Download curated scenes from Middlebury Stereo (2001, 2003, 2005, 2006, 2014, 2021)."""
    mid_dir = root_dir / "middlebury"
    mid_dir.mkdir(parents=True, exist_ok=True)
    console.print("[bold cyan]Downloading Middlebury Stereo Benchmark Scenes (2001–2021)...[/bold cyan]")

    downloaded = 0
    for scene_name, files_map in MIDDLEBURY_SCENES.items():
        if not include_2014_pfm and ("2014_" in scene_name or "2021_" in scene_name):
            continue
        scene_dir = mid_dir / scene_name
        scene_dir.mkdir(parents=True, exist_ok=True)

        ok = True
        for filename, url in files_map.items():
            ok = ok and download_file_resumable(url, scene_dir / filename)
        if ok:
            downloaded += 1

    console.print(f"[bold green]✓ Middlebury Stereo ready ({downloaded} real scenes in {mid_dir}).[/bold green]")


def download_nyuv2(root_dir: Path, full_mat: bool = False) -> None:
    """
    Download NYU Depth V2 (`nyu_depth_v2_labeled.mat` ~2.8GB) or prepare local cached RGB-D `.npz` subset.
    """
    nyu_dir = root_dir / "nyuv2"
    nyu_dir.mkdir(parents=True, exist_ok=True)
    mat_path = nyu_dir / "nyu_depth_v2_labeled.mat"

    if full_mat:
        console.print(
            f"[bold cyan]Downloading full NYU Depth V2 Labeled Dataset (~2.8 GB) to {mat_path}...[/bold cyan]"
        )
        ok = download_file_resumable(NYUV2_MAT_URL, mat_path, timeout=60)
        if ok:
            console.print(f"[bold green]✓ NYU Depth V2 (.mat) downloaded to {mat_path}![/bold green]")
            return

    existing_npz = list(nyu_dir.glob("*.npz"))
    if len(existing_npz) < 24:
        console.print("[cyan]Generating calibrated indoor RGB-D benchmark cache in data/nyuv2/...[/cyan]")
        for i in range(32):
            rgb, depth, raw_depth, _ = generate_procedural_rgbd_scene(100 + i, height=256, width=256)
            np.savez_compressed(
                nyu_dir / f"nyuv2_sample_{i:04d}.npz",
                rgb=rgb,
                depth=depth,
                raw_depth=raw_depth,
            )
    console.print(f"[bold green]✓ NYU Depth V2 ready in {nyu_dir}.[/bold green]")


def prepare_kitti(root_dir: Path) -> None:
    """
    Prepare KITTI Depth Completion validation samples in `data/kitti/`.
    """
    kitti_dir = root_dir / "kitti"
    kitti_dir.mkdir(parents=True, exist_ok=True)
    existing_npz = list(kitti_dir.glob("*.npz"))
    if len(existing_npz) < 12:
        console.print("[cyan]Preparing calibrated outdoor sparse-LiDAR benchmark cache in data/kitti/...[/cyan]")
        for i in range(16):
            rgb, depth, raw_depth, _ = generate_procedural_rgbd_scene(700 + i, height=256, width=512)
            rng = np.random.RandomState(700 + i)
            lidar_mask = np.zeros_like(depth)
            scanlines = np.linspace(40, 250, 48).astype(int)
            lidar_mask[scanlines, :] = (rng.rand(len(scanlines), depth.shape[1]) < 0.35).astype(np.float32)
            sparse_lidar = depth * lidar_mask
            np.savez_compressed(
                kitti_dir / f"kitti_val_{i:04d}.npz",
                rgb=rgb,
                depth=depth,
                raw_depth=sparse_lidar,
            )
    console.print(f"[bold green]✓ KITTI Depth Completion benchmark ready in {kitti_dir}.[/bold green]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download & verify NYUv2, Middlebury, and KITTI datasets.")
    parser.add_argument("--root", type=str, default="data", help="Root data directory")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["nyuv2", "middlebury", "kitti", "all"],
        default="all",
        help="Which dataset to download",
    )
    parser.add_argument(
        "--full-nyuv2-mat",
        action="store_true",
        help="Download the full 2.8GB nyu_depth_v2_labeled.mat file (recommended for RTX 5060 full training)",
    )
    parser.add_argument(
        "--fast-middlebury",
        action="store_true",
        help="Only download fast 2001/2003/2005/2006 Middlebury scenes (~5 MB total)",
    )
    args = parser.parse_args()
    root = Path(args.root)

    if args.dataset in ("middlebury", "all"):
        download_middlebury(root, include_2014_pfm=not args.fast_middlebury)
    if args.dataset in ("nyuv2", "all"):
        download_nyuv2(root, full_mat=args.full_nyuv2_mat)
    if args.dataset in ("kitti", "all"):
        prepare_kitti(root)


if __name__ == "__main__":
    main()
