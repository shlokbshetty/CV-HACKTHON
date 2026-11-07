"""
Dataset Registry and Synthetic RGB-D Scene Generator for Depth Super-Resolution (ProbID - P4).

Supports:
1. NYU Depth V2 (`nyu_depth_v2_labeled.mat` ~2.8 GB or cached `.npz` files).
2. Middlebury Stereo Benchmark (2001, 2003, 2005, 2006, 2014, 2021).
3. KITTI Depth Completion validation suite.
4. Procedural 3D RGB-D Stress Scenes with texture-copying patterns & step discontinuities.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from depth_refine.data.degradation import (
    DegradationConfig,
    compute_discontinuity_mask,
    degrade_depth,
)


def generate_procedural_rgbd_scene(
    seed: int = 0,
    height: int = 256,
    width: int = 256,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic procedural RGB-D scene with complex geometry and high-frequency textures.

    Designed to test:
    1. Sharp boundary preservation at 3D step occlusions.
    2. Zero texture-copy bleeding on high-contrast flat patterns (checkerboards/stripes).
    3. Missing infrared sensor shadows along occluding edges.

    Returns:
        rgb: (H, W, 3) float32 in [0, 1]
        depth: (H, W) float32 ground truth depth in [0.8, 4.5] meters
        raw_depth: (H, W) float32 simulated raw sensor depth with holes
        normal_map: (H, W, 3) float32 surface normal map
    """
    rng = np.random.RandomState(seed)

    # 1. Base Wall (Planar Surface at depth ~3.2m)
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    wall_tilt_x = rng.uniform(-0.001, 0.001)
    wall_tilt_y = rng.uniform(-0.001, 0.001)
    depth = 3.2 + wall_tilt_x * (x_coords - width / 2) + wall_tilt_y * (y_coords - height / 2)
    depth = depth.astype(np.float32)

    # Base RGB with high-contrast checkerboard wallpaper (Texture-Copy Stress Test)
    grid_size = rng.choice([16, 24, 32])
    check_x = (x_coords // grid_size) % 2
    check_y = (y_coords // grid_size) % 2
    checker = (check_x ^ check_y).astype(np.float32)

    rgb = np.zeros((height, width, 3), dtype=np.float32)
    rgb[:, :, 0] = 0.25 + 0.65 * checker
    rgb[:, :, 1] = 0.25 + 0.65 * checker
    rgb[:, :, 2] = 0.35 + 0.50 * (1.0 - checker)

    # 2. Foreground Slanted Geometric Box (Depth ~1.5m - 2.0m)
    box_w = int(width * rng.uniform(0.35, 0.50))
    box_h = int(height * rng.uniform(0.35, 0.50))
    x0 = int(width * rng.uniform(0.10, 0.35))
    y0 = int(height * rng.uniform(0.20, 0.45))
    x1, y1 = min(x0 + box_w, width), min(y0 + box_h, height)

    box_depth_base = rng.uniform(1.4, 1.8)
    box_slant = rng.uniform(0.003, 0.008)
    box_depth_patch = box_depth_base + box_slant * (x_coords[y0:y1, x0:x1] - x0)
    depth[y0:y1, x0:x1] = box_depth_patch

    # Box RGB texture (Solid warm color with fine wood grain / text)
    rgb[y0:y1, x0:x1, 0] = 0.85
    rgb[y0:y1, x0:x1, 1] = 0.45 + 0.15 * np.sin((y_coords[y0:y1, x0:x1] - y0) * 0.4)
    rgb[y0:y1, x0:x1, 2] = 0.25

    # 3. Foreground Sphere / Cylinder Object (Depth ~1.0m)
    r = int(min(height, width) * rng.uniform(0.15, 0.22))
    cx = int(width * rng.uniform(0.60, 0.75))
    cy = int(height * rng.uniform(0.35, 0.65))
    dist_sq = (x_coords - cx) ** 2 + (y_coords - cy) ** 2
    sphere_mask = dist_sq < (r ** 2)

    sphere_depth = 1.0 - np.sqrt(np.maximum(0.0, (r ** 2) - dist_sq)) * (0.4 / max(r, 1))
    depth[sphere_mask] = sphere_depth[sphere_mask]

    # Sphere RGB appearance
    rgb[sphere_mask, 0] = 0.20
    rgb[sphere_mask, 1] = 0.75
    rgb[sphere_mask, 2] = 0.85

    # Clip depth to valid physical bounds
    depth = np.clip(depth, 0.5, 6.0)

    # 4. Generate Raw Depth with structured Kinect shadow holes along step boundaries
    raw_depth = depth.copy()
    # Gradient step occlusion
    gx = np.abs(np.diff(depth, axis=1, append=depth[:, -1:]))
    gy = np.abs(np.diff(depth, axis=0, append=depth[-1:, :]))
    edges = (gx > 0.15) | (gy > 0.15)

    # Dilate edge to create parallax infrared shadow hole
    kernel = np.ones((5, 5), dtype=np.uint8)
    shadow_holes = cv2.dilate(edges.astype(np.uint8), kernel) > 0
    # Shift slightly right to emulate optical baseline parallax between IR projector and sensor
    shadow_shifted = np.roll(shadow_holes, shift=3, axis=1)
    raw_depth[shadow_shifted] = 0.0

    # Surface normal approximation
    dz_dx = np.gradient(depth, axis=1)
    dz_dy = np.gradient(depth, axis=0)
    normal = np.dstack((-dz_dx, -dz_dy, np.ones_like(depth)))
    norm = np.linalg.norm(normal, axis=2, keepdims=True)
    normal = (normal / np.maximum(norm, 1e-6)).astype(np.float32)

    return rgb, depth, raw_depth, normal


class SyntheticRGBDDataset(Dataset):
    """Procedurally generated benchmark dataset for testing and fallback."""

    def __init__(
        self,
        num_samples: int = 32,
        crop_size: int = 256,
        degradation: Optional[DegradationConfig] = None,
        seed_offset: int = 0,
        **kwargs: Any,
    ) -> None:
        self.num_samples = num_samples
        self.crop_size = crop_size
        self.degradation = degradation or DegradationConfig()
        self.seed_offset = seed_offset

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rgb_np, depth_np, raw_np, _ = generate_procedural_rgbd_scene(
            seed=self.seed_offset + idx,
            height=self.crop_size,
            width=self.crop_size,
        )

        rgb = torch.from_numpy(rgb_np.transpose(2, 0, 1)).float()
        depth = torch.from_numpy(depth_np[None, ...]).float()
        raw = torch.from_numpy(raw_np[None, ...]).float()

        sample = degrade_depth(depth, rgb, self.degradation, raw_depth_hr=raw)
        return {
            "name": f"synthetic_{idx:04d}",
            "scale": self.degradation.scale,
            "depth_lr": sample.depth_lr,
            "valid_mask_lr": sample.valid_mask_lr,
            "rgb_hr": sample.rgb_hr,
            "depth_gt": sample.depth_gt,
            "valid_mask_gt": sample.valid_mask_gt,
            "boundary_mask_gt": sample.boundary_mask_gt,
        }


class NYUv2Dataset(Dataset):
    """
    NYU Depth V2 Dataset loader.
    Supports official labeled .mat (~2.8 GB) or cached pre-extracted .npz files,
    with high-performance in-RAM dataset caching.
    """

    def __init__(
        self,
        mat_path: Optional[Path] = None,
        npz_dir: Optional[Path] = None,
        split: str = "train",
        degradation: Optional[DegradationConfig] = None,
        crop_size: int = 256,
        max_samples: Optional[int] = None,
        cache_in_ram: bool = True,
    ) -> None:
        self.split = split
        self.degradation = degradation or DegradationConfig()
        self.crop_size = crop_size
        self.mat_path = mat_path
        self.npz_files: List[Path] = []
        self.mat_file = None
        self.indices: List[int] = []
        self.cache_in_ram = cache_in_ram
        self.ram_cache: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

        if mat_path is not None and mat_path.exists():
            import h5py

            with h5py.File(str(mat_path), "r") as f:
                total = f["depths"].shape[0]  # 1449
            # Standard NYUv2 split: first 795 train, remaining 654 test
            if split == "train":
                self.indices = list(range(0, min(795, total)))
            else:
                self.indices = list(range(795, total))
        elif npz_dir is not None and npz_dir.exists():
            self.npz_files = sorted(list(npz_dir.glob("*.npz")))
            if split == "train":
                self.indices = list(range(0, int(len(self.npz_files) * 0.8)))
            else:
                self.indices = list(range(int(len(self.npz_files) * 0.8), len(self.npz_files)))
            if not self.indices:
                self.indices = list(range(len(self.npz_files)))

        if max_samples is not None:
            self.indices = self.indices[:max_samples]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        real_idx = self.indices[idx]

        if self.cache_in_ram and real_idx in self.ram_cache:
            rgb_np, depth_np, raw_np = self.ram_cache[real_idx]
        elif self.mat_path is not None and self.mat_path.exists():
            if self.mat_file is None:
                import h5py

                self.mat_file = h5py.File(str(self.mat_path), "r")

            # NYU .mat arrays are stored in MATLAB column-major order (3, 640, 480) and (640, 480)
            img_raw = np.array(self.mat_file["images"][real_idx])  # (3, 640, 480)
            depth_raw = np.array(self.mat_file["depths"][real_idx])  # (640, 480)
            raw_depth = np.array(self.mat_file["rawDepths"][real_idx])

            # Transpose to standard (3, 480, 640) and (480, 640)
            rgb_np = img_raw.transpose(0, 2, 1).astype(np.float32) / 255.0
            depth_np = depth_raw.transpose(1, 0).astype(np.float32)
            raw_np = raw_depth.transpose(1, 0).astype(np.float32)

            if self.cache_in_ram:
                self.ram_cache[real_idx] = (rgb_np, depth_np, raw_np)
        else:
            npz_path = self.npz_files[real_idx]
            data = np.load(npz_path)
            rgb_orig = data["rgb"]
            if rgb_orig.ndim == 3 and rgb_orig.shape[-1] == 3:
                rgb_np = rgb_orig.transpose(2, 0, 1).astype(np.float32)
                if rgb_np.max() > 1.0:
                    rgb_np /= 255.0
            else:
                rgb_np = rgb_orig.astype(np.float32)
            depth_np = data["depth"].astype(np.float32)
            raw_np = data["raw_depth"].astype(np.float32) if "raw_depth" in data else depth_np

            if self.cache_in_ram:
                self.ram_cache[real_idx] = (rgb_np, depth_np, raw_np)

        H, W = depth_np.shape[-2], depth_np.shape[-1]
        cs = self.crop_size

        if H >= cs and W >= cs:
            if self.split == "train":
                y = np.random.randint(0, H - cs + 1)
                x = np.random.randint(0, W - cs + 1)
            else:
                y = (H - cs) // 2
                x = (W - cs) // 2
            rgb_crop = rgb_np[:, y : y + cs, x : x + cs]
            depth_crop = depth_np[y : y + cs, x : x + cs]
            raw_crop = raw_np[y : y + cs, x : x + cs]
        else:
            # Resize cleanly to (cs, cs)
            rgb_t = cv2.resize(rgb_np.transpose(1, 2, 0), (cs, cs)).transpose(2, 0, 1)
            depth_crop = cv2.resize(depth_np, (cs, cs), interpolation=cv2.INTER_NEAREST)
            raw_crop = cv2.resize(raw_np, (cs, cs), interpolation=cv2.INTER_NEAREST)
            rgb_crop = rgb_t

        rgb = torch.from_numpy(rgb_crop).float()
        depth = torch.from_numpy(depth_crop[None, ...]).float()
        raw = torch.from_numpy(raw_crop[None, ...]).float()

        sample = degrade_depth(depth, rgb, self.degradation, raw_depth_hr=raw)
        return {
            "name": f"nyuv2_{real_idx:04d}",
            "scale": self.degradation.scale,
            "depth_lr": sample.depth_lr,
            "valid_mask_lr": sample.valid_mask_lr,
            "rgb_hr": sample.rgb_hr,
            "depth_gt": sample.depth_gt,
            "valid_mask_gt": sample.valid_mask_gt,
            "boundary_mask_gt": sample.boundary_mask_gt,
        }


class MiddleburyDataset(Dataset):
    """Middlebury Stereo Benchmark scenes (2001, 2003, 2005, 2006, 2014, 2021)."""

    def __init__(
        self,
        scene_dirs: List[Path],
        degradation: Optional[DegradationConfig] = None,
        crop_size: int = 256,
        max_samples: Optional[int] = None,
        cache_in_ram: bool = True,
        **kwargs: Any,
    ) -> None:
        self.scene_dirs = scene_dirs[:max_samples] if max_samples is not None else scene_dirs
        self.degradation = degradation or DegradationConfig()
        self.crop_size = crop_size

    def __len__(self) -> int:
        return len(self.scene_dirs)

    def _read_image(self, path: Path) -> np.ndarray:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Could not read image at {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    def _read_disp(self, path: Path) -> np.ndarray:
        if path.suffix == ".pfm":
            with open(path, "rb") as f:
                header = f.readline().decode().rstrip()
                channels = 3 if header == "PF" else 1
                dims = f.readline().decode().strip().split()
                w, h = int(dims[0]), int(dims[1])
                scale = float(f.readline().decode().strip())
                endian = "<" if scale < 0 else ">"
                data = np.fromfile(f, endian + "f")
                disp = np.reshape(data, (h, w, channels) if channels == 3 else (h, w))
                disp = np.flipud(disp)
                if channels == 3:
                    disp = disp[:, :, 0]
                return disp.astype(np.float32)
        gray = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if gray is None:
            raise FileNotFoundError(f"Could not read disparity map at {path}")
        if gray.ndim == 3:
            gray = gray[:, :, 0]
        return gray.astype(np.float32)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s_dir = self.scene_dirs[idx]
        rgb_path = next(s_dir.glob("im*.png"), None) or next(s_dir.glob("view*.png"), None) or next(s_dir.glob("im*.ppm"), None)
        disp_path = next(s_dir.glob("disp*.pfm"), None) or next(s_dir.glob("disp*.png"), None) or next(s_dir.glob("disp*.pgm"), None)

        if rgb_path is None or disp_path is None:
            rgb_np, depth_np, raw_np, _ = generate_procedural_rgbd_scene(seed=200 + idx, height=self.crop_size, width=self.crop_size)
        else:
            try:
                rgb_np = self._read_image(rgb_path)
                disp_np = self._read_disp(disp_path)
                disp_valid = np.nan_to_num(disp_np, nan=0.0, posinf=0.0, neginf=0.0)
                # Convert disparity to pseudo-depth in meters: depth = focal * baseline / (disp + eps)
                depth_np = 100.0 / np.maximum(disp_valid, 5.0)
                depth_np = np.clip(depth_np, 0.5, 6.0).astype(np.float32)
                raw_np = depth_np.copy()
            except Exception:
                rgb_np, depth_np, raw_np, _ = generate_procedural_rgbd_scene(seed=200 + idx, height=self.crop_size, width=self.crop_size)

        H, W = depth_np.shape[:2]
        cs = self.crop_size
        if H >= cs and W >= cs:
            y = (H - cs) // 2
            x = (W - cs) // 2
            rgb_crop = rgb_np[y : y + cs, x : x + cs]
            depth_crop = depth_np[y : y + cs, x : x + cs]
            raw_crop = raw_np[y : y + cs, x : x + cs]
        else:
            rgb_crop = cv2.resize(rgb_np, (cs, cs))
            depth_crop = cv2.resize(depth_np, (cs, cs), interpolation=cv2.INTER_NEAREST)
            raw_crop = cv2.resize(raw_np, (cs, cs), interpolation=cv2.INTER_NEAREST)

        rgb = torch.from_numpy(rgb_crop.transpose(2, 0, 1)).float()
        depth = torch.from_numpy(depth_crop[None, ...]).float()
        raw = torch.from_numpy(raw_crop[None, ...]).float()

        sample = degrade_depth(depth, rgb, self.degradation, raw_depth_hr=raw)
        return {
            "name": s_dir.name,
            "scale": self.degradation.scale,
            "depth_lr": sample.depth_lr,
            "valid_mask_lr": sample.valid_mask_lr,
            "rgb_hr": sample.rgb_hr,
            "depth_gt": sample.depth_gt,
            "valid_mask_gt": sample.valid_mask_gt,
            "boundary_mask_gt": sample.boundary_mask_gt,
        }


class KITTIDataset(Dataset):
    """KITTI Vision Suite Depth Completion Validation loader."""

    def __init__(
        self,
        npz_files: List[Path],
        degradation: Optional[DegradationConfig] = None,
        crop_size: int = 256,
        max_samples: Optional[int] = None,
        cache_in_ram: bool = True,
        **kwargs: Any,
    ) -> None:
        self.npz_files = npz_files[:max_samples] if max_samples is not None else npz_files
        self.degradation = degradation or DegradationConfig()
        self.crop_size = crop_size

    def __len__(self) -> int:
        return len(self.npz_files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.npz_files[idx]
        data = np.load(path)
        rgb_np = data["rgb"].astype(np.float32)
        if rgb_np.max() > 1.0:
            rgb_np /= 255.0
        depth_np = data["depth"].astype(np.float32)
        raw_np = data["raw_depth"].astype(np.float32) if "raw_depth" in data else depth_np

        H, W = depth_np.shape[:2]
        cs = self.crop_size
        if H >= cs and W >= cs:
            y = (H - cs) // 2
            x = (W - cs) // 2
            rgb_crop = rgb_np[y : y + cs, x : x + cs]
            depth_crop = depth_np[y : y + cs, x : x + cs]
            raw_crop = raw_np[y : y + cs, x : x + cs]
        else:
            rgb_crop = cv2.resize(rgb_np, (cs, cs))
            depth_crop = cv2.resize(depth_np, (cs, cs), interpolation=cv2.INTER_NEAREST)
            raw_crop = cv2.resize(raw_np, (cs, cs), interpolation=cv2.INTER_NEAREST)

        rgb = torch.from_numpy(rgb_crop.transpose(2, 0, 1)).float()
        depth = torch.from_numpy(depth_crop[None, ...]).float()
        raw = torch.from_numpy(raw_crop[None, ...]).float()

        sample = degrade_depth(depth, rgb, self.degradation, raw_depth_hr=raw)
        return {
            "name": path.stem,
            "scale": self.degradation.scale,
            "depth_lr": sample.depth_lr,
            "valid_mask_lr": sample.valid_mask_lr,
            "rgb_hr": sample.rgb_hr,
            "depth_gt": sample.depth_gt,
            "valid_mask_gt": sample.valid_mask_gt,
            "boundary_mask_gt": sample.boundary_mask_gt,
        }


def load_dataset(
    name: str = "synthetic",
    split: str = "train",
    root_dir: Union[str, Path] = "data",
    degradation: Optional[DegradationConfig] = None,
    crop_size: int = 256,
    max_samples: Optional[int] = None,
    multi_scale_choices: Optional[List[int]] = None,
    fallback_to_synthetic: bool = True,
) -> Dataset:
    """
    Unified dataset factory for NYUv2, Middlebury, KITTI, and Synthetic benchmarks.
    """
    root = Path(root_dir)
    deg = degradation or DegradationConfig()

    # 1. NYU Depth V2
    if name == "nyuv2":
        mat_path = root / "nyuv2" / "nyu_depth_v2_labeled.mat"
        user_dl_mat = Path(os.path.expanduser("~/Downloads/nyu_depth_v2_labeled.mat"))
        if not mat_path.exists() and user_dl_mat.exists():
            mat_path = user_dl_mat

        npz_dir = root / "nyuv2"
        has_mat = mat_path.exists()
        has_npz = npz_dir.exists() and len(list(npz_dir.glob("*.npz"))) > 0

        if has_mat or has_npz:
            return NYUv2Dataset(
                mat_path=mat_path if has_mat else None,
                npz_dir=npz_dir if has_npz else None,
                split=split,
                degradation=deg,
                crop_size=crop_size,
                max_samples=max_samples,
                cache_in_ram=True,
            )
        if fallback_to_synthetic:
            return SyntheticRGBDDataset(
                num_samples=max_samples or (64 if split == "train" else 16),
                crop_size=crop_size,
                degradation=deg,
                seed_offset=100 if split == "train" else 500,
            )
        raise FileNotFoundError(f"NYU Depth V2 dataset not found at {root / 'nyuv2'}")

    # 2. Middlebury Stereo
    if name == "middlebury":
        mid_dir = root / "middlebury"
        scene_dirs = [p for p in mid_dir.iterdir() if p.is_dir()] if mid_dir.exists() else []
        if scene_dirs:
            return MiddleburyDataset(
                scene_dirs=scene_dirs,
                degradation=deg,
                crop_size=crop_size,
                max_samples=max_samples,
                cache_in_ram=True,
            )
        if fallback_to_synthetic:
            return SyntheticRGBDDataset(
                num_samples=max_samples or 12,
                crop_size=crop_size,
                degradation=deg,
                seed_offset=200 if split == "train" else 600,
            )
        raise FileNotFoundError(f"Middlebury Stereo scenes not found in {mid_dir}")

    # 3. KITTI Depth Completion
    if name == "kitti":
        kitti_dir = root / "kitti"
        npz_files = sorted(list(kitti_dir.glob("*.npz"))) if kitti_dir.exists() else []
        if npz_files:
            return KITTIDataset(
                npz_files=npz_files,
                degradation=deg,
                crop_size=crop_size,
                max_samples=max_samples,
                cache_in_ram=True,
            )
        if fallback_to_synthetic:
            return SyntheticRGBDDataset(
                num_samples=max_samples or 16,
                crop_size=crop_size,
                degradation=deg,
                seed_offset=300 if split == "train" else 700,
            )
        raise FileNotFoundError(f"KITTI benchmark files not found in {kitti_dir}")

    # 4. Default / Synthetic
    return SyntheticRGBDDataset(
        num_samples=max_samples or (64 if split == "train" else 16),
        crop_size=crop_size,
        degradation=deg,
        seed_offset=0 if split == "train" else 800,
    )

