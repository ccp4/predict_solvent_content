"""
Compute statistical, 3D Sobel gradient, vector-ridge skeleton, and solvent-void
descriptors from origin-masked, normalized Patterson maps.
"""

import argparse
from glob import glob
import os
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import scipy.ndimage
from scipy.ndimage import distance_transform_edt
from scipy.stats import kurtosis as scipy_kurtosis
from scipy.stats import skew as scipy_skew
from skimage.morphology import skeletonize

from patterson import (
    DEFAULT_D_MIN,
    DEFAULT_GRID_SIZE,
    DEFAULT_GRID_SPACING,
    DEFAULT_ORIGIN_MASK_RADIUS,
    PattersonResult,
    compute_patterson_from_file,
    load_patterson_ccp4_map,
)

SOBEL_KERNEL_X = np.array(
    [
        [[1, 2, 1], [2, 4, 2], [1, 2, 1]],
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]],
    ],
    dtype=np.float32,
)

SOBEL_KERNEL_Y = np.array(
    [
        [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
        [[2, 0, -2], [4, 0, -4], [2, 0, -2]],
        [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
    ],
    dtype=np.float32,
)

SOBEL_KERNEL_Z = np.array(
    [
        [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
        [[2, 4, 2], [0, 0, 0], [-2, -4, -2]],
        [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
    ],
    dtype=np.float32,
)

# 26-connectivity neighbor kernel (center = 0, all 26 neighbors = 1)
_NEIGHBOR_KERNEL_3D = np.ones((3, 3, 3), dtype=np.int16)
_NEIGHBOR_KERNEL_3D[1, 1, 1] = 0


def calc_kurtosis(arr: np.ndarray) -> float:
    return float(scipy_kurtosis(arr, nan_policy="omit"))


def calc_skew(arr: np.ndarray) -> float:
    return float(scipy_skew(arr, nan_policy="omit"))


def calc_mean(arr: np.ndarray) -> float:
    return float(np.mean(arr))


def calc_median(arr: np.ndarray) -> float:
    return float(np.median(arr))


def calc_std(arr: np.ndarray) -> float:
    return float(np.std(arr))


def calc_gradient_info(arr: np.ndarray) -> Tuple[float, float, float, float]:
    gradient_x = scipy.ndimage.convolve(arr, SOBEL_KERNEL_X)
    gradient_y = scipy.ndimage.convolve(arr, SOBEL_KERNEL_Y)
    gradient_z = scipy.ndimage.convolve(arr, SOBEL_KERNEL_Z)

    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2 + gradient_z**2)

    return (
        float(np.mean(gradient_magnitude)),
        float(np.max(gradient_magnitude)),
        float(np.var(gradient_magnitude)),
        float(np.percentile(gradient_magnitude, 90)),
    )


def get_p_map_grid(file_path: str):
    grid, _ = load_patterson_ccp4_map(file_path)
    return grid


def get_p_map_corrected(
    file_path: str,
    grid_size: Tuple[int, int, int] = DEFAULT_GRID_SIZE,
    grid_spacing: float = DEFAULT_GRID_SPACING,
    d_min: Optional[float] = DEFAULT_D_MIN,
    origin_mask_radius: float = DEFAULT_ORIGIN_MASK_RADIUS,
) -> np.ndarray:
    res = compute_patterson_from_file(
        file_path=file_path,
        grid_size=grid_size,
        grid_spacing=grid_spacing,
        d_min=d_min,
        origin_mask_radius=origin_mask_radius,
    )
    return res.array


def calc_skeleton_info(
    arr: np.ndarray,
    ridge_sigma_threshold: float = 1.0,
    void_sigma_threshold: float = -0.5,
    grid_spacing: float = DEFAULT_GRID_SPACING,
) -> Tuple[int, int, float, float]:
    """
    Threshold normalized Patterson ridges (`arr > ridge_sigma_threshold`), compute
    3D skeleton length and true 26-connected topological branch points, and measure
    low-density void radii (`arr < void_sigma_threshold`) via Euclidean distance transform.
    """
    ridge_mask = arr > ridge_sigma_threshold
    if np.any(ridge_mask):
        skel_bool = skeletonize(ridge_mask) > 0
        skeleton_length = int(np.sum(skel_bool))
        neighbor_counts = scipy.ndimage.convolve(
            skel_bool.astype(np.int16),
            _NEIGHBOR_KERNEL_3D,
            mode="constant",
            cval=0,
        )
        num_branch_points = int(np.sum(skel_bool & (neighbor_counts > 2)))
    else:
        skeleton_length = 0
        num_branch_points = 0

    void_mask = arr < void_sigma_threshold
    if np.any(void_mask) and not np.all(void_mask):
        void_edt = distance_transform_edt(void_mask, sampling=grid_spacing)
        void_radii = void_edt[void_mask]
        mean_void_radius = float(np.mean(void_radii))
        p90_void_radius = float(np.percentile(void_radii, 90))
    else:
        mean_void_radius = 0.0
        p90_void_radius = 0.0

    return skeleton_length, num_branch_points, mean_void_radius, p90_void_radius


def extract_descriptors_for_file(
    file_path: str,
    grid_size: Tuple[int, int, int] = DEFAULT_GRID_SIZE,
    grid_spacing: float = DEFAULT_GRID_SPACING,
    d_min: Optional[float] = DEFAULT_D_MIN,
    origin_mask_radius: float = DEFAULT_ORIGIN_MASK_RADIUS,
) -> dict:
    """Compute all descriptors for a single map/MTZ/CIF file."""
    res: PattersonResult = compute_patterson_from_file(
        file_path=file_path,
        grid_size=grid_size,
        grid_spacing=grid_spacing,
        d_min=d_min,
        origin_mask_radius=origin_mask_radius,
    )
    arr = res.array
    skel_len, branch_pts, mean_void_r, p90_void_r = calc_skeleton_info(
        arr, grid_spacing=grid_spacing
    )
    mean_grad, max_grad, var_grad, per_grad = calc_gradient_info(arr)
    flat = arr.ravel()
    return {
        "origin_to_sigma": res.origin_to_sigma,
        "non_origin_std": res.non_origin_std,
        "avg": calc_mean(flat),
        "med": calc_median(flat),
        "std": calc_std(flat),
        "skew": calc_skew(flat),
        "kurtosis": calc_kurtosis(flat),
        "mean_grad": mean_grad,
        "max_grad": max_grad,
        "var_grad": var_grad,
        "per_grad": per_grad,
        "skeleton_length": skel_len,
        "num_branch_points": branch_pts,
        "mean_void_radius": mean_void_r,
        "p90_void_radius": p90_void_r,
    }


def _collect_input_files(input_spec: str) -> List[str]:
    p = Path(input_spec)
    if p.is_file() and p.suffix.lower() == ".txt":
        with open(p, "r", encoding="utf-8") as f:
            return [line.split()[-1] for line in f if line.strip()]
    if p.is_file():
        return [str(p)]
    return sorted(glob(input_spec))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate 3D Patterson map descriptors for tree-based solvent models."
    )
    parser.add_argument(
        "--input",
        default="/data1/ccp4/db/pdb_maps/*/*.gz",
        help="Glob pattern, single file, or .txt file list of maps/MTZ/CIFs.",
    )
    parser.add_argument(
        "--output",
        default="additional_descriptors.txt",
        help="Output whitespace-delimited descriptor table (default: additional_descriptors.txt).",
    )
    parser.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE[0])
    parser.add_argument("--grid-spacing", type=float, default=DEFAULT_GRID_SPACING)
    parser.add_argument("--d-min", type=float, default=DEFAULT_D_MIN)
    parser.add_argument("--origin-mask-radius", type=float, default=DEFAULT_ORIGIN_MASK_RADIUS)
    args = parser.parse_args()

    map_files = _collect_input_files(args.input)
    print(f"num map files {len(map_files)}")

    existing_names = set()
    output_path = Path(args.output)
    write_header = not output_path.exists() or output_path.stat().st_size == 0
    if not write_header:
        with open(output_path, "r", encoding="utf-8") as existing_f:
            for line in existing_f:
                parts = line.split()
                if parts and parts[0] != "name":
                    existing_names.add(parts[0])

    grid_size = (args.grid_size, args.grid_size, args.grid_size)
    d_min = args.d_min if (args.d_min is not None and args.d_min > 0) else None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as out_f:
        if write_header:
            out_f.write(
                "name origin_to_sigma non_origin_std avg med std skew kurtosis "
                "mean_grad max_grad var_grad per_grad skeleton_length "
                "num_branch_points mean_void_radius p90_void_radius\n"
            )

        for file_path in map_files:
            name = os.path.basename(file_path).split(".")[0]
            if name in existing_names:
                continue
            try:
                desc = extract_descriptors_for_file(
                    file_path=file_path,
                    grid_size=grid_size,
                    grid_spacing=args.grid_spacing,
                    d_min=d_min,
                    origin_mask_radius=args.origin_mask_radius,
                )
                row = (
                    f"{name} {desc['origin_to_sigma']:.6f} {desc['non_origin_std']:.6e} "
                    f"{desc['avg']:.6f} {desc['med']:.6f} {desc['std']:.6f} "
                    f"{desc['skew']:.6f} {desc['kurtosis']:.6f} {desc['mean_grad']:.6f} "
                    f"{desc['max_grad']:.6f} {desc['var_grad']:.6f} {desc['per_grad']:.6f} "
                    f"{desc['skeleton_length']} {desc['num_branch_points']} "
                    f"{desc['mean_void_radius']:.4f} {desc['p90_void_radius']:.4f}\n"
                )
                out_f.write(row)
                out_f.flush()
                existing_names.add(name)
            except Exception as exc:
                print(f"{name} failed: {exc}")


if __name__ == "__main__":
    main()
