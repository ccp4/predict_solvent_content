"""
Compute crystallographic Patterson descriptors for tree-ensemble
solvent content prediction (Random Forest / HistGradientBoosting / LightGBM).

Features are derived from the Patterson density distribution itself:
  - Single unit-cell & Patterson-ASU non-origin statistical moments, quantiles,
     and spike-winsorized fourth moments (immune to 128 A box-tiling bias and
     robust to mis-assigned space groups / tNCS peaks).
  - Radially binned spherical shell profiles (4-10, 10-20, 20-35, 35-60 A),
     intermolecular-to-intramolecular variance & MAD contrast ratios
     (e.g. Var(P_35-60) / Var(P_5-15)), and radial log-variance decay slope.
  - Thresholded low-density trough (P < -0.5 sigma) and quiet-zone (|P| < 0.2 sigma)
     volume fractions plus 3D Euclidean Distance Transform (EDT) channel radii.
  - 3D Sobel gradient magnitudes and vector-ridge 3D skeleton branch topology.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from glob import glob
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
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

_NEIGHBOR_KERNEL_3D = np.ones((3, 3, 3), dtype=np.int16)
_NEIGHBOR_KERNEL_3D[1, 1, 1] = 0

RADIAL_SHELLS: Tuple[Tuple[str, float, float], ...] = (
    ("4_10", 4.0, 10.0),
    ("10_20", 10.0, 20.0),
    ("20_35", 20.0, 35.0),
    ("35_60", 35.0, 60.0),
)

PATTERSON_ONLY_COLUMNS: Tuple[str, ...] = (
    # Global / Unit-cell & ASU non-origin moments (tiling-free)
    "origin_to_sigma",
    "max_secondary_peak",
    "uc_skew",
    "uc_kurtosis",
    "uc_robust_kurtosis",
    "asu_skew",
    "asu_kurtosis",
    "uc_iqr",
    "uc_p05",
    "uc_p10",
    "uc_p90",
    "uc_p99",
    # Cartesian box summary
    "avg",
    "med",
    "std",
    "skew",
    "kurtosis",
    # Radial spherical shells (4-10, 10-20, 20-35, 35-60 A)
    "var_4_10",
    "skew_4_10",
    "kurt_4_10",
    "flat_frac_4_10",
    "var_10_20",
    "skew_10_20",
    "kurt_10_20",
    "flat_frac_10_20",
    "var_20_35",
    "skew_20_35",
    "kurt_20_35",
    "flat_frac_20_35",
    "var_35_60",
    "skew_35_60",
    "kurt_35_60",
    "flat_frac_35_60",
    # Cross-shell intermolecular / intramolecular contrast ratios & decay
    "var_ratio_35_60_to_5_15",
    "var_ratio_35_60_to_4_10",
    "var_ratio_20_35_to_5_15",
    "mad_ratio_35_60_to_5_15",
    "radial_log_var_slope",
    # Void / quiet-zone fractions & 3D EDT channel radii
    "frac_below_m05",
    "frac_below_m10",
    "frac_flat_02",
    "frac_flat_035",
    "frac_above_p10",
    "mean_void_radius",
    "p50_void_radius",
    "p90_void_radius",
    "p50_flat_radius",
    "p90_flat_radius",
    # 3D Sobel gradient & ridge skeleton topology
    "mean_grad",
    "max_grad",
    "var_grad",
    "per_grad",
    "skeleton_length",
    "num_branch_points",
    "branch_density",
)

# Standard data-processing / indexing / scaling metadata available before phasing
# (automatically populated from .mtz / .cif / .map headers; NaN when absent)
DIFFRACTION_METADATA_COLUMNS: Tuple[str, ...] = (
    "native_d_min",
    "d_max",
    "wilson_b_est",
    "low_res_intensity_ratio",
    "acentric_e2_minus_1",
    "mean_e4",
    "log_cell_volume",
    "log_asu_volume",
    "asu_equiv_radius",
    "cell_min_len",
    "cell_mid_len",
    "cell_max_len",
    "cell_anisotropy",
    "spacegroup_number",
    "point_group_order",
)

DESCRIPTOR_COLUMNS: Tuple[str, ...] = PATTERSON_ONLY_COLUMNS + DIFFRACTION_METADATA_COLUMNS


def calc_kurtosis(arr: np.ndarray) -> float:
    if arr.size < 4 or float(np.std(arr)) < 1e-12:
        return 0.0
    val = float(scipy_kurtosis(arr, nan_policy="omit"))
    return val if math.isfinite(val) else 0.0


def calc_skew(arr: np.ndarray) -> float:
    if arr.size < 3 or float(np.std(arr)) < 1e-12:
        return 0.0
    val = float(scipy_skew(arr, nan_policy="omit"))
    return val if math.isfinite(val) else 0.0


def calc_mean(arr: np.ndarray) -> float:
    return float(np.mean(arr))


def calc_median(arr: np.ndarray) -> float:
    return float(np.median(arr))


def calc_std(arr: np.ndarray) -> float:
    return float(np.std(arr))


@lru_cache(maxsize=8)
def _cached_radial_masks(
    grid_size: Tuple[int, int, int],
    grid_spacing: float,
    origin_mask_radius: float,
) -> Dict[str, object]:
    """Precompute 3D radial distance grid and concentric spherical shell masks."""
    nx, ny, nz = grid_size
    cx = (np.arange(nx, dtype=np.float32) - 0.5 * nx) * grid_spacing
    cy = (np.arange(ny, dtype=np.float32) - 0.5 * ny) * grid_spacing
    cz = (np.arange(nz, dtype=np.float32) - 0.5 * nz) * grid_spacing
    r = np.sqrt(
        cx[:, None, None] ** 2 + cy[None, :, None] ** 2 + cz[None, None, :] ** 2
    )

    max_inscribed_r = 0.5 * min(nx, ny, nz) * grid_spacing
    shells: Dict[str, np.ndarray] = {}
    for label, r_lo, r_hi in RADIAL_SHELLS:
        eff_hi = min(r_hi, max_inscribed_r)
        eff_lo = min(r_lo, eff_hi)
        shells[label] = (r >= eff_lo) & (r < eff_hi)

    # Reference intramolecular shell (5-15 A) for roadmap ratio Var(35-60) / Var(5-15)
    shells["5_15"] = (r >= min(5.0, max_inscribed_r)) & (r < min(15.0, max_inscribed_r))
    shells["non_origin"] = r >= origin_mask_radius

    # Fine 5 A rings from 5 A up to min(60 A, max_inscribed_r) for radial log-variance decay slope
    ring_masks: List[Tuple[float, np.ndarray]] = []
    r_start = 5.0
    while r_start + 3.0 <= max_inscribed_r and r_start < 60.0:
        r_end = min(r_start + 5.0, max_inscribed_r, 60.0)
        mask = (r >= r_start) & (r < r_end)
        if np.any(mask):
            ring_masks.append((0.5 * (r_start + r_end), mask))
        r_start += 5.0

    return {"shells": shells, "rings": ring_masks}


def calc_unit_cell_and_asu_descriptors(
    uc_vals: np.ndarray,
    asu_vals: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute tiling-free single-unit-cell and Patterson-ASU moments & quantiles.
    Includes winsorized kurtosis (`|P| <= 6 sigma`) so mis-indexed centering or
    strong tNCS/Harker spikes do not distort packing contrast.
    """
    if uc_vals is None or uc_vals.size == 0:
        return {
            "max_secondary_peak": 0.0,
            "uc_skew": 0.0,
            "uc_kurtosis": 0.0,
            "uc_robust_kurtosis": 0.0,
            "asu_skew": 0.0,
            "asu_kurtosis": 0.0,
            "uc_iqr": 0.0,
            "uc_p05": 0.0,
            "uc_p10": 0.0,
            "uc_p90": 0.0,
            "uc_p99": 0.0,
        }

    p05, p10, p25, p75, p90, p99 = np.percentile(uc_vals, [5, 10, 25, 75, 90, 99])
    clipped_uc = np.clip(uc_vals, -6.0, 6.0)
    asu_arr = asu_vals if (asu_vals is not None and asu_vals.size > 0) else uc_vals

    return {
        "max_secondary_peak": float(np.max(uc_vals)),
        "uc_skew": calc_skew(uc_vals),
        "uc_kurtosis": calc_kurtosis(uc_vals),
        "uc_robust_kurtosis": calc_kurtosis(clipped_uc),
        "asu_skew": calc_skew(asu_arr),
        "asu_kurtosis": calc_kurtosis(asu_arr),
        "uc_iqr": float(p75 - p25),
        "uc_p05": float(p05),
        "uc_p10": float(p10),
        "uc_p90": float(p90),
        "uc_p99": float(p99),
    }


def calc_radial_shell_descriptors(
    arr: np.ndarray,
    grid_spacing: float = DEFAULT_GRID_SPACING,
    origin_mask_radius: float = DEFAULT_ORIGIN_MASK_RADIUS,
) -> Dict[str, float]:
    """
    Compute shell-wise variance, skewness, kurtosis, quiet-zone fraction,
    and intermolecular-to-intramolecular contrast ratios across spherical shells.
    """
    masks_bundle = _cached_radial_masks(arr.shape, grid_spacing, origin_mask_radius)
    shells: Dict[str, np.ndarray] = masks_bundle["shells"]  # type: ignore[assignment]
    rings: List[Tuple[float, np.ndarray]] = masks_bundle["rings"]  # type: ignore[assignment]

    out: Dict[str, float] = {}
    for label, _, _ in RADIAL_SHELLS:
        m = shells[label]
        if np.any(m):
            vals = arr[m]
            out[f"var_{label}"] = float(np.var(vals, dtype=np.float64))
            out[f"skew_{label}"] = calc_skew(vals)
            out[f"kurt_{label}"] = calc_kurtosis(vals)
            out[f"flat_frac_{label}"] = float(np.mean(np.abs(vals) < 0.25))
        else:
            out[f"var_{label}"] = 0.0
            out[f"skew_{label}"] = 0.0
            out[f"kurt_{label}"] = 0.0
            out[f"flat_frac_{label}"] = 0.0

    m_5_15 = shells["5_15"]
    m_35_60 = shells["35_60"]
    if np.any(m_5_15):
        vals_5_15 = arr[m_5_15]
        var_5_15 = max(1e-8, float(np.var(vals_5_15, dtype=np.float64)))
        mad_5_15 = max(1e-8, float(np.mean(np.abs(vals_5_15), dtype=np.float64)))
    else:
        var_5_15 = 1.0
        mad_5_15 = 1.0

    var_4_10 = max(1e-8, out["var_4_10"])
    var_20_35 = out["var_20_35"]
    var_35_60 = out["var_35_60"]
    mad_35_60 = (
        float(np.mean(np.abs(arr[m_35_60]), dtype=np.float64))
        if np.any(m_35_60)
        else 0.0
    )

    out["var_ratio_35_60_to_5_15"] = var_35_60 / var_5_15
    out["var_ratio_35_60_to_4_10"] = var_35_60 / var_4_10
    out["var_ratio_20_35_to_5_15"] = var_20_35 / var_5_15
    out["mad_ratio_35_60_to_5_15"] = mad_35_60 / mad_5_15

    if len(rings) >= 2:
        radii_x = np.array([rc for rc, _ in rings], dtype=np.float64)
        log_vars_y = np.array(
            [
                math.log(max(1e-6, float(np.var(arr[rm], dtype=np.float64))))
                for _, rm in rings
            ],
            dtype=np.float64,
        )
        slope = float(np.polyfit(radii_x, log_vars_y, 1)[0])
    else:
        slope = 0.0
    out["radial_log_var_slope"] = slope
    return out


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


def calc_void_and_skeleton_descriptors(
    arr: np.ndarray,
    non_origin_mask: np.ndarray,
    ridge_sigma_threshold: float = 1.0,
    void_sigma_threshold: float = -0.5,
    flat_sigma_threshold: float = 0.25,
    grid_spacing: float = DEFAULT_GRID_SPACING,
) -> Dict[str, float]:
    """
    Compute void/quiet-zone volume fractions, 3D Euclidean Distance Transform
    channel radii (in Angstroms), and 3D ridge skeleton length & branch density.
    """
    valid_vals = arr[non_origin_mask] if np.any(non_origin_mask) else arr.ravel()
    frac_below_m05 = float(np.mean(valid_vals < -0.5))
    frac_below_m10 = float(np.mean(valid_vals < -1.0))
    frac_flat_02 = float(np.mean(np.abs(valid_vals) < 0.2))
    frac_flat_035 = float(np.mean(np.abs(valid_vals) < 0.35))
    frac_above_p10 = float(np.mean(valid_vals > 1.0))

    # 3D skeleton of strong interatomic vector ridges
    ridge_mask = (arr > ridge_sigma_threshold) & non_origin_mask
    if np.any(ridge_mask):
        skel_bool = skeletonize(ridge_mask) > 0
        skeleton_voxels = int(np.sum(skel_bool))
        skeleton_length = float(skeleton_voxels) * grid_spacing
        neighbor_counts = scipy.ndimage.convolve(
            skel_bool.astype(np.int16),
            _NEIGHBOR_KERNEL_3D,
            mode="constant",
            cval=0,
        )
        num_branch_points = int(np.sum(skel_bool & (neighbor_counts > 2)))
    else:
        skeleton_length = 0.0
        num_branch_points = 0

    branch_density = float(num_branch_points) / max(1.0, skeleton_length)

    # 3D EDT on low-density troughs (P < -0.5 sigma)
    void_mask = (arr < void_sigma_threshold) & non_origin_mask
    if np.any(void_mask) and not np.all(void_mask):
        void_edt = distance_transform_edt(void_mask, sampling=grid_spacing)
        v_rad = void_edt[void_mask]
        mean_void_r = float(np.mean(v_rad))
        p50_void_r, p90_void_r = (float(x) for x in np.percentile(v_rad, [50, 90]))
    else:
        mean_void_r = p50_void_r = p90_void_r = 0.0

    # 3D EDT on quiet bulk-solvent vector-free zones (|P| < 0.25 sigma, excluding masked origin)
    flat_mask = (np.abs(arr) < flat_sigma_threshold) & non_origin_mask
    if np.any(flat_mask) and not np.all(flat_mask):
        flat_edt = distance_transform_edt(flat_mask, sampling=grid_spacing)
        f_rad = flat_edt[flat_mask]
        p50_flat_r, p90_flat_r = (float(x) for x in np.percentile(f_rad, [50, 90]))
    else:
        p50_flat_r = p90_flat_r = 0.0

    return {
        "frac_below_m05": frac_below_m05,
        "frac_below_m10": frac_below_m10,
        "frac_flat_02": frac_flat_02,
        "frac_flat_035": frac_flat_035,
        "frac_above_p10": frac_above_p10,
        "mean_void_radius": mean_void_r,
        "p50_void_radius": p50_void_r,
        "p90_void_radius": p90_void_r,
        "p50_flat_radius": p50_flat_r,
        "p90_flat_radius": p90_flat_r,
        "skeleton_length": skeleton_length,
        "num_branch_points": float(num_branch_points),
        "branch_density": branch_density,
    }


def calc_skeleton_info(
    arr: np.ndarray,
    ridge_sigma_threshold: float = 1.0,
    void_sigma_threshold: float = -0.5,
    grid_spacing: float = DEFAULT_GRID_SPACING,
) -> Tuple[int, int, float, float]:
    """Backward-compatible helper returning `(skeleton_length, num_branch_points, mean_void_radius, p90_void_radius)`."""
    non_origin = np.ones(arr.shape, dtype=bool)
    stats = calc_void_and_skeleton_descriptors(
        arr,
        non_origin_mask=non_origin,
        ridge_sigma_threshold=ridge_sigma_threshold,
        void_sigma_threshold=void_sigma_threshold,
        grid_spacing=grid_spacing,
    )
    return (
        int(round(stats["skeleton_length"] / max(1e-6, grid_spacing))),
        int(stats["num_branch_points"]),
        stats["mean_void_radius"],
        stats["p90_void_radius"],
    )


def extract_descriptors_for_file(
    file_path: str,
    grid_size: Tuple[int, int, int] = DEFAULT_GRID_SIZE,
    grid_spacing: float = DEFAULT_GRID_SPACING,
    d_min: Optional[float] = DEFAULT_D_MIN,
    origin_mask_radius: float = DEFAULT_ORIGIN_MASK_RADIUS,
) -> Dict[str, float]:
    """Compute all 55 metadata-free Patterson descriptors for a map, MTZ, CIF, or NPY file."""
    res: PattersonResult = compute_patterson_from_file(
        file_path=file_path,
        grid_size=grid_size,
        grid_spacing=grid_spacing,
        d_min=d_min,
        origin_mask_radius=origin_mask_radius,
        return_unit_cell_values=True,
    )
    arr = res.array
    masks_bundle = _cached_radial_masks(arr.shape, grid_spacing, origin_mask_radius)
    non_origin_cart: np.ndarray = masks_bundle["shells"]["non_origin"]  # type: ignore[index]

    uc_stats = calc_unit_cell_and_asu_descriptors(
        res.unit_cell_non_origin, res.asu_non_origin
    )
    radial_stats = calc_radial_shell_descriptors(
        arr, grid_spacing=grid_spacing, origin_mask_radius=origin_mask_radius
    )
    morph_stats = calc_void_and_skeleton_descriptors(
        arr, non_origin_mask=non_origin_cart, grid_spacing=grid_spacing
    )
    mean_grad, max_grad, var_grad, per_grad = calc_gradient_info(arr)

    cart_vals = arr[non_origin_cart] if np.any(non_origin_cart) else arr.ravel()

    desc: Dict[str, float] = {
        "origin_to_sigma": res.origin_to_sigma,
        "avg": calc_mean(cart_vals),
        "med": calc_median(cart_vals),
        "std": calc_std(cart_vals),
        "skew": calc_skew(cart_vals),
        "kurtosis": calc_kurtosis(cart_vals),
        "mean_grad": mean_grad,
        "max_grad": max_grad,
        "var_grad": var_grad,
        "per_grad": per_grad,
        # Standard experimental indexing/scaling metadata (NaN if headerless .npy / .map)
        "native_d_min": res.native_d_min,
        "d_max": res.d_max,
        "wilson_b_est": res.wilson_b_est,
        "low_res_intensity_ratio": res.low_res_intensity_ratio,
        "acentric_e2_minus_1": res.acentric_e2_minus_1,
        "mean_e4": res.mean_e4,
        "log_cell_volume": math.log(res.cell_volume) if (math.isfinite(res.cell_volume) and res.cell_volume > 1.0) else float("nan"),
        "log_asu_volume": math.log(res.asu_volume) if (math.isfinite(res.asu_volume) and res.asu_volume > 1.0) else float("nan"),
        "asu_equiv_radius": res.asu_equiv_radius,
        "cell_min_len": res.cell_min_len,
        "cell_mid_len": res.cell_mid_len,
        "cell_max_len": res.cell_max_len,
        "cell_anisotropy": res.cell_anisotropy,
        "spacegroup_number": float(res.spacegroup_number) if res.spacegroup_number > 0 else float("nan"),
        "point_group_order": float(res.point_group_order) if res.point_group_order > 0 else float("nan"),
    }
    desc.update(uc_stats)
    desc.update(radial_stats)
    desc.update(morph_stats)
    return desc


def _worker_extract(
    task: Tuple[str, Tuple[int, int, int], float, Optional[float], float],
) -> Tuple[str, Optional[Dict[str, float]], Optional[str]]:
    file_path, grid_size, grid_spacing, d_min, origin_mask_radius = task
    name = os.path.basename(file_path).split(".")[0]
    try:
        desc = extract_descriptors_for_file(
            file_path=file_path,
            grid_size=grid_size,
            grid_spacing=grid_spacing,
            d_min=d_min,
            origin_mask_radius=origin_mask_radius,
        )
        return name, desc, None
    except Exception as exc:
        return name, None, str(exc)


def _collect_input_files(input_spec: str) -> List[str]:
    p = Path(input_spec)
    if p.is_file() and p.suffix.lower() == ".txt":
        with open(p, "r", encoding="utf-8") as f:
            return [
                line.split()[-1]
                for line in f
                if line.strip() and not line.startswith("#")
            ]
    if p.is_file():
        return [str(p)]
    return sorted(glob(input_spec))


def _format_descriptor_row(name: str, desc: Dict[str, float]) -> str:
    vals_str = " ".join(f"{desc[col]:.6g}" for col in DESCRIPTOR_COLUMNS)
    return f"{name} {vals_str}\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate metadata-free unit-cell, radial-shell, and void-EDT Patterson descriptors."
    )
    parser.add_argument(
        "--input",
        default="/data1/ccp4/db/pdb_maps/*/*.gz",
        help="Glob pattern, single file (.map/.mtz/.cif/.npy), or .txt file list.",
    )
    parser.add_argument(
        "--output",
        default="additional_descriptors.txt",
        help="Output whitespace-delimited descriptor table (default: additional_descriptors.txt).",
    )
    parser.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE[0])
    parser.add_argument("--grid-spacing", type=float, default=DEFAULT_GRID_SPACING)
    parser.add_argument("--d-min", type=float, default=DEFAULT_D_MIN)
    parser.add_argument(
        "--origin-mask-radius", type=float, default=DEFAULT_ORIGIN_MASK_RADIUS
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of parallel worker processes."
    )
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

    pending_files = [
        fp
        for fp in map_files
        if os.path.basename(fp).split(".")[0] not in existing_names
    ]
    grid_size = (args.grid_size, args.grid_size, args.grid_size)
    d_min = args.d_min if (args.d_min is not None and args.d_min > 0) else None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as out_f:
        if write_header:
            out_f.write("name " + " ".join(DESCRIPTOR_COLUMNS) + "\n")

        tasks = [
            (fp, grid_size, args.grid_spacing, d_min, args.origin_mask_radius)
            for fp in pending_files
        ]
        if args.workers > 1 and len(tasks) > 1:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                for name, desc, err in pool.map(_worker_extract, tasks):
                    if desc is not None:
                        out_f.write(_format_descriptor_row(name, desc))
                        out_f.flush()
                    else:
                        print(f"{name} failed: {err}")
        else:
            for task in tasks:
                name, desc, err = _worker_extract(task)
                if desc is not None:
                    out_f.write(_format_descriptor_row(name, desc))
                    out_f.flush()
                else:
                    print(f"{name} failed: {err}")


if __name__ == "__main__":
    main()
