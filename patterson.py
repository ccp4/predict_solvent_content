"""
Core crystallographic utilities for computing, loading, origin-masking,
normalizing, and interpolating 3D Patterson maps onto a centered Cartesian grid.

Key physical & numerical guarantees:
1. Pure-Python/Gemmi E^2 Patterson synthesis from .mtz or structure-factor .cif
   files (no external CCP4 ecalc/fft/cif2mtz subprocesses required).
2. Uniform high-resolution truncation (default d_min = 3.5 A) to prevent models
   from exploiting the PDB resolution-solvent correlation via peak sharpness.
3. Full crystallographic map setup (`setup=True`) when reading .map/.map.gz files,
   fixing axis permutations and expanding Patterson symmetry across the unit cell.
4. Exact origin centering (`tr.vec = -0.5 * grid_size * grid_spacing`) so (0, 0, 0)
   maps to voxel `(N_x // 2, N_y // 2, N_z // 2)` and all 8 octants are sampled.
5. Periodic + Bravais-centering origin-peak masking (default r < 4.0 A) and
   unit-cell Z-score normalization, preserving `origin_to_sigma` as an explicit
   scalar feature.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import math
import gemmi
import numpy as np

DEFAULT_GRID_SIZE: Tuple[int, int, int] = (128, 128, 128)
DEFAULT_GRID_SPACING: float = 1.0
DEFAULT_D_MIN: Optional[float] = 3.5
DEFAULT_ORIGIN_MASK_RADIUS: float = 4.0

# Calculated / map-coefficient MTZ column prefixes/names to avoid when picking F_obs
_EXCLUDED_F_LABELS = {
    "FWT", "2FOFCWT", "DELFWT", "FOFCWT", "FC", "FCALC", "FC_ALL",
    "FC_ALL_LS", "FMODEL", "FDM", "FEM", "F_CALC", "F-model",
}
_PREFERRED_F_LABELS = (
    "FP", "FOBS", "F-obs", "F-obs-filtered", "F_meas_au", "F_meas", "FNAT", "F",
)
_PREFERRED_I_LABELS = (
    "IMEAN", "IOBS", "I-obs", "I_meas_au", "I_meas", "I",
)


@dataclass
class PattersonResult:
    """Container for a normalized Cartesian Patterson grid and crystal metadata."""

    array: np.ndarray
    origin_to_sigma: float
    non_origin_mean: float
    non_origin_std: float
    cell_volume: float
    asu_volume: float
    spacegroup_number: int
    spacegroup_hm: str
    point_group_order: int
    native_d_min: float = float("nan")
    used_d_min: float = float("nan")
    d_max: float = float("nan")
    low_res_intensity_ratio: float = float("nan")
    acentric_e2_minus_1: float = float("nan")


def get_patterson_spacegroup(sg: Optional[gemmi.SpaceGroup]) -> gemmi.SpaceGroup:
    """Return the centrosymmetric, symmorphic Patterson space group (Laue + centering)."""
    if sg is None:
        return gemmi.find_spacegroup_by_name("P -1")
    ops = sg.operations().derive_symmorphic()
    ops.add_inversion()
    patt_sg = gemmi.find_spacegroup_by_ops(ops)
    return patt_sg if patt_sg is not None else gemmi.find_spacegroup_by_name("P -1")


def _resolve_cif_spacegroup(block: gemmi.cif.Block) -> Optional[gemmi.SpaceGroup]:
    """Resolve space group from mmCIF tags or symmetry operator loops."""
    for tag in (
        "_space_group.name_H-M_alt",
        "_symmetry.space_group_name_H-M",
        "_space_group.name_Hall",
        "_symmetry.space_group_name_Hall",
    ):
        val = block.find_value(tag)
        if val and not gemmi.cif.is_null(val):
            sg = gemmi.find_spacegroup_by_name(gemmi.cif.as_string(val).strip())
            if sg is not None:
                return sg

    symops = (
        list(block.find_values("_space_group_symop.operation_xyz"))
        or list(block.find_values("_symmetry_equiv.pos_as_xyz"))
    )
    if symops:
        try:
            ops = gemmi.GroupOps([gemmi.Op(gemmi.cif.as_string(s)) for s in symops])
            sg = gemmi.find_spacegroup_by_ops(ops)
            if sg is not None:
                return sg
        except RuntimeError:
            pass

    for tag in ("_space_group.IT_number", "_symmetry.Int_Tables_number"):
        val = block.find_value(tag)
        if val and not gemmi.cif.is_null(val):
            try:
                return gemmi.find_spacegroup_by_number(gemmi.cif.as_int(val))
            except ValueError:
                pass

    return None


def read_sf_cif_to_mtz(cif_path: Union[str, Path]) -> gemmi.Mtz:
    """Convert a structure-factor mmCIF file to an in-memory gemmi.Mtz object."""
    doc = gemmi.cif.read(str(cif_path))
    rblocks = gemmi.as_refln_blocks(doc)
    if not rblocks:
        raise ValueError(f"No reflection data blocks found in {cif_path}")

    # Pick the first reflection block with valid unit cell and reflections
    chosen_rb = None
    for rb in rblocks:
        if rb.cell.is_crystal() and len(rb.column_labels()) > 0:
            chosen_rb = rb
            break
    if chosen_rb is None:
        chosen_rb = rblocks[0]

    c2m = gemmi.CifToMtz()
    mtz = c2m.convert_block_to_mtz(chosen_rb)
    if mtz.spacegroup is None:
        mtz.spacegroup = _resolve_cif_spacegroup(chosen_rb.block)
    if mtz.spacegroup is None:
        raise ValueError(f"Could not determine space group from CIF file {cif_path}")
    return mtz


def _select_obs_column(
    mtz: gemmi.Mtz,
) -> Tuple[np.ndarray, Optional[np.ndarray], bool]:
    """
    Select observed amplitude (F) or intensity (J) array and its sigma array.

    Returns:
        (values, sigmas, is_intensity)
    """
    all_cols = list(mtz.columns)

    def _find_sigma_for(col_idx: int, col_label: str, sig_type: str) -> Optional[np.ndarray]:
        for candidate in (f"SIG{col_label}", f"{col_label}_sigma", f"SIG_{col_label}"):
            sc = mtz.column_with_label(candidate)
            if sc is not None and sc.type == sig_type:
                return np.array(sc.array, copy=True)
        if col_idx + 1 < len(all_cols) and all_cols[col_idx + 1].type == sig_type:
            return np.array(all_cols[col_idx + 1].array, copy=True)
        sig_cols = mtz.columns_with_type(sig_type)
        if sig_cols:
            return np.array(sig_cols[0].array, copy=True)
        return None

    # 1. Prefer observed structure factor amplitudes (type 'F')
    f_cols = [
        (idx, c)
        for idx, c in enumerate(all_cols)
        if c.type == "F" and c.label.upper() not in _EXCLUDED_F_LABELS
    ]
    if f_cols:
        chosen_idx, chosen_col = f_cols[0]
        for pref in _PREFERRED_F_LABELS:
            for idx, c in f_cols:
                if c.label.upper() == pref.upper():
                    chosen_idx, chosen_col = idx, c
                    break
            else:
                continue
            break
        sig = _find_sigma_for(chosen_idx, chosen_col.label, "Q")
        return np.array(chosen_col.array, copy=True), sig, False

    # 2. Fallback to observed intensities (type 'J')
    j_cols = [(idx, c) for idx, c in enumerate(all_cols) if c.type == "J"]
    if j_cols:
        chosen_idx, chosen_col = j_cols[0]
        for pref in _PREFERRED_I_LABELS:
            for idx, c in j_cols:
                if c.label.upper() == pref.upper():
                    chosen_idx, chosen_col = idx, c
                    break
            else:
                continue
            break
        sig = _find_sigma_for(chosen_idx, chosen_col.label, "M")
        return np.array(chosen_col.array, copy=True), sig, True

    # 3. Fallback to Friedel pairs F(+)/F(-) (type 'G') or I(+)/I(-) (type 'K')
    g_cols = [c for c in all_cols if c.type == "G"]
    if len(g_cols) >= 2:
        fp = np.array(g_cols[0].array, copy=True)
        fm = np.array(g_cols[1].array, copy=True)
        f_mean = np.nanmean(np.stack([fp, fm], axis=0), axis=0)
        return f_mean, None, False

    k_cols = [c for c in all_cols if c.type == "K"]
    if len(k_cols) >= 2:
        ip = np.array(k_cols[0].array, copy=True)
        im = np.array(k_cols[1].array, copy=True)
        i_mean = np.nanmean(np.stack([ip, im], axis=0), axis=0)
        return i_mean, None, True

    raise ValueError(
        f"No suitable observed amplitude (F/G) or intensity (J/K) column in MTZ "
        f"(columns: {[(c.label, c.type) for c in all_cols]})"
    )


def _extract_crystal_metadata(
    cell: gemmi.UnitCell,
    sg: Optional[gemmi.SpaceGroup],
) -> Dict[str, Union[float, int, str]]:
    """Extract unit cell, ASU volume, and symmetry metadata."""
    if sg is None:
        sg = gemmi.find_spacegroup_by_name("P 1")
    ops = sg.operations()
    n_sym = len(ops)
    n_cen = len(ops.cen_ops)
    point_group_order = max(1, len(ops.sym_ops))
    cell_vol = float(cell.volume)
    asu_vol = cell_vol / float(max(1, n_sym))
    return {
        "cell_volume": cell_vol,
        "asu_volume": asu_vol,
        "spacegroup_number": int(sg.number),
        "spacegroup_hm": str(sg.hm),
        "point_group_order": int(point_group_order),
        "n_centering": int(n_cen),
    }


def compute_normalized_patterson_grid_from_mtz(
    mtz: gemmi.Mtz,
    d_min: Optional[float] = DEFAULT_D_MIN,
    grid_spacing: float = DEFAULT_GRID_SPACING,
    max_e: float = 6.0,
) -> Tuple[gemmi.FloatGrid, Dict[str, Union[float, int, str]]]:
    """
    Compute a normalized E^2 Patterson unit-cell FloatGrid directly from an MTZ.

    Applies resolution shell normalization (<|F|^2 / eps> = 1) with smooth radial
    interpolation, optional uniform high-resolution cutoff (`d_min`), and exact
    Patterson symmetry (`derive_symmorphic() + add_inversion()`).
    """
    if mtz.spacegroup is None:
        raise ValueError("MTZ is missing space group symmetry.")

    sg = mtz.spacegroup
    cell = mtz.cell
    ops = sg.operations()

    obs, sig, is_intensity = _select_obs_column(mtz)
    hkl = mtz.make_miller_array()
    d_arr = mtz.make_d_array()
    inv_d2 = mtz.make_1_d2_array()

    valid = np.isfinite(obs) & np.isfinite(inv_d2) & (inv_d2 > 0) & (~ops.systematic_absences(hkl))
    if sig is not None:
        valid &= np.isfinite(sig) & (sig > 0)
        if is_intensity:
            # Reject severely negative outlier intensities while clamping weak negative I to 0
            valid &= obs > -3.0 * sig
        else:
            valid &= obs >= 0.0
    else:
        valid &= obs >= (0.0 if not is_intensity else -1e9)

    if not np.any(valid):
        raise ValueError("No valid reflections remaining in MTZ after filtering.")

    native_d_min = float(np.min(d_arr[valid]))
    d_max = float(np.max(d_arr[valid]))

    # Compute unnormalized |F|^2 / epsilon BEFORE resolution truncation to capture
    # low-resolution Babinet bulk-solvent intensity drop if available.
    eps = ops.epsilon_factor_without_centering_array(hkl).astype(np.float64)
    i_over_eps = np.zeros(len(obs), dtype=np.float64)
    if is_intensity:
        i_over_eps[valid] = np.maximum(obs[valid].astype(np.float64), 0.0) / eps[valid]
    else:
        i_over_eps[valid] = (obs[valid].astype(np.float64) ** 2) / eps[valid]

    low_res_mask = valid & (d_arr > 10.0)
    ref_res_mask = valid & (d_arr >= 4.0) & (d_arr <= 6.0)
    if np.sum(low_res_mask) >= 5 and np.sum(ref_res_mask) >= 10:
        ref_mean = float(np.mean(i_over_eps[ref_res_mask]))
        low_res_ratio = float(np.mean(i_over_eps[low_res_mask]) / ref_mean) if ref_mean > 0 else float("nan")
    else:
        low_res_ratio = float("nan")

    # Apply uniform high-resolution cutoff to prevent d_min peak-sharpness confounding
    if d_min is not None and d_min > 0:
        trunc_valid = valid & (d_arr >= d_min)
        if np.sum(trunc_valid) >= 20:
            valid = trunc_valid
    used_d_min = float(np.min(d_arr[valid]))

    # Shell normalization (<E^2> = 1) with piecewise-linear interpolation across 1/d^2 centers
    n_valid = int(np.sum(valid))
    nbins = max(1, min(60, n_valid // 100))
    e2 = np.zeros(len(obs), dtype=np.float32)

    if nbins == 1:
        mean_val = float(np.mean(i_over_eps[valid]))
        if mean_val > 0:
            e2[valid] = (i_over_eps[valid] / mean_val).astype(np.float32)
    else:
        binner = gemmi.Binner()
        binner.setup_from_1_d2(nbins, gemmi.Binner.Method.EqualCount, inv_d2[valid], cell)
        bins = binner.get_bins_from_1_d2(inv_d2[valid])
        shell_cnt = np.bincount(bins, minlength=nbins)
        shell_sum_i = np.bincount(bins, weights=i_over_eps[valid], minlength=nbins)
        shell_sum_x = np.bincount(bins, weights=inv_d2[valid], minlength=nbins)

        nonempty = shell_cnt > 0
        centers_x = shell_sum_x[nonempty] / shell_cnt[nonempty]
        means_i = shell_sum_i[nonempty] / shell_cnt[nonempty]
        order = np.argsort(centers_x)
        centers_x = centers_x[order]
        means_i = np.maximum(means_i[order], 1e-12)

        scale = np.interp(inv_d2[valid], centers_x, means_i)
        e2[valid] = (i_over_eps[valid] / scale).astype(np.float32)

    if max_e is not None and max_e > 0:
        np.clip(e2, 0.0, float(max_e ** 2), out=e2)

    acentric_e2_minus_1 = float(np.mean(np.abs(e2[valid] - 1.0)))

    # Build minimal in-memory Patterson MTZ and run FFT with symmorphic + inversion symmetry
    patt_mtz = gemmi.Mtz(with_base=True)
    patt_mtz.spacegroup = get_patterson_spacegroup(sg)
    patt_mtz.set_cell_for_all(cell)
    patt_mtz.add_dataset("patterson")
    patt_mtz.add_column("E2", "F")
    patt_mtz.add_column("PHI0", "P")

    valid_idx = np.flatnonzero(valid)
    data = np.zeros((len(valid_idx), 5), dtype=np.float32)
    data[:, :3] = hkl[valid_idx]
    data[:, 3] = e2[valid_idx]
    data[:, 4] = 0.0
    patt_mtz.set_data(data)

    # Oversample fractional FFT grid (~0.45 * grid_spacing) so trilinear Cartesian interpolation is exact
    target_fft_spacing = min(0.5, grid_spacing * 0.5)
    sample_rate = max(2.5, used_d_min / max(0.2, target_fft_spacing))
    grid = patt_mtz.transform_f_phi_to_map("E2", "PHI0", sample_rate=sample_rate)

    meta = _extract_crystal_metadata(cell, sg)
    meta.update({
        "native_d_min": native_d_min,
        "used_d_min": used_d_min,
        "d_max": d_max,
        "low_res_intensity_ratio": low_res_ratio,
        "acentric_e2_minus_1": acentric_e2_minus_1,
    })
    return grid, meta


def load_patterson_ccp4_map(
    map_path: Union[str, Path],
) -> Tuple[gemmi.FloatGrid, Dict[str, Union[float, int, str]]]:
    """
    Load a CCP4 .map / .map.gz file with full crystallographic setup (`setup=True`),
    guaranteeing standard X, Y, Z axis order and full unit-cell symmetry expansion.
    """
    ccp4 = gemmi.read_ccp4_map(str(map_path), setup=True)
    grid = ccp4.grid
    sg = grid.spacegroup
    meta = _extract_crystal_metadata(grid.unit_cell, sg)
    # Ensure grid carries Patterson symmetry for origin/centering masking
    grid.spacegroup = get_patterson_spacegroup(sg)
    return grid, meta


def interpolate_and_normalize_patterson(
    grid: gemmi.FloatGrid,
    grid_size: Tuple[int, int, int] = DEFAULT_GRID_SIZE,
    grid_spacing: float = DEFAULT_GRID_SPACING,
    origin_mask_radius: float = DEFAULT_ORIGIN_MASK_RADIUS,
    metadata: Optional[Dict[str, Union[float, int, str]]] = None,
) -> PattersonResult:
    """
    Mask periodic + Bravais-centering origin peaks on a copy of the unit-cell grid,
    Z-score normalize non-origin density (`mean=0, std=1`), and trilinearly
    interpolate onto a centered Cartesian box `[-L/2, +L/2)` around (0, 0, 0).
    """
    work_grid = grid.clone()
    patt_sg = get_patterson_spacegroup(work_grid.spacegroup)
    work_grid.spacegroup = patt_sg

    raw_origin = float(work_grid.get_value(0, 0, 0))
    grid_arr = np.array(work_grid, copy=False)

    # Replace any NaNs (e.g. incomplete ASU in malformed maps) with finite mean before statistics
    finite_mask = np.isfinite(grid_arr)
    if not np.all(finite_mask):
        fill_val = float(np.mean(grid_arr[finite_mask])) if np.any(finite_mask) else 0.0
        grid_arr[~finite_mask] = fill_val

    if origin_mask_radius > 0.0:
        origin_mask_grid = gemmi.Int8Grid(*work_grid.shape)
        origin_mask_grid.set_unit_cell(work_grid.unit_cell)
        origin_mask_grid.spacegroup = patt_sg
        origin_mask_grid.set_points_around(
            gemmi.Position(0.0, 0.0, 0.0),
            radius=float(origin_mask_radius),
            value=1,
        )
        # Because patt_sg is symmorphic, the orbit of (0,0,0) consists exclusively
        # of exact Bravais centering translations (never Harker peaks).
        origin_mask_grid.symmetrize_max()
        is_origin = np.array(origin_mask_grid, copy=False) > 0
    else:
        is_origin = np.zeros(work_grid.shape, dtype=bool)

    non_origin = ~is_origin
    if np.any(non_origin):
        mu = float(np.mean(grid_arr[non_origin], dtype=np.float64))
        sigma = float(np.std(grid_arr[non_origin], dtype=np.float64))
    else:
        mu = float(np.mean(grid_arr, dtype=np.float64))
        sigma = float(np.std(grid_arr, dtype=np.float64))

    if sigma < 1e-12:
        sigma = 1.0

    origin_to_sigma = (raw_origin - mu) / sigma

    # Normalize in-place on unit-cell grid; set masked origin spheres to background mean (0.0)
    grid_arr -= mu
    grid_arr /= sigma
    if origin_mask_radius > 0.0:
        grid_arr[is_origin] = 0.0

    # Interpolate onto centered Cartesian grid: r = diag(spacing) * idx - 0.5 * grid_size * spacing
    arr = np.zeros(grid_size, dtype=np.float32)
    tr = gemmi.Transform()
    tr.mat.fromlist([
        [grid_spacing, 0.0, 0.0],
        [0.0, grid_spacing, 0.0],
        [0.0, 0.0, grid_spacing],
    ])
    tr.vec.fromlist([
        -0.5 * grid_size[0] * grid_spacing,
        -0.5 * grid_size[1] * grid_spacing,
        -0.5 * grid_size[2] * grid_spacing,
    ])
    work_grid.interpolate_values(arr, tr)

    meta = metadata or _extract_crystal_metadata(work_grid.unit_cell, work_grid.spacegroup)
    return PattersonResult(
        array=arr,
        origin_to_sigma=origin_to_sigma,
        non_origin_mean=mu,
        non_origin_std=sigma,
        cell_volume=float(meta.get("cell_volume", float("nan"))),
        asu_volume=float(meta.get("asu_volume", float("nan"))),
        spacegroup_number=int(meta.get("spacegroup_number", 1)),
        spacegroup_hm=str(meta.get("spacegroup_hm", "P 1")),
        point_group_order=int(meta.get("point_group_order", 1)),
        native_d_min=float(meta.get("native_d_min", float("nan"))),
        used_d_min=float(meta.get("used_d_min", float("nan"))),
        d_max=float(meta.get("d_max", float("nan"))),
        low_res_intensity_ratio=float(meta.get("low_res_intensity_ratio", float("nan"))),
        acentric_e2_minus_1=float(meta.get("acentric_e2_minus_1", float("nan"))),
    )


def compute_patterson_from_file(
    file_path: Union[str, Path],
    grid_size: Tuple[int, int, int] = DEFAULT_GRID_SIZE,
    grid_spacing: float = DEFAULT_GRID_SPACING,
    d_min: Optional[float] = DEFAULT_D_MIN,
    origin_mask_radius: float = DEFAULT_ORIGIN_MASK_RADIUS,
) -> PattersonResult:
    """
    Unified entry point supporting .mtz, .cif/.ent structure-factor files,
    and .map/.map.gz/.ccp4 Patterson map files.
    """
    path = Path(file_path)
    name_lower = path.name.lower()

    if name_lower.endswith((".map", ".map.gz", ".ccp4", ".ccp4.gz", ".mrc", ".mrc.gz")):
        grid, meta = load_patterson_ccp4_map(path)
    elif name_lower.endswith(".mtz"):
        mtz = gemmi.read_mtz_file(str(path))
        grid, meta = compute_normalized_patterson_grid_from_mtz(
            mtz, d_min=d_min, grid_spacing=grid_spacing
        )
    elif name_lower.endswith((".cif", ".cif.gz", ".ent", ".ent.gz", ".sf")):
        mtz = read_sf_cif_to_mtz(path)
        grid, meta = compute_normalized_patterson_grid_from_mtz(
            mtz, d_min=d_min, grid_spacing=grid_spacing
        )
    else:
        raise ValueError(f"Unsupported crystallographic file extension: {file_path}")

    return interpolate_and_normalize_patterson(
        grid=grid,
        grid_size=grid_size,
        grid_spacing=grid_spacing,
        origin_mask_radius=origin_mask_radius,
        metadata=meta,
    )
