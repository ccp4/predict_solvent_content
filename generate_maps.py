"""
Script to convert diffraction files (.mtz, .cif/.ent) or Patterson maps (.map, .map.gz)
into origin-masked, Z-score normalized 3D numpy arrays on a centered 128 A^3 Cartesian grid
(default: 128^3 voxels at 1.0 A spacing in float16 = 4 MB/map, truncated at d_min = 3.5 A).
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple
import gemmi
import numpy as np

from patterson import (
    DEFAULT_D_MIN,
    DEFAULT_GRID_SIZE,
    DEFAULT_GRID_SPACING,
    DEFAULT_ORIGIN_MASK_RADIUS,
    PattersonResult,
    compute_patterson_from_file,
    load_patterson_ccp4_map,
)


def get_p_map_grid(file_path: str) -> gemmi.FloatGrid:
    """Load a CCP4 Patterson map with full crystallographic symmetry and XYZ setup."""
    grid, _ = load_patterson_ccp4_map(file_path)
    return grid


def get_p_map_corrected(
    file_path: str,
    grid_size: Tuple[int, int, int] = DEFAULT_GRID_SIZE,
    grid_spacing: float = DEFAULT_GRID_SPACING,
    d_min: Optional[float] = DEFAULT_D_MIN,
    origin_mask_radius: float = DEFAULT_ORIGIN_MASK_RADIUS,
) -> np.ndarray:
    """
    Compute or load a Patterson map, mask periodic/centering origin peaks,
    Z-score normalize non-origin density, and interpolate onto a centered Cartesian grid.
    """
    result = compute_patterson_from_file(
        file_path=file_path,
        grid_size=grid_size,
        grid_spacing=grid_spacing,
        d_min=d_min,
        origin_mask_radius=origin_mask_radius,
    )
    return result.array


def _derive_output_npy_path(input_path: Path, output_dir: Optional[Path]) -> Path:
    name = input_path.name
    for suffix in (".map.gz", ".ccp4.gz", ".mrc.gz", ".cif.gz", ".ent.gz", ".map", ".ccp4", ".mrc", ".mtz", ".cif", ".ent", ".sf"):
        if name.lower().endswith(suffix):
            stem = name[: -len(suffix)]
            break
    else:
        stem = input_path.stem

    target_dir = output_dir if output_dir is not None else input_path.parent
    return target_dir / f"{stem}.npy"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate centered, origin-masked, normalized 3D Patterson .npy grids."
    )
    parser.add_argument(
        "input_file",
        help="Text file listing paths (last whitespace-delimited token per line) or a single crystallographic file.",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=DEFAULT_GRID_SIZE[0],
        help=f"Cubic grid dimension N for an (N, N, N) array (default: {DEFAULT_GRID_SIZE[0]}).",
    )
    parser.add_argument(
        "--grid-spacing",
        type=float,
        default=DEFAULT_GRID_SPACING,
        help=f"Cartesian grid spacing in Angstroms (default: {DEFAULT_GRID_SPACING}).",
    )
    parser.add_argument(
        "--d-min",
        type=float,
        default=DEFAULT_D_MIN,
        help=f"Uniform high-resolution cutoff in Angstroms for MTZ/CIF inputs; <=0 disables (default: {DEFAULT_D_MIN}).",
    )
    parser.add_argument(
        "--origin-mask-radius",
        type=float,
        default=DEFAULT_ORIGIN_MASK_RADIUS,
        help=f"Radius (A) masked around origin and Bravais centering peaks (default: {DEFAULT_ORIGIN_MASK_RADIUS}).",
    )
    parser.add_argument(
        "--dtype",
        choices=("float16", "float32"),
        default="float16",
        help="Storage dtype for output .npy arrays (default: float16, 4 MB per 128^3 map).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for generated .npy files (defaults to alongside each input file).",
    )
    parser.add_argument(
        "--metadata-out",
        type=Path,
        default=Path("map_metadata.tsv"),
        help="TSV file recording scalar Patterson & crystal metadata (default: map_metadata.tsv).",
    )
    args = parser.parse_args()

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    grid_size = (args.grid_size, args.grid_size, args.grid_size)
    d_min = args.d_min if (args.d_min is not None and args.d_min > 0) else None
    out_dtype = np.float16 if args.dtype == "float16" else np.float32

    in_path = Path(args.input_file)
    if in_path.suffix.lower() == ".txt":
        with open(in_path, "r", encoding="utf-8") as f:
            raw_paths = [line.split()[-1] for line in f if line.strip()]
    else:
        raw_paths = [str(in_path)]

    write_header = not args.metadata_out.exists()
    args.metadata_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.metadata_out, "a", encoding="utf-8") as meta_f:
        if write_header:
            meta_f.write(
                "name\tnpy_path\torigin_to_sigma\tasu_volume\tcell_volume\t"
                "spacegroup_number\tpoint_group_order\tnative_d_min\tused_d_min\t"
                "low_res_intensity_ratio\tacentric_e2_minus_1\n"
            )

        for file_str in raw_paths:
            src_path = Path(file_str)
            out_path = _derive_output_npy_path(src_path, args.output_dir)
            name = out_path.stem
            try:
                res: PattersonResult = compute_patterson_from_file(
                    file_path=src_path,
                    grid_size=grid_size,
                    grid_spacing=args.grid_spacing,
                    d_min=d_min,
                    origin_mask_radius=args.origin_mask_radius,
                )
                np.save(out_path, res.array.astype(out_dtype))
                meta_f.write(
                    f"{name}\t{out_path}\t{res.origin_to_sigma:.6f}\t"
                    f"{res.asu_volume:.3f}\t{res.cell_volume:.3f}\t"
                    f"{res.spacegroup_number}\t{res.point_group_order}\t"
                    f"{res.native_d_min:.3f}\t{res.used_d_min:.3f}\t"
                    f"{res.low_res_intensity_ratio:.6f}\t{res.acentric_e2_minus_1:.6f}\n"
                )
                meta_f.flush()
            except Exception as exc:
                print(f"Failed on {src_path}: {exc}")


if __name__ == "__main__":
    main()
