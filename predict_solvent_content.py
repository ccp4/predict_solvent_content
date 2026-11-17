"""
Author: David McDonagh (david.mcdonagh@stfc.ac.uk)

Unified inference script for calculating protein crystal solvent content (%)
without requiring sequence or crystal metadata, supporting:
  - 3D ResidualUNetSE3D (`unet_model.onnx`)
  - 3D ViT-V-Net (`vitvnet_model.onnx`)
  - Patterson descriptor ensembles (`descriptor_model.joblib`)

Usage:
    python predict_solvent_content.py example.map
    python predict_solvent_content.py example.mtz descriptor_model.joblib
    python predict_solvent_content.py example.cif vitvnet_model.onnx
    python predict_solvent_content.py example.txt [model.onnx | model.joblib]
"""

import argparse
import os
from pathlib import Path
from typing import Generator, List, Optional, Tuple
import numpy as np

from patterson import (
    DEFAULT_D_MIN,
    DEFAULT_GRID_SIZE,
    DEFAULT_GRID_SPACING,
    DEFAULT_ORIGIN_MASK_RADIUS,
    compute_patterson_from_file,
)


def _expand_input_list(file_path: str) -> List[str]:
    p = Path(file_path)
    if p.suffix.lower() == ".txt":
        with open(p, "r", encoding="utf-8") as f:
            return [
                line.strip() for line in f if line.strip() and not line.startswith("#")
            ]
    return [file_path]


def parse_input(
    file_path: str,
    grid_size: Tuple[int, int, int] = DEFAULT_GRID_SIZE,
    grid_spacing: float = DEFAULT_GRID_SPACING,
    d_min: Optional[float] = DEFAULT_D_MIN,
    origin_mask_radius: float = DEFAULT_ORIGIN_MASK_RADIUS,
) -> Generator[Tuple[str, np.ndarray], None, None]:
    """
    Yield `(name, patterson_tensor)` pairs where `patterson_tensor` has shape
    `(1, 1, N_x, N_y, N_z)` in float32, computed purely in Python via Gemmi.
    """
    for filename in _expand_input_list(file_path):
        item_path = Path(filename)
        res = compute_patterson_from_file(
            file_path=item_path,
            grid_size=grid_size,
            grid_spacing=grid_spacing,
            d_min=d_min,
            origin_mask_radius=origin_mask_radius,
        )
        tensor = res.array.astype(np.float32, copy=False).reshape(
            1, 1, grid_size[0], grid_size[1], grid_size[2]
        )
        yield item_path.name, tensor


def _run_joblib_descriptor_inference(
    input_path: str,
    model_path: str,
    grid_size_override: Optional[int],
    grid_spacing_override: Optional[float],
    d_min_override: Optional[float],
    origin_mask_override: Optional[float],
    d_min_hint: Optional[float] = None,
) -> None:
    import joblib
    import math
    from generate_descriptors import extract_descriptors_for_file

    bundle = joblib.load(model_path)
    model = bundle["model"]
    feat_cols: List[str] = bundle["feature_names"]

    grid_size = (
        (grid_size_override, grid_size_override, grid_size_override)
        if grid_size_override is not None
        else tuple(bundle.get("grid_size", DEFAULT_GRID_SIZE))
    )
    grid_spacing = (
        grid_spacing_override
        if grid_spacing_override is not None
        else float(bundle.get("grid_spacing", DEFAULT_GRID_SPACING))
    )
    if d_min_override is not None:
        d_min = d_min_override if d_min_override > 0 else None
    else:
        d_min = bundle.get("d_min", DEFAULT_D_MIN)
    origin_mask_radius = (
        origin_mask_override
        if origin_mask_override is not None
        else float(bundle.get("origin_mask_radius", DEFAULT_ORIGIN_MASK_RADIUS))
    )

    for fp in _expand_input_list(input_path):
        desc = extract_descriptors_for_file(
            file_path=fp,
            grid_size=grid_size,  # type: ignore[arg-type]
            grid_spacing=grid_spacing,
            d_min=d_min,
            origin_mask_radius=origin_mask_radius,
        )
        if d_min_hint is not None and not math.isfinite(desc.get("native_d_min", float("nan"))):
            desc["native_d_min"] = d_min_hint
        vec = np.array(
            [[desc.get(col, float("nan")) for col in feat_cols]], dtype=np.float64
        )
        vec = np.where(np.isfinite(vec), vec, np.nan)
        pred_pct = float(np.clip(model.predict(vec)[0], 0.0, 100.0))
        print(f"{Path(fp).name}: {round(pred_pct, 3)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict protein crystal solvent content (%) from MTZ, CIF, CCP4 map, or NPY files."
    )
    parser.add_argument(
        "input_path",
        help="Path to .map/.map.gz, .mtz, .cif, .npy, or .txt list of files.",
    )
    parser.add_argument(
        "model_path",
        nargs="?",
        default="unet_model.onnx",
        help="Path to ONNX (.onnx) or tree-descriptor (.joblib) model file (default: unet_model.onnx).",
    )
    parser.add_argument("--grid-size", type=int, default=None)
    parser.add_argument("--grid-spacing", type=float, default=None)
    parser.add_argument("--d-min", type=float, default=None)
    parser.add_argument("--d-min-hint", type=float, default=None, help="Optional native d_min (A) hint for .map inputs.")
    parser.add_argument("--origin-mask-radius", type=float, default=None)
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Enable CUDAExecutionProvider for ONNX models if available.",
    )
    args = parser.parse_args()

    resolved_model = (
        args.model_path
        if os.path.isabs(args.model_path)
        else os.path.join(os.getcwd(), args.model_path)
    )

    if resolved_model.lower().endswith((".joblib", ".pkl")):
        _run_joblib_descriptor_inference(
            input_path=args.input_path,
            model_path=resolved_model,
            grid_size_override=args.grid_size,
            grid_spacing_override=args.grid_spacing,
            d_min_override=args.d_min,
            origin_mask_override=args.origin_mask_radius,
            d_min_hint=args.d_min_hint,
        )
        return

    import onnxruntime as ort

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if args.gpu
        else ["CPUExecutionProvider"]
    )
    onnx_model = ort.InferenceSession(resolved_model, providers=providers)

    if args.grid_size is not None:
        grid_size = (args.grid_size, args.grid_size, args.grid_size)
    else:
        in_shape = onnx_model.get_inputs()[0].shape
        if len(in_shape) == 5 and all(
            isinstance(d, int) and d > 0 for d in in_shape[2:]
        ):
            grid_size = (int(in_shape[2]), int(in_shape[3]), int(in_shape[4]))
        else:
            grid_size = DEFAULT_GRID_SIZE

    grid_spacing = (
        args.grid_spacing if args.grid_spacing is not None else DEFAULT_GRID_SPACING
    )
    if args.d_min is not None:
        d_min = args.d_min if args.d_min > 0 else None
    else:
        d_min = DEFAULT_D_MIN
    origin_mask_radius = (
        args.origin_mask_radius
        if args.origin_mask_radius is not None
        else DEFAULT_ORIGIN_MASK_RADIUS
    )

    input_data = parse_input(
        args.input_path,
        grid_size=grid_size,
        grid_spacing=grid_spacing,
        d_min=d_min,
        origin_mask_radius=origin_mask_radius,
    )

    input_name = onnx_model.get_inputs()[0].name
    for name, pmap in input_data:
        result = onnx_model.run(None, {input_name: pmap})
        pred_pct = float(np.asarray(result[0]).ravel()[0]) * 100.0
        print(f"{name}: {round(pred_pct, 3)}")


if __name__ == "__main__":
    main()
