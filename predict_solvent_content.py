"""
Author: David McDonagh (david.mcdonagh@stfc.ac.uk)

Script for calculating protein crystal solvent content using a modified
ViT-V-Net transformer model or ResidualUNetSE3D encoder model.

Requirements:
    numpy
    gemmi
    onnxruntime (or onnxruntime-gpu)

Usage:
    python predict_solvent_content.py example.map
    python predict_solvent_content.py example.mtz
    python predict_solvent_content.py example.cif
    python predict_solvent_content.py example.txt [model.onnx]
"""

import argparse
import os
from pathlib import Path
from typing import Generator, Optional, Tuple
import numpy as np
import onnxruntime as ort

from patterson import (
    DEFAULT_D_MIN,
    DEFAULT_GRID_SIZE,
    DEFAULT_GRID_SPACING,
    DEFAULT_ORIGIN_MASK_RADIUS,
    compute_patterson_from_file,
)


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
    p = Path(file_path)
    if p.suffix.lower() == ".txt":
        with open(p, "r", encoding="utf-8") as f:
            filenames = [line.strip() for line in f if line.strip()]
    else:
        filenames = [file_path]

    for filename in filenames:
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict protein crystal solvent content (%) from MTZ, CIF, or CCP4 map files."
    )
    parser.add_argument(
        "input_path",
        help="Path to .map/.map.gz, .mtz, .cif, or .txt list of files.",
    )
    parser.add_argument(
        "model_path",
        nargs="?",
        default="unet_model.onnx",
        help="Path to ONNX model file (default: unet_model.onnx).",
    )
    parser.add_argument("--grid-size", type=int, default=None)
    parser.add_argument("--grid-spacing", type=float, default=DEFAULT_GRID_SPACING)
    parser.add_argument("--d-min", type=float, default=DEFAULT_D_MIN)
    parser.add_argument("--origin-mask-radius", type=float, default=DEFAULT_ORIGIN_MASK_RADIUS)
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Enable CUDAExecutionProvider if available.",
    )
    args = parser.parse_args()

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if args.gpu
        else ["CPUExecutionProvider"]
    )
    onnx_model_path = (
        args.model_path
        if os.path.isabs(args.model_path)
        else os.path.join(os.getcwd(), args.model_path)
    )
    onnx_model = ort.InferenceSession(onnx_model_path, providers=providers)

    if args.grid_size is not None:
        grid_size = (args.grid_size, args.grid_size, args.grid_size)
    else:
        in_shape = onnx_model.get_inputs()[0].shape
        if len(in_shape) == 5 and all(isinstance(d, int) and d > 0 for d in in_shape[2:]):
            grid_size = (int(in_shape[2]), int(in_shape[3]), int(in_shape[4]))
        else:
            grid_size = DEFAULT_GRID_SIZE

    d_min = args.d_min if (args.d_min is not None and args.d_min > 0) else None
    input_data = parse_input(
        args.input_path,
        grid_size=grid_size,
        grid_spacing=args.grid_spacing,
        d_min=d_min,
        origin_mask_radius=args.origin_mask_radius,
    )

    input_name = onnx_model.get_inputs()[0].name
    for name, pmap in input_data:
        result = onnx_model.run(None, {input_name: pmap})
        pred_pct = float(np.asarray(result[0]).ravel()[0]) * 100.0
        print(f"{name}: {round(pred_pct, 3)}")


if __name__ == "__main__":
    main()
