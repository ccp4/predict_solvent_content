"""
Author: David McDonagh (david.mcdonagh@stfc.ac.uk)

Unified inference script for calculating protein crystal solvent content (%),
automatically fusing 3D Patterson maps with pre-phasing diffraction/lattice
metadata (d_min, unit cell, V_asu, space group, Wilson B) across all model formats:
  - PyTorch checkpoints (`models/best_unet.pt`, `models/best_vitvnet.pt`)
  - ONNX Runtime models (`unet_model.onnx`, `vitvnet_model.onnx`)
  - Tree-descriptor ensembles (`descriptor_model.joblib`)

Usage:
    python predict_solvent_content.py example.mtz models/best_unet.pt
    python predict_solvent_content.py example.mtz unet_model.onnx
    python predict_solvent_content.py example.mtz descriptor_model.joblib
    python predict_solvent_content.py example.txt models/best_unet.pt --gpu
"""

import argparse
import math
import os
from pathlib import Path
from typing import Generator, List, Optional, Tuple
import numpy as np

from dl_common import patterson_result_to_metadata_vec
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
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return [file_path]


def parse_input_with_metadata(
    file_path: str,
    grid_size: Tuple[int, int, int] = DEFAULT_GRID_SIZE,
    grid_spacing: float = DEFAULT_GRID_SPACING,
    d_min: Optional[float] = DEFAULT_D_MIN,
    origin_mask_radius: float = DEFAULT_ORIGIN_MASK_RADIUS,
    d_min_hint: Optional[float] = None,
) -> Generator[Tuple[str, np.ndarray, np.ndarray], None, None]:
    """
    Yield `(name, patterson_tensor, metadata_tensor)` tuples where:
      - `patterson_tensor` has shape `(1, 1, N_x, N_y, N_z)` in float32
      - `metadata_tensor` has shape `(1, 17)` in float32 (`NaN` where unknown)
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
        vol_tensor = res.array.astype(np.float32, copy=False).reshape(
            1, 1, grid_size[0], grid_size[1], grid_size[2]
        )
        meta_vec = patterson_result_to_metadata_vec(res, d_min_hint=d_min_hint).reshape(1, -1)
        yield item_path.name, vol_tensor, meta_vec


def parse_input(
    file_path: str,
    grid_size: Tuple[int, int, int] = DEFAULT_GRID_SIZE,
    grid_spacing: float = DEFAULT_GRID_SPACING,
    d_min: Optional[float] = DEFAULT_D_MIN,
    origin_mask_radius: float = DEFAULT_ORIGIN_MASK_RADIUS,
) -> Generator[Tuple[str, np.ndarray], None, None]:
    """Backward-compatible generator yielding `(name, patterson_tensor)`."""
    for name, vol_tensor, _ in parse_input_with_metadata(
        file_path=file_path,
        grid_size=grid_size,
        grid_spacing=grid_spacing,
        d_min=d_min,
        origin_mask_radius=origin_mask_radius,
    ):
        yield name, vol_tensor


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


def _run_pytorch_inference(
    input_path: str,
    model_path: str,
    arch_override: Optional[str],
    grid_size_override: Optional[int],
    grid_spacing: float,
    d_min: Optional[float],
    origin_mask_radius: float,
    d_min_hint: Optional[float],
    use_gpu: bool,
) -> None:
    import torch
    from usolv import ResidualUNetSE3D
    from vitsolv import ViTVNet, get_3DReg_config

    device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
    ckpt_obj = torch.load(model_path, map_location=device, weights_only=True)

    if isinstance(ckpt_obj, dict) and "state_dict" in ckpt_obj:
        state_dict = ckpt_obj["state_dict"]
        arch = arch_override or ckpt_obj.get("model_type", "unet")
        grid_n = grid_size_override or int(ckpt_obj.get("grid_size", DEFAULT_GRID_SIZE[0]))
    else:
        state_dict = ckpt_obj
        arch = arch_override or ("vitvnet" if "vit" in os.path.basename(model_path).lower() else "unet")
        grid_n = grid_size_override or DEFAULT_GRID_SIZE[0]

    grid_size = (grid_n, grid_n, grid_n)
    if arch == "vitvnet":
        model = ViTVNet(config=get_3DReg_config(grid_size=grid_n), img_size=grid_size)
    else:
        model = ResidualUNetSE3D(in_channels=1, f_maps=[16, 32, 64, 128, 256])

    clean_state = {k.removeprefix("module."): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state)
    model = model.to(device)
    model.eval()

    stream = parse_input_with_metadata(
        input_path,
        grid_size=grid_size,
        grid_spacing=grid_spacing,
        d_min=d_min,
        origin_mask_radius=origin_mask_radius,
        d_min_hint=d_min_hint,
    )
    with torch.no_grad():
        for name, vol_np, meta_np in stream:
            vol_t = torch.from_numpy(vol_np).to(device)
            meta_t = torch.from_numpy(meta_np).to(device)
            out = model(vol_t, meta_t)
            pred_pct = float(out.detach().cpu().numpy().ravel()[0]) * 100.0
            print(f"{name}: {round(pred_pct, 3)}")


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
        help="Path to PyTorch (.pt), ONNX (.onnx), or tree-descriptor (.joblib) model file.",
    )
    parser.add_argument("--arch", choices=("unet", "vitvnet"), default=None, help="Architecture hint for raw PyTorch state_dicts.")
    parser.add_argument("--grid-size", type=int, default=None)
    parser.add_argument("--grid-spacing", type=float, default=None)
    parser.add_argument("--d-min", type=float, default=None)
    parser.add_argument("--d-min-hint", type=float, default=None, help="Optional native d_min (A) hint for .map/.npy inputs.")
    parser.add_argument("--origin-mask-radius", type=float, default=None)
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Enable GPU execution if CUDA is available.",
    )
    args = parser.parse_args()

    resolved_model = (
        args.model_path
        if os.path.isabs(args.model_path)
        else os.path.join(os.getcwd(), args.model_path)
    )
    lower_model = resolved_model.lower()

    if lower_model.endswith((".joblib", ".pkl")):
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

    grid_spacing = args.grid_spacing if args.grid_spacing is not None else DEFAULT_GRID_SPACING
    d_min = (args.d_min if args.d_min > 0 else None) if args.d_min is not None else DEFAULT_D_MIN
    origin_mask_radius = (
        args.origin_mask_radius if args.origin_mask_radius is not None else DEFAULT_ORIGIN_MASK_RADIUS
    )

    if not lower_model.endswith(".onnx"):
        _run_pytorch_inference(
            input_path=args.input_path,
            model_path=resolved_model,
            arch_override=args.arch,
            grid_size_override=args.grid_size,
            grid_spacing=grid_spacing,
            d_min=d_min,
            origin_mask_radius=origin_mask_radius,
            d_min_hint=args.d_min_hint,
            use_gpu=args.gpu,
        )
        return

    import onnxruntime as ort

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if args.gpu
        else ["CPUExecutionProvider"]
    )
    onnx_model = ort.InferenceSession(resolved_model, providers=providers)

    inputs_meta = onnx_model.get_inputs()
    if args.grid_size is not None:
        grid_size = (args.grid_size, args.grid_size, args.grid_size)
    else:
        in_shape = inputs_meta[0].shape
        if len(in_shape) == 5 and all(isinstance(d, int) and d > 0 for d in in_shape[2:]):
            grid_size = (int(in_shape[2]), int(in_shape[3]), int(in_shape[4]))
        else:
            grid_size = DEFAULT_GRID_SIZE

    stream = parse_input_with_metadata(
        args.input_path,
        grid_size=grid_size,
        grid_spacing=grid_spacing,
        d_min=d_min,
        origin_mask_radius=origin_mask_radius,
        d_min_hint=args.d_min_hint,
    )

    vol_name = inputs_meta[0].name
    meta_name = inputs_meta[1].name if len(inputs_meta) >= 2 else None

    for name, pmap, meta_vec in stream:
        feed = {vol_name: pmap}
        if meta_name is not None:
            feed[meta_name] = meta_vec
        result = onnx_model.run(None, feed)
        pred_pct = float(np.asarray(result[0]).ravel()[0]) * 100.0
        print(f"{name}: {round(pred_pct, 3)}")


if __name__ == "__main__":
    main()
