"""
Script for converting trained multi-modal PyTorch solvent-content models
(`unet` or `vitvnet`, accepting `(input, metadata)`) to ONNX.

Usage:
    python export_to_onnx.py models/best_unet.pt unet
    python export_to_onnx.py models/best_vitvnet.pt vitvnet --output vitvnet_model.onnx
"""

import argparse
import os
import torch

from dl_common import NUM_DL_METADATA
from patterson import DEFAULT_GRID_SIZE
from usolv import ResidualUNetSE3D
from vitsolv import ViTVNet, get_3DReg_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Export PyTorch 3D Patterson + metadata model to ONNX.")
    parser.add_argument("model_path", help="Path to trained PyTorch checkpoint (.pt or state_dict).")
    parser.add_argument("model_type", nargs="?", choices=("unet", "vitvnet"), default=None)
    parser.add_argument("--grid-size", type=int, default=None)
    parser.add_argument("--output", default=None, help="Output ONNX file path (default: <model_type>_model.onnx).")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = args.model_path if os.path.isabs(args.model_path) else os.path.join(os.getcwd(), args.model_path)
    ckpt_obj = torch.load(ckpt_path, map_location=device, weights_only=True)

    if isinstance(ckpt_obj, dict) and "state_dict" in ckpt_obj:
        state_dict = ckpt_obj["state_dict"]
        model_type = args.model_type or ckpt_obj.get("model_type", "unet")
        grid_n = args.grid_size or int(ckpt_obj.get("grid_size", DEFAULT_GRID_SIZE[0]))
    else:
        state_dict = ckpt_obj
        model_type = args.model_type or "unet"
        grid_n = args.grid_size or DEFAULT_GRID_SIZE[0]

    img_size = (grid_n, grid_n, grid_n)
    if model_type == "vitvnet":
        config = get_3DReg_config(grid_size=grid_n)
        model = ViTVNet(config=config, img_size=img_size)
    else:
        model = ResidualUNetSE3D(
            in_channels=1,
            f_maps=[16, 32, 64, 128, 256],
        )

    clean_state = {k.removeprefix("module."): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state)
    model = model.to(device)
    model.eval()

    dummy_vol = torch.randn(1, 1, *img_size, device=device)
    dummy_meta = torch.zeros(1, NUM_DL_METADATA, device=device)
    out_name = args.output or f"{model_type}_model.onnx"
    onnx_model_path = out_name if os.path.isabs(out_name) else os.path.join(os.getcwd(), out_name)

    torch.onnx.export(
        model,
        (dummy_vol, dummy_meta),
        onnx_model_path,
        export_params=True,
        opset_version=args.opset,
        input_names=["input", "metadata"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "metadata": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    print(f"Exported {model_type} ONNX model (grid={img_size}) to {onnx_model_path}")


if __name__ == "__main__":
    main()
