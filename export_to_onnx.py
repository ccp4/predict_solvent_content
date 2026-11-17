"""
Script for converting trained PyTorch solvent-content models to ONNX.

Usage:
    python export_to_onnx.py path/to/pytorch/model unet
    python export_to_onnx.py path/to/pytorch/model vitvnet --output vitvnet_model.onnx
"""

import argparse
import os
import torch

from patterson import DEFAULT_GRID_SIZE
from usolv import ResidualUNetSE3D
from vitsolv import ViTVNet, get_3DReg_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Export PyTorch 3D Patterson model to ONNX.")
    parser.add_argument("model_path", help="Path to trained PyTorch state_dict checkpoint.")
    parser.add_argument("model_type", choices=("unet", "vitvnet"), help="Model architecture.")
    parser.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE[0])
    parser.add_argument("--output", default=None, help="Output ONNX file path (default: <model_type>_model.onnx).")
    parser.add_argument("--opset", type=int, default=14)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = (args.grid_size, args.grid_size, args.grid_size)

    if args.model_type == "vitvnet":
        config = get_3DReg_config(grid_size=args.grid_size)
        model = ViTVNet(config=config, img_size=img_size)
    else:
        model = ResidualUNetSE3D(
            in_channels=1,
            f_maps=[16, 32, 64, 128, 256],
        )

    ckpt_path = args.model_path if os.path.isabs(args.model_path) else os.path.join(os.getcwd(), args.model_path)
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    clean_state = {k.removeprefix("module."): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state)
    model = model.to(device)
    model.eval()

    dummy_input = torch.randn(1, 1, *img_size, device=device)
    out_name = args.output or f"{args.model_type}_model.onnx"
    onnx_model_path = out_name if os.path.isabs(out_name) else os.path.join(os.getcwd(), out_name)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_model_path,
        export_params=True,
        opset_version=args.opset,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print(f"Exported {args.model_type} ONNX model to {onnx_model_path}")


if __name__ == "__main__":
    main()
