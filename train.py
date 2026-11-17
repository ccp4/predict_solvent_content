"""
Script for training the unet and vitvnet models.

Usage:
    python train.py training_data.txt validation_data.txt unet
    python train.py training_data.txt validation_data.txt vitvnet --resume models/model_5 --epoch-start 6
"""

import argparse
import os
from random import shuffle
import time
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from get_data import MapDataset
from patterson import DEFAULT_GRID_SIZE
from usolv import ResidualUNetSE3D
from vitsolv import ViTVNet, get_3DReg_config


class WeightedMSE(nn.Module):
    """
    Custom MSE cost function to weight structures with extreme high/low
    solvent content more heavily.
    """

    def __init__(self, weight: float):
        super(WeightedMSE, self).__init__()
        self.weight = weight

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return torch.mean(
            (predictions - targets) ** 2 * (torch.abs(targets - 0.5) * self.weight + 1.0)
        )


def _read_split_file(file_path: str, do_shuffle: bool = False) -> Tuple[List[str], np.ndarray, List[str]]:
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    if do_shuffle:
        shuffle(lines)

    paths: List[str] = []
    targets: List[float] = []
    names: List[str] = []

    for line in lines:
        ls = line.split()
        names.append(ls[0])
        val = float(ls[1])
        # Normalize percentage [0, 100] to fraction [0, 1] if needed
        targets.append(val / 100.0 if val > 1.0 else val)
        paths.append(ls[2])

    target_arr = np.expand_dims(np.asarray(targets, dtype=np.float32), axis=1)
    return paths, target_arr, names


def get_training_data(
    training_input_file_path: str,
    validation_input_file_path: str,
) -> Tuple[List[str], np.ndarray, List[str], List[str], np.ndarray, List[str]]:
    train_in, train_out, train_names = _read_split_file(training_input_file_path, do_shuffle=True)
    val_in, val_out, val_names = _read_split_file(validation_input_file_path, do_shuffle=False)
    return train_in, train_out, train_names, val_in, val_out, val_names


def _pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return float("nan")
    return float(np.corrcoef(y_true.ravel(), y_pred.ravel())[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train 3D Patterson solvent content models.")
    parser.add_argument("training_file", help="Path to training split file (name solvent_pct npy_path).")
    parser.add_argument("validation_file", help="Path to validation split file.")
    parser.add_argument("model_type", choices=("unet", "vitvnet"), help="Model architecture.")
    parser.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE[0])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--loss-weight", type=float, default=8.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from.")
    parser.add_argument("--epoch-start", type=int, default=0, help="Starting epoch index when resuming.")
    args = parser.parse_args()

    abs_path = os.getcwd()
    training_input_file_path = (
        args.training_file if os.path.isabs(args.training_file) else os.path.join(abs_path, args.training_file)
    )
    validation_input_file_path = (
        args.validation_file if os.path.isabs(args.validation_file) else os.path.join(abs_path, args.validation_file)
    )

    img_size = (args.grid_size, args.grid_size, args.grid_size)
    if args.model_type == "vitvnet":
        config = get_3DReg_config(grid_size=args.grid_size)
        model: nn.Module = ViTVNet(config=config, img_size=img_size)
    else:
        model = ResidualUNetSE3D(
            in_channels=1,
            f_maps=[16, 32, 64, 128, 256],
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if device.type == "cuda":
        print(torch.cuda.get_device_name(0))
        torch.cuda.empty_cache()
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

    model = model.to(device)

    (
        training_input,
        training_output,
        training_names,
        validation_input,
        validation_output,
        validation_names,
    ) = get_training_data(training_input_file_path, validation_input_file_path)

    training_data = MapDataset(training_input, training_output, training_names)
    validation_data = MapDataset(validation_input, validation_output, validation_names)
    print(f"Training data size: {len(training_input)}")
    print(f"Validation data size: {len(validation_input)}")

    pin_mem = device.type == "cuda"
    training_loader = DataLoader(
        training_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_mem,
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_mem,
    )

    loss_fn = WeightedMSE(args.loss_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        amsgrad=True,
    )

    if args.resume is not None:
        print(f"Loading model {args.resume} (starting at epoch {args.epoch_start})")
        state_dict = torch.load(args.resume, map_location=device, weights_only=True)
        if isinstance(model, nn.DataParallel) and not any(k.startswith("module.") for k in state_dict):
            model.module.load_state_dict(state_dict)
        elif not isinstance(model, nn.DataParallel) and any(k.startswith("module.") for k in state_dict):
            model.load_state_dict({k.removeprefix("module."): v for k, v in state_dict.items()})
        else:
            model.load_state_dict(state_dict)

    models_dir = os.path.join(abs_path, "models")
    os.makedirs(models_dir, exist_ok=True)

    for epoch in range(args.epoch_start, args.epochs):
        t0 = time.time()
        model.train()
        train_losses: List[float] = []
        train_preds: List[float] = []
        train_targets: List[float] = []

        train_corr_path = os.path.join(abs_path, f"training_correlation_{epoch}.txt")
        with open(train_corr_path, "w", encoding="utf-8") as training_file:
            for inputs, labels, names in training_loader:
                inputs = inputs.to(device, dtype=torch.float32, non_blocking=pin_mem)
                labels = labels.to(device, dtype=torch.float32, non_blocking=pin_mem)

                optimizer.zero_grad(set_to_none=True)
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

                train_losses.append(float(loss.item()))
                lbl_np = labels.detach().cpu().numpy().ravel()
                out_np = outputs.detach().cpu().numpy().ravel()
                train_targets.extend(lbl_np.tolist())
                train_preds.extend(out_np.tolist())
                for name, t_val, p_val in zip(names, lbl_np, out_np):
                    training_file.write(f"{name} {t_val:.6f} {p_val:.6f}\n")

        raw_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        ckpt_path = os.path.join(models_dir, f"model_{epoch}")
        torch.save(raw_state, ckpt_path)

        model.eval()
        val_losses: List[float] = []
        val_preds: List[float] = []
        val_targets: List[float] = []
        val_corr_path = os.path.join(abs_path, f"validation_correlation_{epoch}.txt")
        with open(val_corr_path, "w", encoding="utf-8") as validation_file, torch.no_grad():
            for vinputs, vlabels, vnames in validation_loader:
                vinputs = vinputs.to(device, dtype=torch.float32, non_blocking=pin_mem)
                vlabels = vlabels.to(device, dtype=torch.float32, non_blocking=pin_mem)
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels)

                val_losses.append(float(vloss.item()))
                vlbl_np = vlabels.cpu().numpy().ravel()
                vout_np = voutputs.cpu().numpy().ravel()
                val_targets.extend(vlbl_np.tolist())
                val_preds.extend(vout_np.tolist())
                for name, t_val, p_val in zip(vnames, vlbl_np, vout_np):
                    validation_file.write(f"{name} {t_val:.6f} {p_val:.6f}\n")

        tr_t = np.asarray(train_targets) * 100.0
        tr_p = np.asarray(train_preds) * 100.0
        va_t = np.asarray(val_targets) * 100.0
        va_p = np.asarray(val_preds) * 100.0
        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:02d} ({elapsed:.1f}s) | "
            f"Train Loss={np.mean(train_losses):.5f}, MAE={np.mean(np.abs(tr_t - tr_p)):.2f}%, r={_pearson_r(tr_t, tr_p):.4f} | "
            f"Val Loss={np.mean(val_losses):.5f}, MAE={np.mean(np.abs(va_t - va_p)):.2f}%, r={_pearson_r(va_t, va_p):.4f}"
        )


if __name__ == "__main__":
    main()
