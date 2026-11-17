"""
Script for training the multi-modal 3D ResidualUNetSE3D (`unet`) and ViTVNet (`vitvnet`)
solvent-content models with pre-phasing diffraction/lattice metadata fusion.

Priority 3 training features:
1. 3D octahedral cube-symmetry augmentations (48 origin-preserving rotations/reflections).
2. Weighted Smooth L1 (Huber) loss with Inverse Label-Density Weighting (LDS) to
   prevent regression-to-the-mean while resisting extreme PDB outlier labels.
3. Automatic Mixed Precision (`torch.amp.autocast`), `AdamW` + linear warmup +
   `CosineAnnealingLR`, gradient clipping, and `DistributedDataParallel` (`torchrun`).

Usage:
    python train.py training_data.txt validation_data.txt unet --map-metadata map_metadata.tsv
    torchrun --nproc_per_node=2 train.py training_data.txt validation_data.txt unet
"""

import argparse
import math
import os
from pathlib import Path
from random import shuffle
import time
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dl_common import load_dl_metadata_tsv
from get_data import MapDataset
from patterson import DEFAULT_GRID_SIZE
from usolv import ResidualUNetSE3D
from vitsolv import ViTVNet, get_3DReg_config


class WeightedMSE(nn.Module):
    """Legacy tail-weighted MSE loss."""

    def __init__(self, weight: float = 8.0) -> None:
        super().__init__()
        self.weight = weight

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        per_sample = (predictions - targets) ** 2 * (torch.abs(targets - 0.5) * self.weight + 1.0)
        if sample_weights is not None:
            per_sample = per_sample * sample_weights
        return torch.mean(per_sample)


class WeightedHuberLoss(nn.Module):
    """
    Outlier-resilient Smooth L1 (Huber) loss combined with Inverse Label-Density
    Weighting (LDS) so extreme high/low solvent crystals receive balanced gradient
    signal without chasing mis-annotated PDB outliers.
    """

    def __init__(self, delta: float = 0.04, tail_boost: float = 2.0) -> None:
        super().__init__()
        self.delta = delta
        self.tail_boost = tail_boost

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        base = F.huber_loss(predictions, targets, reduction="none", delta=self.delta)
        if sample_weights is not None:
            return torch.mean(base * sample_weights)
        tail_factor = 1.0 + self.tail_boost * torch.abs(targets - 0.5)
        return torch.mean(base * tail_factor)


def compute_lds_weights(targets_01: np.ndarray, bandwidth: float = 0.04) -> np.ndarray:
    """
    Compute Inverse Label-Density Smoothing weights `1 / sqrt(KDE(y))` over
    training targets in [0, 1], normalized to mean 1.0 and clipped to [0.5, 6.0].
    """
    y = targets_01.ravel().astype(np.float64)
    if y.size < 4:
        return np.ones_like(y, dtype=np.float32)
    diffs = (y[:, None] - y[None, :]) / max(0.005, bandwidth)
    kde = np.mean(np.exp(-0.5 * diffs * diffs), axis=1)
    w = 1.0 / np.sqrt(np.maximum(kde, 1e-6))
    w /= np.mean(w)
    return np.clip(w, 0.5, 6.0).astype(np.float32)


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


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, (nn.DataParallel, DDP)) else model


def _build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    total_epochs: int,
    warmup_epochs: int,
    min_lr_ratio: float = 0.02,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup = max(0, min(warmup_epochs, total_epochs - 1))

    def _lr_lambda(epoch: int) -> float:
        if warmup > 0 and epoch < warmup:
            return float(epoch + 1) / float(warmup)
        if total_epochs <= warmup + 1:
            return 1.0
        progress = float(epoch - warmup) / float(max(1, total_epochs - warmup - 1))
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)


def main() -> None:
    default_workers = min(8, os.cpu_count() or 1)
    parser = argparse.ArgumentParser(description="Train 3D Patterson + metadata solvent content models.")
    parser.add_argument("training_file", help="Path to training split file (name solvent_pct npy_path).")
    parser.add_argument("validation_file", help="Path to validation split file.")
    parser.add_argument("model_type", choices=("unet", "vitvnet"), help="Model architecture.")
    parser.add_argument(
        "--map-metadata",
        type=Path,
        default=None,
        help="Path to map_metadata.tsv for pre-phasing metadata fusion (auto-detected if ./map_metadata.tsv exists).",
    )
    parser.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE[0])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--loss", choices=("huber", "mse"), default="huber")
    parser.add_argument("--huber-delta", type=float, default=0.04, help="Huber delta on [0, 1] scale (0.04 = 4% solvent).")
    parser.add_argument("--loss-weight", type=float, default=8.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=default_workers)
    parser.add_argument("--no-augment", action="store_true", help="Disable 3D cube-symmetry training augmentations.")
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA automatic mixed precision.")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from.")
    parser.add_argument("--epoch-start", type=int, default=0, help="Starting epoch index when resuming.")
    args = parser.parse_args()

    # DistributedDataParallel setup if launched via `torchrun`
    is_ddp = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if is_ddp:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")
        is_main = dist.get_rank() == 0
    else:
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_main = True

    abs_path = os.getcwd()
    training_input_file_path = (
        args.training_file if os.path.isabs(args.training_file) else os.path.join(abs_path, args.training_file)
    )
    validation_input_file_path = (
        args.validation_file if os.path.isabs(args.validation_file) else os.path.join(abs_path, args.validation_file)
    )

    meta_tsv = args.map_metadata
    if meta_tsv is None:
        default_meta = Path(abs_path) / "map_metadata.tsv"
        if default_meta.is_file():
            meta_tsv = default_meta
    metadata_map = load_dl_metadata_tsv(meta_tsv)

    if is_main:
        print(f"Using device: {device} (DDP={is_ddp})")
        if device.type == "cuda":
            print(torch.cuda.get_device_name(local_rank))
        if metadata_map:
            print(f"Loaded pre-phasing metadata for {len(metadata_map)} entries from {meta_tsv}")

    (
        training_input,
        training_output,
        training_names,
        validation_input,
        validation_output,
        validation_names,
    ) = get_training_data(training_input_file_path, validation_input_file_path)

    lds_weights = compute_lds_weights(training_output)
    training_data = MapDataset(
        training_input,
        training_output,
        training_names,
        metadata_map=metadata_map,
        sample_weights=lds_weights,
        augment=not args.no_augment,
        return_extra=True,
    )
    validation_data = MapDataset(
        validation_input,
        validation_output,
        validation_names,
        metadata_map=metadata_map,
        augment=False,
        return_extra=True,
    )

    if is_main:
        print(f"Training data size: {len(training_input)} | Validation data size: {len(validation_input)}")

    img_size = (args.grid_size, args.grid_size, args.grid_size)
    if args.model_type == "vitvnet":
        config = get_3DReg_config(grid_size=args.grid_size)
        base_model: nn.Module = ViTVNet(config=config, img_size=img_size)
    else:
        base_model = ResidualUNetSE3D(
            in_channels=1,
            f_maps=[16, 32, 64, 128, 256],
        )

    # Fit internal metadata standardization buffers directly from finite training metadata
    base_model.fit_metadata_scaler(training_data.metadata.numpy())  # type: ignore[operator]

    if args.resume is not None:
        if is_main:
            print(f"Loading model {args.resume} (starting at epoch {args.epoch_start})")
        ckpt_obj = torch.load(args.resume, map_location="cpu", weights_only=True)
        state_dict = ckpt_obj["state_dict"] if isinstance(ckpt_obj, dict) and "state_dict" in ckpt_obj else ckpt_obj
        clean_state = {k.removeprefix("module."): v for k, v in state_dict.items()}
        base_model.load_state_dict(clean_state)

    model: nn.Module = base_model.to(device)
    if is_ddp:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)
    elif device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    pin_mem = device.type == "cuda"
    train_sampler: Optional[DistributedSampler] = DistributedSampler(training_data, shuffle=True) if is_ddp else None
    training_loader = DataLoader(
        training_data,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=pin_mem,
        persistent_workers=(args.num_workers > 0),
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_mem,
        persistent_workers=(args.num_workers > 0),
    )

    loss_fn: nn.Module = (
        WeightedHuberLoss(delta=args.huber_delta)
        if args.loss == "huber"
        else WeightedMSE(weight=args.loss_weight)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = _build_warmup_cosine_scheduler(
        optimizer, total_epochs=args.epochs, warmup_epochs=args.warmup_epochs
    )
    for _ in range(args.epoch_start):
        scheduler.step()

    use_amp = (device.type == "cuda") and (not args.no_amp)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    models_dir = os.path.join(abs_path, "models")
    if is_main:
        os.makedirs(models_dir, exist_ok=True)

    best_val_rmse = float("inf")

    try:
        for epoch in range(args.epoch_start, args.epochs):
            t0 = time.time()
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            model.train()
            train_losses: List[float] = []
            train_preds: List[float] = []
            train_targets: List[float] = []

            train_corr_path = os.path.join(abs_path, f"training_correlation_{epoch}.txt")
            training_file = open(train_corr_path, "w", encoding="utf-8") if is_main else None
            try:
                for inputs, metas, labels, weights, names in training_loader:
                    inputs = inputs.to(device, dtype=torch.float32, non_blocking=pin_mem)
                    metas = metas.to(device, dtype=torch.float32, non_blocking=pin_mem)
                    labels = labels.to(device, dtype=torch.float32, non_blocking=pin_mem)
                    weights = weights.to(device, dtype=torch.float32, non_blocking=pin_mem)

                    optimizer.zero_grad(set_to_none=True)
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        outputs = model(inputs, metas)
                        loss = loss_fn(outputs, labels, weights)

                    scaler.scale(loss).backward()
                    if args.grad_clip > 0.0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()

                    train_losses.append(float(loss.item()))
                    lbl_np = labels.detach().cpu().numpy().ravel()
                    out_np = outputs.detach().float().cpu().numpy().ravel()
                    train_targets.extend(lbl_np.tolist())
                    train_preds.extend(out_np.tolist())
                    if training_file is not None:
                        for name, t_val, p_val in zip(names, lbl_np, out_np):
                            training_file.write(f"{name} {t_val:.6f} {p_val:.6f}\n")
            finally:
                if training_file is not None:
                    training_file.close()

            scheduler.step()

            if is_main:
                raw_state = _unwrap_model(model).state_dict()
                ckpt_path = os.path.join(models_dir, f"model_{epoch}")
                torch.save(raw_state, ckpt_path)

                model.eval()
                val_losses: List[float] = []
                val_preds: List[float] = []
                val_targets: List[float] = []
                val_corr_path = os.path.join(abs_path, f"validation_correlation_{epoch}.txt")
                with open(val_corr_path, "w", encoding="utf-8") as validation_file, torch.no_grad():
                    for vinputs, vmetas, vlabels, _, vnames in validation_loader:
                        vinputs = vinputs.to(device, dtype=torch.float32, non_blocking=pin_mem)
                        vmetas = vmetas.to(device, dtype=torch.float32, non_blocking=pin_mem)
                        vlabels = vlabels.to(device, dtype=torch.float32, non_blocking=pin_mem)
                        with torch.amp.autocast("cuda", enabled=use_amp):
                            voutputs = _unwrap_model(model)(vinputs, vmetas)
                            vloss = loss_fn(voutputs, vlabels)

                        val_losses.append(float(vloss.item()))
                        vlbl_np = vlabels.cpu().numpy().ravel()
                        vout_np = voutputs.float().cpu().numpy().ravel()
                        val_targets.extend(vlbl_np.tolist())
                        val_preds.extend(vout_np.tolist())
                        for name, t_val, p_val in zip(vnames, vlbl_np, vout_np):
                            validation_file.write(f"{name} {t_val:.6f} {p_val:.6f}\n")

                tr_t = np.asarray(train_targets) * 100.0
                tr_p = np.asarray(train_preds) * 100.0
                va_t = np.asarray(val_targets) * 100.0
                va_p = np.asarray(val_preds) * 100.0
                val_rmse = float(np.sqrt(np.mean((va_t - va_p) ** 2))) if len(va_t) > 0 else float("nan")
                cur_lr = optimizer.param_groups[0]["lr"]

                if math.isfinite(val_rmse) and val_rmse <= best_val_rmse:
                    best_val_rmse = val_rmse
                    best_path = os.path.join(models_dir, f"best_{args.model_type}.pt")
                    torch.save(
                        {
                            "state_dict": raw_state,
                            "model_type": args.model_type,
                            "grid_size": args.grid_size,
                            "epoch": epoch,
                            "val_rmse": val_rmse,
                        },
                        best_path,
                    )

                elapsed = time.time() - t0
                print(
                    f"Epoch {epoch:02d} ({elapsed:.1f}s, lr={cur_lr:.2e}) | "
                    f"Train Loss={np.mean(train_losses):.5f}, MAE={np.mean(np.abs(tr_t - tr_p)):.2f}%, r={_pearson_r(tr_t, tr_p):.4f} | "
                    f"Val Loss={np.mean(val_losses):.5f}, RMSE={val_rmse:.2f}%, MAE={np.mean(np.abs(va_t - va_p)):.2f}%, r={_pearson_r(va_t, va_p):.4f}"
                )
    finally:
        if is_ddp and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
