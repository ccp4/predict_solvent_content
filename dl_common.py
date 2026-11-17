"""
Shared multi-modal building blocks for 3D Patterson deep learning models:
  - Extraction & standardization of the 17 permutation-invariant pre-phasing
    diffraction / crystal-lattice metadata features.
  - Missing-value-safe `MetadataFusionHead` with FiLM modulation and modality
    dropout (allowing inference with full .mtz/.cif metadata, partial .map
    metadata, or zero metadata on .npy grids).
  - 3D Multi-Statistic Contrast Pooling (`[AvgPool, StdPool, MaxPool]`) and
    3D Concentric Radial Shell Pooling around the Patterson origin.
"""

import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

from patterson import PattersonResult

# 17 rotation- and axis-permutation-invariant pre-phasing features
DL_METADATA_COLUMNS: Tuple[str, ...] = (
    "origin_to_sigma",
    "log_cell_volume",
    "log_asu_volume",
    "asu_equiv_radius",
    "cell_min_len",
    "cell_mid_len",
    "cell_max_len",
    "cell_anisotropy",
    "spacegroup_number",
    "point_group_order",
    "n_symops",
    "native_d_min",
    "d_max",
    "wilson_b_est",
    "low_res_intensity_ratio",
    "acentric_e2_minus_1",
    "mean_e4",
)

NUM_DL_METADATA: int = len(DL_METADATA_COLUMNS)

# Sensible crystallographic priors before fitting empirical dataset statistics
_DEFAULT_META_MEAN = [
    45.0,   # origin_to_sigma
    12.5,   # log_cell_volume (~270,000 A^3)
    11.0,   # log_asu_volume (~60,000 A^3)
    24.0,   # asu_equiv_radius (A)
    50.0,   # cell_min_len (A)
    75.0,   # cell_mid_len (A)
    110.0,  # cell_max_len (A)
    2.2,    # cell_anisotropy
    40.0,   # spacegroup_number
    4.0,    # point_group_order
    6.0,    # n_symops
    2.2,    # native_d_min (A)
    40.0,   # d_max (A)
    30.0,   # wilson_b_est (A^2)
    0.6,    # low_res_intensity_ratio
    0.74,   # acentric_e2_minus_1
    2.0,    # mean_e4
]

_DEFAULT_META_STD = [
    25.0,
    1.2,
    1.1,
    10.0,
    25.0,
    35.0,
    55.0,
    1.2,
    55.0,
    4.0,
    8.0,
    0.7,
    20.0,
    18.0,
    0.4,
    0.12,
    0.5,
]


def patterson_result_to_metadata_vec(
    res: PattersonResult,
    d_min_hint: Optional[float] = None,
) -> np.ndarray:
    """Convert a `PattersonResult` into a `(17,)` float32 vector (`NaN` where unknown)."""
    native_d = res.native_d_min
    if (not math.isfinite(native_d)) and d_min_hint is not None:
        native_d = float(d_min_hint)

    log_cv = math.log(res.cell_volume) if (math.isfinite(res.cell_volume) and res.cell_volume > 1.0) else float("nan")
    log_av = math.log(res.asu_volume) if (math.isfinite(res.asu_volume) and res.asu_volume > 1.0) else float("nan")
    sg_num = float(res.spacegroup_number) if res.spacegroup_number > 0 else float("nan")
    pg_ord = float(res.point_group_order) if res.point_group_order > 0 else float("nan")
    nsym = float(res.n_symops) if res.n_symops > 0 else float("nan")

    return np.array(
        [
            res.origin_to_sigma,
            log_cv,
            log_av,
            res.asu_equiv_radius,
            res.cell_min_len,
            res.cell_mid_len,
            res.cell_max_len,
            res.cell_anisotropy,
            sg_num,
            pg_ord,
            nsym,
            native_d,
            res.d_max,
            res.wilson_b_est,
            res.low_res_intensity_ratio,
            res.acentric_e2_minus_1,
            res.mean_e4,
        ],
        dtype=np.float32,
    )


def load_dl_metadata_tsv(tsv_path: Optional[Path]) -> Dict[str, np.ndarray]:
    """Load `{name_lower: (17,) float32 metadata_vector}` from `map_metadata.tsv`."""
    if tsv_path is None or not tsv_path.is_file():
        return {}

    out: Dict[str, np.ndarray] = {}
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            name = row["name"].strip().lower()

            def _gf(key: str) -> float:
                raw = row.get(key)
                if raw is None or raw == "":
                    return float("nan")
                try:
                    return float(raw)
                except ValueError:
                    return float("nan")

            cv = _gf("cell_volume")
            av = _gf("asu_volume")
            sg = _gf("spacegroup_number")
            pg = _gf("point_group_order")
            ns = _gf("n_symops")
            asu_r = _gf("asu_equiv_radius")
            if not math.isfinite(asu_r) and math.isfinite(av) and av > 0:
                asu_r = ((3.0 * av) / (4.0 * math.pi)) ** (1.0 / 3.0)

            vec = np.array(
                [
                    _gf("origin_to_sigma"),
                    math.log(cv) if (math.isfinite(cv) and cv > 1.0) else float("nan"),
                    math.log(av) if (math.isfinite(av) and av > 1.0) else float("nan"),
                    asu_r,
                    _gf("cell_min_len"),
                    _gf("cell_mid_len"),
                    _gf("cell_max_len"),
                    _gf("cell_anisotropy"),
                    sg if (math.isfinite(sg) and sg > 0) else float("nan"),
                    pg if (math.isfinite(pg) and pg > 0) else float("nan"),
                    ns if (math.isfinite(ns) and ns > 0) else float("nan"),
                    _gf("native_d_min"),
                    _gf("d_max"),
                    _gf("wilson_b_est"),
                    _gf("low_res_intensity_ratio"),
                    _gf("acentric_e2_minus_1"),
                    _gf("mean_e4"),
                ],
                dtype=np.float32,
            )
            out[name] = vec
    return out


class ContrastPool3D(nn.Module):
    """
    Concatenate `[GlobalAvgPool3D, GlobalStdPool3D, GlobalMaxPool3D]` across
    spatial dimensions `(D, H, W)` so downstream heads observe mean density,
    packing contrast (standard deviation), and peak extrema simultaneously.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[0], x.shape[1]
        mean = self.avg_pool(x).view(b, c)
        mx = self.max_pool(x).view(b, c)
        # Compute population variance safely in float32 under AMP
        flat = x.float().view(b, c, -1)
        var = torch.mean((flat - flat.mean(dim=-1, keepdim=True)) ** 2, dim=-1)
        std = torch.sqrt(var + self.eps).to(x.dtype)
        return torch.cat([mean, std, mx], dim=1)


class RadialShellPool3D(nn.Module):
    """
    Pool per-channel mean & variance across 4 concentric spherical shells around
    the grid center `(D/2, H/2, W/2)`, giving explicit radial envelope contrast
    from intramolecular vectors (inner shell) to intermolecular regions (outer shell).
    """

    def __init__(self, spatial_size: int = 32, num_shells: int = 4, eps: float = 1e-6) -> None:
        super().__init__()
        self.spatial_size = spatial_size
        self.num_shells = num_shells
        self.eps = eps
        coords = (torch.arange(spatial_size, dtype=torch.float32) + 0.5) / float(spatial_size) - 0.5
        r = torch.sqrt(coords[:, None, None] ** 2 + coords[None, :, None] ** 2 + coords[None, None, :] ** 2)
        edges = torch.linspace(0.0, 0.5001, steps=num_shells + 1)
        weights = []
        for s in range(num_shells):
            mask = ((r >= edges[s]) & (r < edges[s + 1])).float().reshape(-1)
            w = mask / mask.sum().clamp(min=1.0)
            weights.append(w)
        # Shape: (D*H*W, num_shells) for single-GEMM matmul pooling
        self.register_buffer("shell_weights_flat", torch.stack(weights, dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_d = self.spatial_size
        if x.shape[-1] != target_d or x.shape[-2] != target_d or x.shape[-3] != target_d:
            x = nn.functional.interpolate(
                x, size=(target_d, target_d, target_d), mode="trilinear", align_corners=False
            )

        xf = x.float().flatten(2)  # (B, C, K)
        w = self.shell_weights_flat.float()  # (K, S)
        means = torch.matmul(xf, w)  # (B, C, S)
        sq_means = torch.matmul(xf * xf, w)  # (B, C, S)
        stds = torch.sqrt(torch.clamp(sq_means - means * means, min=0.0) + self.eps)
        pooled = torch.cat([means.flatten(1), stds.flatten(1)], dim=1)
        return pooled.to(x.dtype)


class MetadataFusionHead(nn.Module):
    """
    Missing-value-safe pre-phasing metadata encoder + FiLM modulation + MLP head.

    Stores dataset `meta_mean` and `meta_std` as registered buffers so ONNX export
    and standalone inference are 100% self-contained with zero external scalers.
    During training, randomly drops metadata (`modality_dropout`) so the 3D backbone
    remains accurate even when some or all metadata fields are `NaN` at inference.
    """

    def __init__(
        self,
        vol_dim: int,
        meta_dim: int = NUM_DL_METADATA,
        hidden_dim: int = 128,
        dropout_prob: float = 0.15,
        modality_dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.vol_dim = vol_dim
        self.meta_dim = meta_dim
        self.modality_dropout = modality_dropout

        self.register_buffer("meta_mean", torch.tensor(_DEFAULT_META_MEAN[:meta_dim], dtype=torch.float32).unsqueeze(0))
        self.register_buffer("meta_std", torch.tensor(_DEFAULT_META_STD[:meta_dim], dtype=torch.float32).unsqueeze(0))

        self.vol_norm = nn.LayerNorm(vol_dim)
        self.meta_encoder = nn.Sequential(
            nn.Linear(meta_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        # FiLM feature-wise linear modulation of 3D pooled representation by metadata
        self.film = nn.Linear(hidden_dim, vol_dim * 2)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

        self.head = nn.Sequential(
            nn.Linear(vol_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    @torch.no_grad()
    def fit_metadata_scaler(self, meta_matrix: np.ndarray) -> None:
        """Fit `meta_mean` and `meta_std` buffers from finite training set metadata."""
        if meta_matrix.ndim != 2 or meta_matrix.shape[1] != self.meta_dim:
            return
        means: List[float] = []
        stds: List[float] = []
        for j in range(self.meta_dim):
            col = meta_matrix[:, j]
            fin = col[np.isfinite(col)]
            if fin.size >= 4 and float(np.std(fin)) > 1e-6:
                means.append(float(np.mean(fin)))
                stds.append(float(np.std(fin)))
            else:
                means.append(float(self.meta_mean[0, j].item()))
                stds.append(float(self.meta_std[0, j].item()))
        self.meta_mean.copy_(torch.tensor(means, dtype=torch.float32, device=self.meta_mean.device).unsqueeze(0))
        self.meta_std.copy_(torch.tensor(stds, dtype=torch.float32, device=self.meta_std.device).unsqueeze(0))

    def forward(
        self,
        vol_features: torch.Tensor,
        metadata: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz = vol_features.shape[0]
        v = self.vol_norm(vol_features)

        if metadata is None:
            norm_meta = torch.zeros(bsz, self.meta_dim, dtype=v.dtype, device=v.device)
            present = torch.zeros(bsz, self.meta_dim, dtype=v.dtype, device=v.device)
        else:
            meta_f = metadata.float()
            finite = ~torch.isnan(meta_f) & ~torch.isinf(meta_f)
            if self.training and self.modality_dropout > 0.0:
                # Randomly mask whole-sample metadata or individual fields during training
                drop_sample = (torch.rand(bsz, 1, device=v.device) >= self.modality_dropout)
                drop_field = (torch.rand(bsz, self.meta_dim, device=v.device) >= (0.5 * self.modality_dropout))
                finite = finite & drop_sample & drop_field

            present = finite.to(v.dtype)
            safe_meta = torch.where(finite, meta_f, self.meta_mean)
            scaled = torch.clamp((safe_meta - self.meta_mean) / self.meta_std.clamp(min=1e-6), -8.0, 8.0)
            norm_meta = scaled.to(v.dtype) * present

        meta_in = torch.cat([norm_meta, present], dim=1)
        meta_emb = self.meta_encoder(meta_in)

        gamma_beta = self.film(meta_emb)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        v_mod = v * (1.0 + 0.5 * torch.tanh(gamma)) + beta

        fused = torch.cat([v_mod, meta_emb], dim=1)
        return self.head(fused)
