"""
Hybrid 3D CNN + Vision Transformer (`ViTVNet`) upgraded for predicting protein
crystal solvent content from 3D Patterson maps + pre-phasing diffraction metadata.
"""

import copy
import math
from typing import Optional
import torch
import torch.nn as nn
from torch.nn import Conv3d, Dropout, LayerNorm, Linear, Softmax
from torch.nn.modules.utils import _triple

from dl_common import NUM_DL_METADATA, MetadataFusionHead

try:
    import ml_collections
except ImportError:
    ml_collections = None


class _AttrDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value


def _make_config_dict(initial=None):
    if ml_collections is not None:
        return ml_collections.ConfigDict(initial or {})
    return _AttrDict(initial or {})


def _gaussian_rbf(
    distances: torch.Tensor, num_kernels: int, d_max: float
) -> torch.Tensor:
    centers = torch.linspace(
        0.0, d_max, steps=num_kernels, device=distances.device, dtype=distances.dtype
    )
    gamma = float(num_kernels) / max(1e-4, d_max)
    diff = distances.unsqueeze(-1) - centers
    return torch.exp(-(gamma**2) * (diff**2))


class Attention(nn.Module):
    def __init__(self, config, vis: bool = False) -> None:
        super().__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.rel_dist_proj = Linear(16, self.num_attention_heads)

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states: torch.Tensor, pair_rbf: torch.Tensor):
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Add 3D rotation-invariant pairwise relative distance bias: (1, H, N, N)
        rel_bias = (
            self.rel_dist_proj(pair_rbf.to(hidden_states.dtype))
            .permute(2, 0, 1)
            .unsqueeze(0)
        )
        attention_scores = attention_scores + rel_bias

        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.proj_dropout(self.out(context_layer))
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = torch.nn.functional.gelu
        self.dropout = Dropout(config.transformer["dropout_rate"])
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.act_fn(self.fc1(x)))
        x = self.dropout(self.fc2(x))
        return x


class Block(nn.Module):
    def __init__(self, config, vis: bool = False) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x: torch.Tensor, pair_rbf: torch.Tensor):
        h = x
        x, weights = self.attn(self.attention_norm(x), pair_rbf)
        x = x + h

        h = x
        x = self.ffn(self.ffn_norm(x))
        x = x + h
        return x, weights


class Encoder(nn.Module):
    def __init__(self, config, vis: bool = False) -> None:
        super().__init__()
        self.vis = vis
        self.layer = nn.ModuleList(
            [
                copy.deepcopy(Block(config, vis))
                for _ in range(config.transformer["num_layers"])
            ]
        )
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)

    def forward(self, hidden_states: torch.Tensor, pair_rbf: torch.Tensor):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states, pair_rbf)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class DoubleConv(nn.Module):
    """(Conv3d => GroupNorm => ELU) * 2."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: Optional[int] = None,
        num_groups: int = 8,
    ) -> None:
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        g1 = min(num_groups, mid_channels)
        while g1 > 1 and mid_channels % g1 != 0:
            g1 -= 1
        g2 = min(num_groups, out_channels)
        while g2 > 1 and out_channels % g2 != 0:
            g2 -= 1
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(g1, mid_channels),
            nn.ELU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(g2, out_channels),
            nn.ELU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class CNNEncoder(nn.Module):
    """Memory-efficient strided stem + AvgPool3d 3D feature stem (`N^3 -> (N/4)^3`)."""

    def __init__(self, config, n_channels: int = 1) -> None:
        super().__init__()
        enc_ch = config.encoder_channels
        g0 = min(8, enc_ch[0])
        while g0 > 1 and enc_ch[0] % g0 != 0:
            g0 -= 1
        self.stem = nn.Sequential(
            nn.Conv3d(
                n_channels, enc_ch[0], kernel_size=5, stride=2, padding=2, bias=False
            ),
            nn.GroupNorm(g0, enc_ch[0]),
            nn.ELU(inplace=True),
            DoubleConv(enc_ch[0], enc_ch[1]),
        )
        self.down = nn.Sequential(
            nn.AvgPool3d(kernel_size=2, stride=2),
            DoubleConv(enc_ch[1], enc_ch[2]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.stem(x))


class Embeddings(nn.Module):
    """
    Patch tokenizer with 3D rotation/reflection-invariant radial origin RBF
    embeddings and precomputed 3D pairwise patch-distance RBF basis.
    """

    def __init__(self, config, img_size=(128, 128, 128)) -> None:
        super().__init__()
        self.config = config
        down_factor = config.down_factor
        patch_size = _triple(config.patches["size"])

        gx = int(img_size[0] // (2**down_factor) // patch_size[0])
        gy = int(img_size[1] // (2**down_factor) // patch_size[1])
        gz = int(img_size[2] // (2**down_factor) // patch_size[2])

        self.hybrid_model = CNNEncoder(config, n_channels=1)
        in_channels = config["encoder_channels"][-1]
        self.patch_embeddings = Conv3d(
            in_channels=in_channels,
            out_channels=config.hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # Normalized 3D patch center coordinates in [-0.5, +0.5]^3
        cx = (torch.arange(gx, dtype=torch.float32) + 0.5) / float(gx) - 0.5
        cy = (torch.arange(gy, dtype=torch.float32) + 0.5) / float(gy) - 0.5
        cz = (torch.arange(gz, dtype=torch.float32) + 0.5) / float(gz) - 0.5
        coords = torch.stack(torch.meshgrid(cx, cy, cz, indexing="ij"), dim=-1).reshape(
            -1, 3
        )

        # Radial distance from Patterson origin (0, 0, 0) -> (1, N, 16)
        origin_dist = torch.linalg.norm(coords, dim=-1)
        origin_rbf = _gaussian_rbf(
            origin_dist, num_kernels=16, d_max=math.sqrt(0.75)
        ).unsqueeze(0)
        self.register_buffer("origin_rbf", origin_rbf)
        self.origin_pos_proj = Linear(16, config.hidden_size)

        # Pairwise 3D Euclidean patch distances -> (N, N, 16)
        pair_dist = torch.cdist(coords, coords, p=2.0)
        pair_rbf = _gaussian_rbf(pair_dist, num_kernels=16, d_max=math.sqrt(3.0))
        self.register_buffer("pair_rbf", pair_rbf)

        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x: torch.Tensor):
        x = self.hybrid_model(x)
        x = self.patch_embeddings(x)
        x = x.flatten(2).transpose(-1, -2)
        radial_pos = self.origin_pos_proj(self.origin_rbf.to(x.dtype))
        embeddings = self.dropout(x + radial_pos)
        return embeddings, self.pair_rbf


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis: bool = False) -> None:
        super().__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(self, input_vol: torch.Tensor):
        embedding_output, pair_rbf = self.embeddings(input_vol)
        encoded, attn_weights = self.encoder(embedding_output, pair_rbf)
        return encoded, attn_weights


class ViTVNet(nn.Module):
    def __init__(
        self,
        config,
        img_size=(128, 128, 128),
        vis: bool = False,
        meta_dim: int = NUM_DL_METADATA,
    ) -> None:
        super().__init__()
        self.transformer = Transformer(config, img_size, vis)
        self.config = config
        vol_dim = 2 * config.hidden_size
        self.fusion_head = MetadataFusionHead(
            vol_dim=vol_dim,
            meta_dim=meta_dim,
            hidden_dim=128,
            dropout_prob=config.transformer["dropout_rate"],
        )

    def fit_metadata_scaler(self, meta_matrix) -> None:
        self.fusion_head.fit_metadata_scaler(meta_matrix)

    def forward(
        self,
        x: torch.Tensor,
        metadata: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        tokens, _ = self.transformer(x)
        mean_tok = tokens.mean(dim=1)
        # Token std across spatial patches captures intermolecular vs intramolecular contrast
        tok_f = tokens.float()
        var_tok = torch.mean((tok_f - tok_f.mean(dim=1, keepdim=True)) ** 2, dim=1)
        std_tok = torch.sqrt(var_tok + 1e-6).to(tokens.dtype)
        vol_feat = torch.cat([mean_tok, std_tok], dim=1)
        return self.fusion_head(vol_feat, metadata)


def get_3DReg_config(grid_size: int = 128):
    patch = max(1, grid_size // 32)
    config = _make_config_dict()
    config.patches = _make_config_dict(
        {"size": (patch, patch, patch), "grid": (8, 8, 8)}
    )
    config.hidden_size = 252
    config.transformer = _make_config_dict(
        {
            "mlp_dim": 1008,
            "num_heads": 12,
            "num_layers": 8,
            "attention_dropout_rate": 0.05,
            "dropout_rate": 0.1,
        }
    )
    config.patch_size = patch
    config.encoder_channels = (16, 32, 32)
    config.down_factor = 2
    config.n_dims = 3
    return config
