import copy
import math
import torch
import torch.nn as nn
from torch.nn import Conv3d, Dropout, LayerNorm, Linear, Softmax
from torch.nn.modules.utils import _triple

try:
    import ml_collections
except ImportError:
    ml_collections = None

"""
ViT-V-Net implementation modified to only use the encoder component and
output a scalar value.

Code adapted from
https://github.com/junyuchen245/ViT-V-Net_for_3D_Image_Registration_Pytorch
"""


class _AttrDict(dict):
    """Lightweight ConfigDict fallback when ml_collections is not installed."""

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


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
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

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class ViTVNet(nn.Module):
    def __init__(self, config, img_size=(128, 128, 128), vis=False):
        super(ViTVNet, self).__init__()
        self.transformer = Transformer(config, img_size, vis)
        self.config = config
        self.to_latent = nn.Identity()
        self.linear_head = nn.Linear(config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x, _ = self.transformer(x)
        x = x.mean(dim=1)
        x = self.to_latent(x)
        return self.sigmoid(self.linear_head(x))


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights


class Embeddings(nn.Module):
    def __init__(self, config, img_size):
        super(Embeddings, self).__init__()
        self.config = config
        down_factor = config.down_factor
        patch_size = _triple(config.patches["size"])
        n_patches = int(
            (img_size[0] // (2**down_factor) // patch_size[0])
            * (img_size[1] // (2**down_factor) // patch_size[1])
            * (img_size[2] // (2**down_factor) // patch_size[2])
        )
        self.hybrid_model = CNNEncoder(config, n_channels=1)
        in_channels = config["encoder_channels"][-1]
        self.patch_embeddings = Conv3d(
            in_channels=in_channels,
            out_channels=config.hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, n_patches, config.hidden_size)
        )
        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        x = self.hybrid_model(x)
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class CNNEncoder(nn.Module):
    def __init__(self, config, n_channels=1):
        super(CNNEncoder, self).__init__()
        self.n_channels = n_channels
        encoder_channels = config.encoder_channels
        self.inc = DoubleConv(n_channels, encoder_channels[0])
        self.down1 = Down(encoder_channels[0], encoder_channels[1])
        self.down2 = Down(encoder_channels[1], encoder_channels[2])
        self.width = encoder_channels[-1]

    def forward(self, x):
        x = self.inc(x)
        x = self.down1(x)
        x = self.down2(x)
        return x


class Down(nn.Module):
    """Downscaling with maxpool then double conv."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class DoubleConv(nn.Module):
    """(Conv3d => GroupNorm => ReLU) * 2."""

    def __init__(self, in_channels, out_channels, mid_channels=None, num_groups=8):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        g1 = min(num_groups, mid_channels)
        while mid_channels % g1 != 0:
            g1 -= 1
        g2 = min(num_groups, out_channels)
        while out_channels % g2 != 0:
            g2 -= 1
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(g1, mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(g2, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = torch.nn.functional.gelu
        self.dropout = Dropout(config.transformer["dropout_rate"])
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


def get_3DReg_config(grid_size: int = 128):
    # Scale patch size so token grid remains 8x8x8 = 512 patches (16 A^3 physical patch)
    patch = max(1, grid_size // 32)
    config = _make_config_dict()
    config.patches = _make_config_dict({"size": (patch, patch, patch), "grid": (8, 8, 8)})
    config.hidden_size = 252
    config.transformer = _make_config_dict({
        "mlp_dim": 1008,
        "num_heads": 12,
        "num_layers": 12,
        "attention_dropout_rate": 0.0,
        "dropout_rate": 0.1,
    })
    config.patch_size = patch
    config.encoder_channels = (16, 32, 32)
    config.down_factor = 2
    config.n_dims = 3
    return config
