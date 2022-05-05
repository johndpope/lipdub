import torch

from custom_types import *


class Mlp(nn.Module):
    def __init__(self, in_dim, h_dim, out_d: Optional[int] = None, act=nnf.relu, dropout=0.):
        super().__init__()
        out_d = out_d if out_d is not None else in_dim
        self.fc1 = nn.Linear(in_dim, h_dim)
        self.act = act
        self.fc2 = nn.Linear(h_dim, out_d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class MultiHeadAttention(nn.Module):

    def __init__(self, dim_self, dim_ref, num_heads, bias=True, dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_self // num_heads
        self.scale = head_dim ** -0.5
        self.to_queries = nn.Linear(dim_self, dim_self, bias=bias)
        self.to_keys_values = nn.Linear(dim_ref, dim_self * 2, bias=bias)
        self.project = nn.Linear(dim_self, dim_self)
        self.dropout = nn.Dropout(dropout)

    def forward_interpolation(self, queries: T, keys: T, values: T, alpha: T, mask: TN = None) -> T:
        attention = torch.einsum('nhd,bmhd->bnmh', queries[0], keys) * self.scale
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            attention = attention.masked_fill(mask.unsqueeze(3), float("-inf"))
        attention = attention.softmax(dim=2)
        attention = attention * alpha[:, None, None, None]
        out = torch.einsum('bnmh,bmhd->nhd', attention, values).reshape(1, attention.shape[1], -1)
        return out, attention

    def forward(self, x, y: Optional[T] = None, mask: Optional[T] = None, alpha: TN = None):
        y = y if y is not None else x
        b_a, n, c = x.shape
        b, m, d = y.shape
        # b n h dh
        queries = self.to_queries(x).reshape(b_a, n, self.num_heads, c // self.num_heads)
        # b m 2 h dh
        keys_values = self.to_keys_values(y).reshape(b, m, 2, self.num_heads, c // self.num_heads)
        keys, values = keys_values[:, :, 0], keys_values[:, :, 1]
        if alpha is not None:
            out, attention = self.forward_interpolation(queries, keys, values, alpha, mask)
        else:
            attention = torch.einsum('bnhd,bmhd->bnmh', queries, keys) * self.scale
            if mask is not None:
                if mask.dim() == 2:
                    mask = mask.unsqueeze(1)
                attention = attention.masked_fill(mask.unsqueeze(3), float("-inf"))
            attention = attention.softmax(dim=2)
            out = torch.einsum('bnmh,bmhd->bnhd', attention, values).reshape(b, n, c)
        out = self.project(out)
        return out, attention


class TransformerLayer(nn.Module):

    def forward_with_attention(self, x, y: Optional[T] = None, mask: Optional[T] = None, alpha: TN = None):
        x_, attention = self.attn(self.norm1(x), y, mask, alpha)
        x = x + x_
        x = x + self.mlp(self.norm2(x))
        return x, attention

    def forward(self, x, y: Optional[T] = None, mask: Optional[T] = None, alpha: TN = None):
        x = x + self.attn(self.norm1(x), y, mask, alpha)[0]
        x = x + self.mlp(self.norm2(x))
        return x

    def __init__(self, dim_self, dim_ref, num_heads, mlp_ratio=4., bias=False, dropout=0., act=nnf.relu,
                 norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim_self)
        self.attn = MultiHeadAttention(dim_self, dim_ref, num_heads, bias=bias, dropout=dropout)
        self.norm2 = norm_layer(dim_self)
        self.mlp = Mlp(dim_self, int(dim_self * mlp_ratio), act=act, dropout=dropout)


class CrossTransformerLayer(TransformerLayer):

    def forward(self, x, y: Optional[T] = None, mask: Optional[T] = None, alpha: TN = None):
        x = x + self.attn0(self.norm0(x), mask)[0]
        return super(CrossTransformerLayer, self).forward(x, y)

    def __init__(self, dim_self, dim_ref, num_heads, mlp_ratio=4., bias=False, dropout=0., act=nnf.relu,
                 norm_layer: nn.Module = nn.LayerNorm):
        super(CrossTransformerLayer, self).__init__(dim_self, dim_ref, num_heads, mlp_ratio, bias,
                                                    dropout, act, norm_layer)
        self.norm0 = norm_layer(dim_self)
        self.attn0 = MultiHeadAttention(dim_self, dim_self, num_heads, bias=bias, dropout=dropout)


class DummyTransformer:

    @staticmethod
    def forward_with_attention(x, *args, **kwargs):
        return x, []

    @staticmethod
    def forward(x, *args, **kwargs):
        return x


class Transformer(nn.Module):

    def forward_with_attention(self, x, y: Optional[T] = None, mask: Optional[T] = None, alpha: TN = None):
        attentions = []
        for layer in self.layers:
            x, att = layer.forward_with_attention(x, y, mask, alpha)
            attentions.append(att)
        return x, attentions

    def forward(self, x, y: TN = None, mask: TN = None, alpha: TN = None):
        for layer in self.layers:
            x = layer(x, y, mask, alpha)
        return x

    def __init__(self, dim_self: int, num_heads: int, num_layers: int, dim_ref: Optional[int] = None,
                 mlp_ratio: float = 2., act=nnf.relu, norm_layer: nn.Module = nn.LayerNorm):
        super(Transformer, self).__init__()
        dim_ref = dim_ref if dim_ref is not None else dim_self
        self.layers = nn.ModuleList([TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act,
                                                      norm_layer=norm_layer) for _ in range(num_layers)])


class CombTransformer(nn.Module):

    def forward(self, x, y: TN = None, mask: TN = None):
        for layer in self.layers:
            x = layer(x, y, mask)
        return x

    def __init__(self, dim_self: int, num_heads: int, num_layers: int, dim_ref: int,
                 mlp_ratio: float = 2., act=nnf.relu, norm_layer: nn.Module = nn.LayerNorm):
        super(CombTransformer, self).__init__()
        self.layers = nn.ModuleList([CrossTransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act,
                                                           norm_layer=norm_layer) for _ in range(num_layers)])


class VisionTransformer(nn.Module):

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        if self.extra_token:
            x = torch.cat((torch.zeros(x.shape[0], 1, x.shape[2], device=x.device), x), dim=1)
        x = x + self.positional_embedding
        x = self.ln_pre(x)
        x = self.transformer(x)
        return x

    def __init__(self, input_resolution: int, patch_size: int, hidden_dim: int, layers: int, heads: int,
                 input_dim: int, extra_token=False):
        super().__init__()
        self.input_resolution = input_resolution
        self.conv1 = nn.Conv2d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=(patch_size, patch_size),
                               stride=(patch_size, patch_size), bias=False)

        scale = hidden_dim ** -0.5
        self.extra_token = extra_token
        self.positional_embedding = nn.Parameter(torch.randn((input_resolution // patch_size) ** 2 + int(extra_token), hidden_dim) * scale)
        self.ln_pre = nn.LayerNorm(hidden_dim)
        self.transformer = Transformer(hidden_dim, heads, layers)


if __name__ == '__main__':
    model = VisionTransformer(128, 16, 512, 6, 8, 3, True)
    x = torch.rand(5, 3, 128, 128)
    out = model(x)