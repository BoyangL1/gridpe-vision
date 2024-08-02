from math import sqrt

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from gridPE.utils import *

# helpers


def cast_tuple(val, num):
    return val if isinstance(val, tuple) else (val,) * num


def conv_output_size(image_size, kernel_size, stride, padding=0):
    return int(((image_size - kernel_size + (2 * padding)) / stride) + 1)


# depthwise convolution, for pooling


class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride, bias=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                dim_in,
                dim_out,
                kernel_size=kernel_size,
                padding=padding,
                groups=dim_in,
                stride=stride,
                bias=bias,
            ),
            nn.Conv2d(dim_out, dim_out, kernel_size=1, bias=bias),
        )

    def forward(self, x):
        return self.net(x)


# pooling layer


class Pool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.downsample = DepthWiseConv2d(
            dim, dim * 2, kernel_size=3, stride=2, padding=1
        )
        self.cls_ff = nn.Linear(dim, dim * 2)

    def forward(self, x):
        cls_token, tokens = x[:, :1], x[:, 1:]

        cls_token = self.cls_ff(cls_token)

        tokens = rearrange(tokens, "b (h w) c -> b c h w", h=int(sqrt(tokens.shape[1])))
        tokens = self.downsample(tokens)
        tokens = rearrange(tokens, "b c h w -> b (h w) c")

        return torch.cat((cls_token, tokens), dim=1)


# main class
class PiT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        channels=3
    ):
        super().__init__()
        assert (
            image_size % patch_size == 0
        ), "Image dimensions must be divisible by the patch size."
        assert isinstance(
            depth, tuple
        ), "depth must be a tuple of integers, specifying the number of blocks before each downsizing"
        heads = cast_tuple(heads, len(depth))

        patch_dim = channels * patch_size**2

        self.to_patch_embedding = nn.Sequential(
            nn.Unfold(kernel_size=patch_size, stride=patch_size // 2),
            Rearrange("b c n -> b n c"),
            nn.Linear(patch_dim, dim),
        )

        output_size = conv_output_size(image_size, patch_size, patch_size // 2)
        num_patches = output_size**2

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        layers = []

        for ind, (layer_depth, layer_heads) in enumerate(zip(depth, heads)):
            not_last = ind < (len(depth) - 1)

            layers.append(
                Transformer(dim, layer_depth, layer_heads, dim_head, mlp_dim, dropout)
            )

            if not_last:
                layers.append(Pool(dim))
                dim *= 2

        self.layers = nn.Sequential(*layers)

        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        _, _, H, W = img.shape

        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :n+1]
        x = self.dropout(x)

        for layer in self.layers:
            if isinstance(layer, Transformer):
                x = layer(x, H, W)
            else:
                x = layer(x)

        return self.mlp_head(x[:, 0])

class PiTRotate(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        channels=3
    ):
        super().__init__()
        assert (
            image_size % patch_size == 0
        ), "Image dimensions must be divisible by the patch size."
        assert isinstance(
            depth, tuple
        ), "depth must be a tuple of integers, specifying the number of blocks before each downsizing"
        heads = cast_tuple(heads, len(depth))

        patch_dim = channels * patch_size**2

        self.to_patch_embedding = nn.Sequential(
            nn.Unfold(kernel_size=patch_size, stride=patch_size // 2),
            Rearrange("b c n -> b n c"),
            nn.Linear(patch_dim, dim),
        )

        output_size = conv_output_size(image_size, patch_size, patch_size // 2)
        num_patches = output_size**2

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        layers = []

        for ind, (layer_depth, layer_heads) in enumerate(zip(depth, heads)):
            not_last = ind < (len(depth) - 1)

            layers.append(
                TransformerRotate(dim, layer_depth, layer_heads, dim_head, mlp_dim, dropout)
            )

            if not_last:
                layers.append(Pool(dim))
                dim *= 2

        self.layers = nn.Sequential(*layers)

        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        _, _, H, W = img.shape

        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.dropout(x)

        for layer in self.layers:
            if isinstance(layer, TransformerRotate):
                x = layer(x, H, W)
            else:
                x = layer(x)

        return self.mlp_head(x[:, 0])
    
class PiTComplex(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        channels=3
    ):
        super().__init__()
        assert (
            image_size % patch_size == 0
        ), "Image dimensions must be divisible by the patch size."
        assert isinstance(
            depth, tuple
        ), "depth must be a tuple of integers, specifying the number of blocks before each downsizing"
        heads = cast_tuple(heads, len(depth))

        patch_dim = channels * patch_size**2

        self.to_patch_embedding = nn.Sequential(
            nn.Unfold(kernel_size=patch_size, stride=patch_size // 2),
            Rearrange("b c n -> b n c"),
            nn.Linear(patch_dim, dim),
        )

        output_size = conv_output_size(image_size, patch_size, patch_size // 2)
        num_patches = output_size**2

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        layers = []

        for ind, (layer_depth, layer_heads) in enumerate(zip(depth, heads)):
            not_last = ind < (len(depth) - 1)

            layers.append(
                TransformerComplex(dim, layer_depth, layer_heads, dim_head, mlp_dim, dropout)
            )

            if not_last:
                layers.append(Pool(dim))
                dim *= 2

        self.layers = nn.Sequential(*layers)

        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        _, _, H, W = img.shape

        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.dropout(x)

        for layer in self.layers:
            if isinstance(layer, TransformerComplex):
                x = layer(x, H, W)
            else:
                x = layer(x)

        return self.mlp_head(x[:, 0])

class PiTMerge(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        channels=3
    ):
        super().__init__()
        assert (
            image_size % patch_size == 0
        ), "Image dimensions must be divisible by the patch size."
        assert isinstance(
            depth, tuple
        ), "depth must be a tuple of integers, specifying the number of blocks before each downsizing"
        heads = cast_tuple(heads, len(depth))

        patch_dim = channels * patch_size**2

        self.to_patch_embedding = nn.Sequential(
            nn.Unfold(kernel_size=patch_size, stride=patch_size // 2),
            Rearrange("b c n -> b n c"),
            nn.Linear(patch_dim, dim),
        )

        output_size = conv_output_size(image_size, patch_size, patch_size // 2)
        num_patches = output_size**2

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        layers = []

        for ind, (layer_depth, layer_heads) in enumerate(zip(depth, heads)):
            not_last = ind < (len(depth) - 1)

            layers.append(
                TransformerMerge(dim, layer_depth, layer_heads, dim_head, mlp_dim, dropout)
            )

            if not_last:
                layers.append(Pool(dim))
                dim *= 2

        self.layers = nn.Sequential(*layers)

        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        _, _, H, W = img.shape

        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.dropout(x)

        for layer in self.layers:
            if isinstance(layer, TransformerMerge):
                x = layer(x, H, W)
            else:
                x = layer(x)

        return self.mlp_head(x[:, 0])

class PiTDeep(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        channels=3
    ):
        super().__init__()
        assert (
            image_size % patch_size == 0
        ), "Image dimensions must be divisible by the patch size."
        assert isinstance(
            depth, tuple
        ), "depth must be a tuple of integers, specifying the number of blocks before each downsizing"
        heads = cast_tuple(heads, len(depth))

        patch_dim = channels * patch_size**2

        self.to_patch_embedding = nn.Sequential(
            nn.Unfold(kernel_size=patch_size, stride=patch_size // 2),
            Rearrange("b c n -> b n c"),
            nn.Linear(patch_dim, dim),
        )

        output_size = conv_output_size(image_size, patch_size, patch_size // 2)
        num_patches = output_size**2

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        layers = []

        for ind, (layer_depth, layer_heads) in enumerate(zip(depth, heads)):
            not_last = ind < (len(depth) - 1)

            layers.append(
                TransformerDeep(dim, layer_depth, layer_heads, dim_head, mlp_dim, dropout)
            )

            if not_last:
                layers.append(Pool(dim))
                dim *= 2

        self.layers = nn.Sequential(*layers)

        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        _, _, H, W = img.shape

        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.dropout(x)

        for layer in self.layers:
            if isinstance(layer, TransformerDeep):
                x = layer(x, H, W)
            else:
                x = layer(x)

        return self.mlp_head(x[:, 0])