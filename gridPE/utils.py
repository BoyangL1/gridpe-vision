from gridPE.gridAttention import *

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(
                            dim,
                            num_heads=heads,
                            qkv_bias=False,
                            qk_scale=None,
                            attn_drop=dropout,
                            proj_drop=dropout,
                            sr_ratio=1,
                        ),
                        FeedForward(dim, mlp_dim, dropout=dropout),
                    ]
                )
            )

    def forward(self, x, H, W):
        for attn, ff in self.layers:
            x = attn(x,H,W) + x
            x = ff(x) + x

        return self.norm(x)
    
class TransformerRotate(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        GridRotateAttention(
                            dim,
                            num_heads=heads,
                            qkv_bias=False,
                            qk_scale=None,
                            attn_drop=dropout,
                            proj_drop=dropout,
                            sr_ratio=1,
                        ),
                        FeedForward(dim, mlp_dim, dropout=dropout),
                    ]
                )
            )

    def forward(self, x, H, W):
        for attn, ff in self.layers:
            x = attn(x,H,W) + x
            x = ff(x) + x

        return self.norm(x)
    
class TransformerMerge(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        GridMergingAttention(
                            dim,
                            num_heads=heads,
                            qkv_bias=False,
                            qk_scale=None,
                            attn_drop=dropout,
                            proj_drop=dropout,
                            sr_ratio=1,
                        ),
                        FeedForward(dim, mlp_dim, dropout=dropout),
                    ]
                )
            )

    def forward(self, x, H, W):
        for attn, ff in self.layers:
            x = attn(x,H,W) + x
            x = ff(x) + x

        return self.norm(x)

class TransformerComplex(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        GridComplexAttention(
                            dim,
                            num_heads=heads,
                            qkv_bias=False,
                            qk_scale=None,
                            attn_drop=dropout,
                            proj_drop=dropout,
                            sr_ratio=1,
                        ),
                        FeedForward(dim, mlp_dim, dropout=dropout),
                    ]
                )
            )

    def forward(self, x, H, W):
        for attn, ff in self.layers:
            x = attn(x,H,W) + x
            x = ff(x) + x

        return self.norm(x)

class TransformerDeep(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        GridDeepAttention(
                            dim,
                            num_heads=heads,
                            qkv_bias=False,
                            qk_scale=None,
                            attn_drop=dropout,
                            proj_drop=dropout,
                            sr_ratio=1,
                        ),
                        FeedForward(dim, mlp_dim, dropout=dropout),
                    ]
                )
            )

    def forward(self, x, H, W):
        for attn, ff in self.layers:
            x = attn(x,H,W) + x
            x = ff(x) + x

        return self.norm(x)