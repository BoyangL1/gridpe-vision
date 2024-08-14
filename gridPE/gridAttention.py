import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

# Grid Positional Encoding
from gridPE.gridPE import *

class Attention(nn.Module):
    """
    GSA: using a  key to summarize the information for a group to be efficient.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        # 在这里加grid编码
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
    
class GridRotateAttention(Attention):
    """
    GridRotateAttention is a class that performs attention mechanism with grid rotation remapping.
    """
    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_)
            H_, W_ = x_.shape[-2], x_.shape[-1]
            x_ = x_.reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        self.head_dim = C // self.num_heads
        if N != H * W:
            H += 1
            q_pos_list = [(i, j) for i in range(H) for j in range(W)]
            q_pos_list = q_pos_list[:N]
        else:
            q_pos_list = [(i, j) for i in range(H) for j in range(W)]
        
        grid_pe_for_q = GridRotatePositionalEncoding(q_pos_list, self.head_dim)
        if self.sr_ratio > 1:
            sample_pos_i = self.sr_ratio * np.arange(H_, dtype=int) + self.sr_ratio // 2
            sample_pos_j = self.sr_ratio * np.arange(W_, dtype=int) + self.sr_ratio // 2
            k_pos_list = [(i, j) for i in sample_pos_i for j in sample_pos_j]
        else:
            k_pos_list = q_pos_list
        grid_pe_for_k = GridRotatePositionalEncoding(k_pos_list, self.head_dim)
        
        q = grid_pe_for_q.apply_encoding(
            q
        )  
        k = grid_pe_for_k.apply_encoding(
            k
        )  

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class GridMergingAttention(Attention):
    def forward(self, x, H, W):
        # 创建位置编码
        B, N, C = x.shape  
        if N != H * W:
            H += 1
            pos_list = [(i, j) for i in range(H) for j in range(W)]
            pos_list = pos_list[:N]
        else:
            pos_list = [(i, j) for i in range(H) for j in range(W)]
        
        grid_pe = GridMergingPositionalEncoding(pos_list, C)
        grid_pe_tensor = torch.from_numpy(grid_pe.grid_pe).float()
        grid_pe_tensor = grid_pe_tensor.transpose(0,1)
        grid_pe_expanded = grid_pe_tensor.unsqueeze(0)

        x += grid_pe_expanded.to(x.device) 

        x = super(GridMergingAttention, self).forward(x, H, W)
        return x


class GridComplexAttention(Attention):
    """
    GridSplitAttention is a class that performs attention mechanism with grid remapping.
    """
    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_)
            H_, W_ = x_.shape[-2], x_.shape[-1]
            x_ = x_.reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        self.head_dim = C // self.num_heads
        if N != H * W:
            H += 1
            q_pos_list = [(i, j) for i in range(H) for j in range(W)]
            q_pos_list = q_pos_list[:N]
        else:
            q_pos_list = [(i, j) for i in range(H) for j in range(W)]

        grid_pe_for_q = GridComplexPositionalEncoding(q_pos_list, self.head_dim)
        if self.sr_ratio > 1:
            sample_pos_i = self.sr_ratio * np.arange(H_, dtype=int) + self.sr_ratio // 2
            sample_pos_j = self.sr_ratio * np.arange(W_, dtype=int) + self.sr_ratio // 2
            k_pos_list = [(i, j) for i in sample_pos_i for j in sample_pos_j]
        else:
            k_pos_list = q_pos_list
        grid_pe_for_k = GridComplexPositionalEncoding(k_pos_list, self.head_dim)


        q = grid_pe_for_q.apply_encoding(
            q
        )  
        k = grid_pe_for_k.apply_encoding(
            k
        )  

        # attn = (q @ k.transpose(-2, -1)) * self.scale
        q_real = q.real
        q_imag = q.imag
        k_real = k.real.transpose(-2, -1)
        k_imag = k.imag.transpose(-2, -1)
        real_part = q_real @ k_real - q_imag @ k_imag
        # imag_part = q_real @ k_imag + q_imag @ k_real
        attn = real_part * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # if attn.dtype != v.dtype:
        #     vtype = v.dtype
        #     v = v.to(attn.dtype)
        #     x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        #     x = x.to(vtype)
        # else:
        #     x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        # fast version
        attn = attn.to(v.dtype)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class GridDeepAttention(Attention):
    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_)
            H_, W_ = x_.shape[-2], x_.shape[-1]
            x_ = x_.reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        self.head_dim = C // self.num_heads
        if N != H * W:
            H += 1
            q_pos_list = [(i, j) for i in range(H) for j in range(W)]
            q_pos_list = q_pos_list[:N]
        else:
            q_pos_list = [(i, j) for i in range(H) for j in range(W)]
        
        grid_pe_for_q = GridDeepPositionalEncoding(q_pos_list, self.head_dim)
        grid_pe_for_q = self._grid_complex_to_real(grid_pe_for_q)

        if self.sr_ratio > 1:
            sample_pos_i = self.sr_ratio * np.arange(H_, dtype=int) + self.sr_ratio // 2
            sample_pos_j = self.sr_ratio * np.arange(W_, dtype=int) + self.sr_ratio // 2
            k_pos_list = [(i, j) for i in sample_pos_i for j in sample_pos_j]
        else:
            k_pos_list = q_pos_list
        grid_pe_for_k = GridDeepPositionalEncoding(k_pos_list, self.head_dim)
        grid_pe_for_k = self._grid_complex_to_real(grid_pe_for_k)


        q = grid_pe_for_q.apply_encoding(
            q
        )  
        k = grid_pe_for_k.apply_encoding(
            k
        )  

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
    
    def _grid_complex_to_real(self, grid_pe):
        model = ComplexToReal(grid_pe.grid_pe.shape[1], grid_pe.grid_pe.shape[1])
        if not isinstance(grid_pe.grid_pe, torch.Tensor):
            grid_pe.grid_pe = torch.view_as_complex(torch.from_numpy(np.stack((grid_pe.grid_pe.real, grid_pe.grid_pe.imag), axis=-1)))
        grid_pe.grid_pe = model(grid_pe.grid_pe)
        
        return grid_pe