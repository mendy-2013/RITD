import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from einops import rearrange


class LayerNorm2d(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()


class SPInterAttModule(nn.Module):
    def __init__(self, dim, num_heads, qk_dim, qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0.):
        super().__init__()
        self.dim = dim
        self.qk_dim = qk_dim
        self.num_heads = num_heads

        self.q = nn.Conv2d(dim, qk_dim, 1, bias=qkv_bias)
        self.k = nn.Conv2d(dim, qk_dim, 1, bias=qkv_bias)
        self.v = nn.Conv2d(dim, dim, 1, bias=qkv_bias)
        self.sp = nn.Conv2d(dim, qk_dim, 1,bias=qkv_bias)

        self.norm = LayerNorm2d(dim)

        head_dim = self.qk_dim // self.num_heads
        self.attn_drop = nn.Dropout(attn_drop)
        self.scale = qk_scale or head_dim ** -0.5

    # def forward_stoken(self, x, affinity_matrix):
    #     x = rearrange(x, 'b c h w -> b (h w) c')
    #     stokens = torch.bmm(affinity_matrix, x) / (affinity_matrix.sum(2, keepdim=True) + 1e-16) # b, n, c
    #     return stokens
    def forward_stoken(self, x, affinity_matrix):
        x = rearrange(x, 'b c h w -> b (h w) c')
        stokens = torch.bmm(affinity_matrix, x) / (affinity_matrix.sum(2, keepdim=True) + 1e-16) # b, n, c
        return stokens

    def forward(self, x, mask):
        """
        Input: x: (B, H*W, C), x_size: (H, W)
        Output: x: (B, H*W, C)
        """
        B, C, H, W = x.shape

        x = self.norm(x)


        # generate superpixel stoken
        mask = self.forward_stoken(x, mask) # b, k, c
        # stoken projection
        # stoken = self.sp(stoken).permute(0,2,1).reshape(B, self.num_heads, self.qk_dim // self.num_heads, num_spixels) # B, H, C, hh*ww
        # mask = self.sp(mask).reshape(B, self.num_heads, self.qk_dim // self.num_heads, H * W)
        # q, k, v projection
        q = self.q(x).reshape(B, self.num_heads, self.qk_dim // self.num_heads, H*W) # B, H, C, H*W
        # print(q.shape)
        k = self.k(x).reshape(B, self.num_heads, self.qk_dim // self.num_heads, H*W) # B, H, C, H*W
        v = self.v(x).reshape(B, self.num_heads, self.dim // self.num_heads, H*W) # B, H, C, N

        # stoken interaction
        # s_attn = F.normalize(k, dim=-2).transpose(-2, -1) @ F.normalize(stoken, dim=-2) # B, H, H*W, hh*ww
        s_attn = k.transpose(-2, -1) @ mask * self.scale # B, H, H*W, hh*ww
        s_attn = self.attn_drop(F.softmax(s_attn, -2))
        s_out = (v @ s_attn) # B, H, C, hh*ww

        # x_attn = F.normalize(stoken, dim=-2).transpose(-2, -1) @ F.normalize(q, dim=-2) # B, H, hh*ww, H*W
        x_attn = mask.transpose(-2, -1) @ q * self.scale
        x_attn = self.attn_drop(F.softmax(x_attn, -2))
        x_out = (s_out @ x_attn).reshape(B, C, H, W) # B, H, C, H*W

        return x_out


class FFN(nn.Module):
    """Feed Forward Network.
    Args:
        dim (int): Base channels.
        hidden_dim (int): Channels of hidden mlp.
    """

    def __init__(self, dim, hidden_dim, out_dim, norm_layer=LayerNorm2d):
        super().__init__()
        self.norm = norm_layer(dim)

        # Channel Attention
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim//8, 1, bias=True),
            nn.GELU(),
            # nn.ReLU(True),
            nn.Conv2d(dim // 8, dim, 1, bias=True),
            nn.Sigmoid()
        )

        # Pixel Attention
        self.pa = nn.Sequential(
            nn.Conv2d(dim, dim // 8, 1, bias=True),
            nn.GELU(),
            # nn.ReLU(True),
            nn.Conv2d(dim // 8, 1, 1, bias=True),
            nn.Sigmoid()
        )

        self.fc1 = nn.Conv2d(dim * 2, hidden_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_dim, out_dim, 1)

    def forward(self, x):
        x = self.norm(x)
        x = torch.cat([self.ca(x) * x, self.pa(x) * x], dim=1)
        return self.fc2(self.act(self.fc1(x)))
