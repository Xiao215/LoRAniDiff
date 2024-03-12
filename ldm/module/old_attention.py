import torch
from torch import nn
from torch.nn import functional as F
import math

def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

class SelfAttention(nn.Module):
    def __init__(self, n_heads:int, d_embed:int, in_proj_bias:bool=True, out_proj_bias=True) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_embed = d_embed

        # d_embed must be divisible by n_heads
        assert d_embed % n_heads == 0
        self.d_head = d_embed // n_heads

        # # TODO: The paper has no normalization layer, but the code has it? I guess it's mostly good to have it for numerical stability.
        # self.norm = Normalize(d_embed)

        # This is the linear layer to project the input to the query, key and value.
        self.in_proj = nn.Linear(d_embed, d_embed*3, bias=in_proj_bias)

        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)

    def forward(self, x:torch.Tensor, causal_mask:bool=False) -> torch.Tensor:
        # x: (Batch_Size, Seq_Length, d_embed)
        h_ = x

        # Extract the size
        b, len_seq, c = h_.size()
        h_ = self.norm(h_)

        # Project the input to the query, key and value.
        qkv = self.in_proj(h_)
        # qkv: (Batch_Size, Seq_Length, d_embed * 3)

        q, k, v = qkv.chunk(3, dim=-1)
        # q, k, v: (Batch_Size, Seq_Length, d_embed)

        # Reshape the query, key and value to
        q = q.view(b, len_seq, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(b, len_seq, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(b, len_seq, self.n_heads, self.d_head).transpose(1, 2)
        # q, k, v: (Batch_Size, n_heads, Seq_Length, d_head)


        # Scaled dot product attention
        w = (torch.matmul(q, k.transpose(-2, -1))) / math.sqrt(self.head_dim)
        # w: (Batch_Size, n_heads, Seq_Length, n_heads)

        if causal_mask:
            # Mask where the upper triangle (above the principal diagonal) is 1
            mask = torch.ones_like(w, dtype=torch.bool).triu(1)
            # Fill the upper triangle with -inf
            w.masked_fill_(mask, -torch.inf)
        w /= math.sqrt(self.head_dim)
        w = F.softmax(w, dim=-1)

        # Apply attention to values
        h_ = torch.matmul(w, v)
        # (Batch_Size, n_heads, Seq_Length, d_head)

        h_.transpose(1, 2)
        # Reshape attended back to (Batch_Size, Seq_Length, d_embed)
        h_ = h_.view(b, len_seq, c)
        h_ = self.out_proj(h_)

        # TODO: The paper doesn't seem to have a residual connection, but the code has it, so I'll keep it for now.
        return x + h_

class CrossAttention(nn.Module):
    def __init__(self, n_heads, d_embed, d_cross, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.q_proj   = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj   = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj   = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x, y):
        # x (latent): # (Batch_Size, Seq_Len_Q, Dim_Q)
        # y (context): # (Batch_Size, Seq_Len_KV, Dim_KV) = (Batch_Size, 77, 768)

        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape
        # Divide each embedding of Q into multiple heads such that d_heads * n_heads = Dim_Q
        interim_shape = (batch_size, -1, self.n_heads, self.d_head)

        # (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, Dim_Q)
        q = self.q_proj(x)
        # (Batch_Size, Seq_Len_KV, Dim_KV) -> (Batch_Size, Seq_Len_KV, Dim_Q)
        k = self.k_proj(y)
        # (Batch_Size, Seq_Len_KV, Dim_KV) -> (Batch_Size, Seq_Len_KV, Dim_Q)
        v = self.v_proj(y)

        # (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, H, Dim_Q / H) -> (Batch_Size, H, Seq_Len_Q, Dim_Q / H)
        q = q.view(interim_shape).transpose(1, 2)
        # (Batch_Size, Seq_Len_KV, Dim_Q) -> (Batch_Size, Seq_Len_KV, H, Dim_Q / H) -> (Batch_Size, H, Seq_Len_KV, Dim_Q / H)
        k = k.view(interim_shape).transpose(1, 2)
        # (Batch_Size, Seq_Len_KV, Dim_Q) -> (Batch_Size, Seq_Len_KV, H, Dim_Q / H) -> (Batch_Size, H, Seq_Len_KV, Dim_Q / H)
        v = v.view(interim_shape).transpose(1, 2)

        # (Batch_Size, H, Seq_Len_Q, Dim_Q / H) @ (Batch_Size, H, Dim_Q / H, Seq_Len_KV) -> (Batch_Size, H, Seq_Len_Q, Seq_Len_KV)
        weight = q @ k.transpose(-1, -2)

        # (Batch_Size, H, Seq_Len_Q, Seq_Len_KV)
        weight /= math.sqrt(self.d_head)

        # (Batch_Size, H, Seq_Len_Q, Seq_Len_KV)
        weight = F.softmax(weight, dim=-1)

        # (Batch_Size, H, Seq_Len_Q, Seq_Len_KV) @ (Batch_Size, H, Seq_Len_KV, Dim_Q / H) -> (Batch_Size, H, Seq_Len_Q, Dim_Q / H)
        output = weight @ v

        # (Batch_Size, H, Seq_Len_Q, Dim_Q / H) -> (Batch_Size, Seq_Len_Q, H, Dim_Q / H)
        output = output.transpose(1, 2).contiguous()

        # (Batch_Size, Seq_Len_Q, H, Dim_Q / H) -> (Batch_Size, Seq_Len_Q, Dim_Q)
        output = output.view(input_shape)

        # (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, Dim_Q)
        output = self.out_proj(output)

        # (Batch_Size, Seq_Len_Q, Dim_Q)
        return output


# TODO: This seems to be another type of attention? for image? It deals with images while above deal with sequences.
class SpatialSelfAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, in_proj_bias: bool = True, out_proj_bias=True) -> None:
        super.__init__()
        self.n_heads = n_heads
        self.d_embed = d_embed

        # d_embed must be divisible by n_heads
        assert d_embed % n_heads == 0
        self.d_head = d_embed // n_heads

        self.norm = Normalize(d_embed)
        # This is the convolutional layer to project the input to the query, key and value.
        self.qkv = nn.Conv2d(d_embed, d_embed*3, kernel_size=1, stride=1, padding=0, bias=in_proj_bias)

        self.out_proj = nn.Conv2d(d_embed, d_embed, kernel_size=1, stride=1, padding=0, bias=out_proj_bias)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # x: (Batch_Size, d_embed, Height, Width)
        h_ = x

        # Extract the size
        b, c, h, w = h_.size()
        h_ = self.norm(h_)

        # Project the input to the query, key and value.
        qkv = self.qkv(h_)
        # qkv: (Batch_Size, d_embed * 3, Height, Width)

        q, k, v = qkv.chunk(3, dim=1)
        # q, k, v: (Batch_Size, d_embed, Height, Width)

        # Reshape the query, key and value to
        q = q.view(b, self.n_heads, self.head_dim, h * w)
        k = k.view(b, self.n_heads, self.head_dim, h * w)
        v = v.view(b, self.n_heads, self.head_dim, h * w)
        # q, k, v: (Batch_Size, n_heads, d_head, Height * Width)

        # Scaled dot product attention
        w = (torch.matmul(q, k.transpose(-2, -1))) / math.sqrt(self.head_dim)
        # w: (Batch_Size, n_heads, Height * Width, Height * Width)
        w = F.softmax(w, dim=-1)

        # Apply attention to values
        h_ = torch.matmul(w, v)
        # (Batch_Size, n_heads, Height * Width, d_head)

        # Reshape attended back to (Batch_Size, d_embed, Height, Width)
        h_ = h_.view(b, c, h, w)
        h_ = self.out_proj(h_)
        return x + h_
