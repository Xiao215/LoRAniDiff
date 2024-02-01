import torch
from torch import nn
from torch.nn import functional as F

class VAE_AttentionBlock(nn.Module):
    ...

class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_ch:int, out_ch:int, dropout:float) -> None:
        """This is the residual block for the Encoder.

        argument:
            in_ch (int): Number of input channels.
            out_ch (int): Number of output channels.
            dropout (float): Dropout probability.
        """

        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.dropout = torch.nn.Dropout(dropout)
