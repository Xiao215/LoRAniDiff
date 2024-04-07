import torch
from torch import nn
from torch.nn import functional as F
from ldm.model.decoder import VAE_AttentionBlock, VAE_ResidualBlock


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()

        self.conv = nn.ConvTranspose2d(
            in_channels, in_channels, kernel_size=3, stride=2, padding=1
        )

    def forward(self, x):
        return self.conv(x)


class VAE_Decoder(nn.Module):
    def __init__(
        self,
        z_channels: int = 4,
        ch: int = 128,
        ch_mult: tuple[int] = (8, 4, 2, 1),
        num_res_blocks: int = 2,
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        self.z_channels = z_channels
        self.ch = ch
        self.ch_mult = ch_mult
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels

        self.conv_in = nn.Conv2d(
            z_channels, ch * ch_mult[0], kernel_size=3, stride=1, padding=1
        )
        self.up = nn.ModuleList()

        for level_idx in range(len(ch_mult) - 1):
            up = nn.Module()
            block_in = ch * ch_mult[level_idx]
            block_out = ch * ch_mult[level_idx + 1]
            up.upsample = UpsampleBlock(block_in)
            residualBlock = nn.ModuleList()
            for _ in range(num_res_blocks):
                residualBlock.append(VAE_ResidualBlock(block_in, block_out))
                block_in = block_out
            up.residualBlock = residualBlock
            self.up.append(up)

        self.mid = nn.Module()
        self.mid.attention = VAE_AttentionBlock(ch)

        self.norm = nn.GroupNorm(32, ch)
        self.activation = nn.SiLU()
        self.conv_out = nn.Conv2d(
            ch, self.in_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x /= 0.18215
        x = self.conv_in(x)
        for up in self.up:
            x = up(x)
        x = self.mid.attention(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.conv_out(x)
        return x
