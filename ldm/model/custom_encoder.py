import torch
from torch import nn
from torch.nn import functional as F
from ldm.model.decoder import VAE_AttentionBlock, VAE_ResidualBlock


def Normalize(in_channels):
    return torch.nn.GroupNorm(
        num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
    )


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels: int, stride: int = 2) -> None:
        """This is the downsample block for the Encoder. It will downsample the input by a factor of 2. (W/=2, H/=2)

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            num_res_blocks (int): Number of residual blocks in the downsample block.
            use_attention (bool): Whether to use attention.
        Return:
        """

        super().__init__()
        self.in_channels = in_channels
        # This is the CNN layer to downsample the input.
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=2, padding=0
        )

    def forward(self, x):
        # Since the stide is 2, a padding will be added to the input.
        pad = (0, 1, 0, 1)
        x = F.pad(x, pad, mode="constant", value=0)


class VAE_Encoder(nn.Module):
    def __init__(
        self,
        ch: int = 128,
        ch_mult: tuple[int] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        dropout: float = 0.0,
        in_channels: int = 3,
        z_channels: int = 4,
    ) -> None:
        """This is the encoder class for the LDM model.

        Args:
            ch (int): Number of channels in the first layer.
            ch_mult (tuple[int]): Multiplier for the number of channels in each layer.
            num_res_blocks (int): Number of residual blocks in each layer/level.
            dropout (float): Dropout probability.
            in_channels (int): Number of input channels.
            z_channels (int): Number of latent channels.

        Return:
        """

        super().__init__()
        self.in_channels = in_channels
        self.z_channels = z_channels
        self.ch = ch
        self.ch_mult = ch_mult
        self.num_res_blocks = num_res_blocks
        self.dropout = dropout

        # downsampling
        self.conv_in = torch.nn.Conv2d(
            self.in_channels, self.ch, kernel_size=3, stride=1, padding=1
        )

        # in channel multiplier
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()

        block_in = None
        block_out = None
        for level_idx in range(len(ch_mult)):
            down = nn.Module()
            block_in = ch * in_ch_mult[level_idx]
            block_out = ch * ch_mult[level_idx]

            # Add a downsample block to the list.
            down.downsample = DownsampleBlock(block_in, stride=2)
            residualBlock = nn.ModuleList()
            for _ in range(num_res_blocks):
                residualBlock.append(
                    VAE_ResidualBlock(
                        in_ch=block_in,
                        out_ch=block_out,
                        dropout=dropout))
                block_in = block_out
            down.residualBlock = residualBlock

        # middle
        self.mid = nn.Module()
        self.mid.residual_1 = VAE_ResidualBlock(
            in_ch=block_in, out_ch=block_in, dropout=dropout
        )
        self.mid.attention = VAE_AttentionBlock(channels=block_in)
        self.mid.residual_2 = VAE_ResidualBlock(
            in_ch=block_in, out_ch=block_in, dropout=dropout
        )

        # end
        self.norm = Normalize(block_in)

        # We could also try sigmoid below?
        self.activation = torch.nn.SiLU()
        self.conv_out = torch.nn.Conv2d(
            block_in, 2 * self.z_channels, kernel_size=3, stride=1, padding=1
        )
        self.quant_cov = torch.nn.Conv2d(
            2 * self.z_channels, 2 * self.z_channels, kernel_size=1
        )

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """This is the forward function for the encoder.

        argument:
            x (torch.Tensor): Input image tensor, (Batch_Size, Channel, Height, Width).
            noise (torch.Tensor): Noise tensor in the latent space, (Batch_Size, z-channels, Compressed_Height, Compressed__Width).
        return:
            x (torch.Tensor): Output distribution.
        """

        # downsampling
        x = self.conv_in(x)
        for down in self.down:
            x = down(x)
        # middle
        x = self.mid.residual_1(x)
        x = self.mid.attention(x)
        x = self.mid.residual_2(x)
        # end
        x = self.norm(x)
        x = self.activation(x)
        x = self.conv_out(x)
        x = self.quant_cov(x)

        # Extract the mean and log variance from the output.
        mean, log_var = x.chunk(2, dim=1)
        # Bound the log variance to prevent numerical instability.
        log_var = torch.clamp(log_var, -30, 20)
        var = log_var.exp()
        stdev = var.sqrt()

        # Compute the modified Gaussian distribution.
        x = mean + stdev * noise

        # scale x by a constant
        # constant taken from:
        # https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/configs/stable-diffusion/v1-inference.yaml#L17C1-L17C1
        x *= 0.18215
        return x
