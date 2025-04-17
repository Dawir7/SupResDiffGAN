import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn import init

# source: https://github.com/final-0/Real-ESRGAN-bicubic/blob/main/esrgan_Gnet.py


def pixel_unshuffle(x: torch.Tensor, scale: int) -> torch.Tensor:
    """Pixel unshuffle operation.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (B, C, H, W).
    scale : int
        Downscaling factor.

    Returns
    -------
    torch.Tensor
        Unshuffled tensor.
    """
    b, c, hh, hw = x.size()
    out_channel = c * (scale**2)
    assert hh % scale == 0 and hw % scale == 0
    h = hh // scale
    w = hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)


def make_layer(basic_block: nn.Module, num_basic_block: int, **kwarg) -> nn.Sequential:
    """Create a sequential layer composed of multiple basic blocks.

    Parameters
    ----------
    basic_block : nn.Module
        Basic block class.
    num_basic_block : int
        Number of basic blocks.

    Returns
    -------
    nn.Sequential
        Sequential layer.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


def default_init_weights(
    module_list: list[nn.Module], scale: float = 1, bias_fill: float = 0, **kwargs
) -> None:
    """Initialize weights of modules.

    Parameters
    ----------
    module_list : list[nn.Module]
        List of modules to initialize.
    scale : float, optional
        Scaling factor for weights, by default 1.
    bias_fill : float, optional
        Value to fill biases, by default 0.
    """
    for module in module_list:
        for m in module.modules():
            init.kaiming_normal_(m.weight, **kwargs)
            m.weight.data *= scale
            if m.bias is not None:
                m.bias.data.fill_(bias_fill)


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block.

    Parameters
    ----------
    num_feat : int, optional
        Number of feature maps, by default 64.
    num_grow_ch : int, optional
        Number of growth channels, by default 32.
    """

    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32) -> None:
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        default_init_weights(
            [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block.

    Parameters
    ----------
    num_feat : int
        Number of feature maps.
    num_grow_ch : int, optional
        Number of growth channels, by default 32.
    """

    def __init__(self, num_feat: int, num_grow_ch: int = 32) -> None:
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class Generator(nn.Module):
    """Generator network for Real-ESRGAN.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    scale_factor : int, optional
        Upscaling factor, by default 4.
    num_feat : int, optional
        Number of feature maps, by default 64.
    num_resblocks : int, optional
        Number of residual blocks, by default 23.
    num_grow_ch : int, optional
        Number of growth channels, by default 32.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: int = 4,
        num_feat: int = 64,
        num_resblocks: int = 23,
        num_grow_ch: int = 32,
    ) -> None:
        super(Generator, self).__init__()
        self.scale_factor = scale_factor
        if scale_factor == 2:
            in_channels = in_channels * 4
        elif scale_factor == 1:
            in_channels = in_channels * 16
        self.conv_first = nn.Conv2d(in_channels, num_feat, 3, 1, 1)
        self.body = make_layer(
            RRDB, num_resblocks, num_feat=num_feat, num_grow_ch=num_grow_ch
        )
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # upsample
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, out_channels, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor in the range [-1, 1].
        """
        if self.scale_factor == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.scale_factor == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x
        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        # upsample
        feat = self.lrelu(
            self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest"))
        )
        feat = self.lrelu(
            self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest"))
        )
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return self.tanh(out)
