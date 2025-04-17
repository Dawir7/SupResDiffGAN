import torch
import torch.nn as nn


# Generator
class Generator(nn.Module):
    """Generator network for Super-Resolution GAN (SRGAN).

    This network takes a low-resolution image as input and generates
    a high-resolution image.

    Parameters
    ----------
    in_channels : int
        Number of input channels for the low-resolution image.
    out_channels : int
        Number of output channels for the high-resolution image.
    scale_factor : int
        Factor by which to upscale the spatial dimensions of the input image.
    num_resblocks : int
        Number of residual blocks in the network.
    """

    def __init__(
        self, in_channels: int, out_channels: int, scale_factor: int, num_resblocks: int
    ) -> None:
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=9, stride=1, padding=4)
        self.relu = nn.ReLU()

        res_blocks = []
        for _ in range(num_resblocks):
            res_blocks.append(ResBlock([64, 64]))

        self.res_blocks = nn.Sequential(*res_blocks)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        upsample_blocks = []
        for _ in range(scale_factor // 2):
            upsample_blocks.append(Upsample(64, 64, 2))
        self.upsample = nn.Sequential(*upsample_blocks)

        self.conv3 = nn.Conv2d(64, out_channels, kernel_size=9, stride=1, padding=4)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Generator network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (batch_size, in_channels, height, width).

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        x = self.relu(self.conv1(x))

        residual = x
        x = self.res_blocks(x)
        x = self.bn2(self.conv2(x))
        x = x + residual

        x = self.upsample(x)

        x = self.tanh(self.conv3(x))
        return x


# Upsample Block
class Upsample(nn.Module):
    """Upsampling layer for a Convolutional Neural Network.

    This layer performs upsampling to increases the spatial resolution of the input tensor.

    Parameters
    ----------
    in_channels : int
        The number of input channels for the convolutional layer.
    out_channels : int
        The number of output channels for the convolutional layer.
    upscale_factor : float
        The factor by which to upscale the spatial dimensions of the input tensor.
    """

    def __init__(
        self, in_channels: int, out_channels: int, upscale_factor: float
    ) -> None:
        super(Upsample, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * (upscale_factor**2),
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Upsample layer.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor with shape (batch_size, in_channels, height, width).

        Returns
        -------
        torch.Tensor
            The upsampled output tensor.
        """
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.relu(x)
        return x


# Residual Block
class ResBlock(nn.Module):
    """Residual Block for a Convolutional Neural Network.

    Implements a residual block commonly used in convolutional neural networks (CNNs) for image processing tasks.
    Residual blocks help to alleviate the vanishing gradient problem and improve the training of deep networks.

    Parameters
    ----------
    channels : list of int
        A list specifying the number of channels for each convolutional layer in the block.
    dropouts : list of float, optional
        A list specifying the dropout rates for each layer. If not provided, no dropout is applied.
    kernel_size : int, optional
        The size of the convolutional kernel. Default is 3.
    stride : int, optional
        The stride of the convolution. Default is 1.
    padding : int, optional
        The padding added to the input. Default is 1.
    """

    def __init__(
        self,
        channels: list[int],
        dropouts: list[float] = [],
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ) -> None:
        super(ResBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.layers.append(
                nn.Conv2d(channels[i], channels[i + 1], kernel_size, stride, padding)
            )
            self.layers.append(nn.BatchNorm2d(channels[i + 1]))
            if len(dropouts) > 0:
                self.layers.append(nn.Dropout(dropouts[i]))
        self.relu = nn.ReLU()
        self.downsample = nn.Sequential(
            nn.Conv2d(channels[0], channels[-1], 1, stride, 0),
            nn.BatchNorm2d(channels[-1]),
        )
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the residual block.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor with shape (batch_size, channels, height, width).

        Returns
        -------
        torch.Tensor
            The output tensor after applying the residual block.
        """
        identity = self.downsample(x)
        out = x.clone()
        for layer in self.layers:
            out = layer(out)
            if isinstance(layer, nn.BatchNorm2d):
                out = self.relu(out)
        out = out + identity
        out = self.relu(out)
        return out
