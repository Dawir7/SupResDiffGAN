import torch
import torch.nn as nn


# Discriminator
class Discriminator(nn.Module):
    """Discriminator network for SRGAN.

    This network distinguishes between real high-resolution images and
    generated high-resolution images.

    Parameters
    ----------
    in_channels : int
        Number of input channels (e.g., 3 for RGB images).
    channels : list of int
        List of channel sizes for each convolutional layer.
    """

    def __init__(self, in_channels: int, channels: list[int]) -> None:
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, channels[0], kernel_size=3, stride=1, padding=1
        )
        self.lrelu = nn.LeakyReLU(0.2)

        blocks = []
        in_channels = channels[0]
        for out_channels in channels[1:]:
            blocks.append(DiscriminatorBlock(in_channels, out_channels, 1))
            blocks.append(DiscriminatorBlock(out_channels, out_channels, 2))
            in_channels = out_channels
        self.blocks = nn.Sequential(*blocks)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels[-1], channels[-1] * 2, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels[-1] * 2, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the discriminator.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, C, H, W).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (N, 1).
        """
        x = self.lrelu(self.conv1(x))
        x = self.blocks(x)
        x = self.classifier(x)
        return x


# Discriminator Block
class DiscriminatorBlock(nn.Module):
    """Discriminator block for SRGAN.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int
        Stride for the convolutional layer.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super(DiscriminatorBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the discriminator block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, C, H, W).

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        x = self.lrelu(self.bn(self.conv(x)))
        return x
