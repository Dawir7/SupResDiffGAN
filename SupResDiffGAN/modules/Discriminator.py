import torch
import torch.nn as nn
from torchvision.models import resnet50


# Discriminator
class ResnetDiscriminator(nn.Module):
    """ResNet-based Discriminator network.

    This network uses a pretrained ResNet50 model with classification
    head to distinguish between real high-resolution images and
    generated high-resolution images.
    """

    def __init__(self) -> None:
        super(ResnetDiscriminator, self).__init__()
        self.resnet50 = resnet50(pretrained=True)
        self.features = nn.Sequential(*list(self.resnet50.children())[:-2])

        # Determine the number of output channels dynamically
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            dummy_output = self.features(dummy_input)
        feature_channels = dummy_output.shape[1]

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feature_channels, 4096, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(4096, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the discriminator.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor representing the discriminator's prediction.
        """
        x = self.features(x)
        x = self.classifier(x)
        return x


class Discriminator(nn.Module):
    """Discriminator network.

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

        in_channels = channels[0]
        blocks = [DiscriminatorBlock(in_channels, in_channels, 2)]
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
        """Forward pass of the Discriminator network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, C, H, W), where N is the batch size,
            C is the number of channels, H is the height, and W is the width.

        Returns
        -------
        torch.Tensor
            Output tensor representing the discriminator's prediction.
            The output shape is (N, 1, 1, 1), where N is the batch size.
        """
        x = self.lrelu(self.conv1(x))
        x = self.blocks(x)
        x = self.classifier(x)
        return x


# Discriminator Block
class DiscriminatorBlock(nn.Module):
    """Discriminator block.

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
