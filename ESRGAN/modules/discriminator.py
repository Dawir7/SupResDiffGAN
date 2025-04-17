import torch
import torch.nn as nn

# source: https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/esrgan/models.py


class Discriminator(nn.Module):
    def __init__(self, in_channels: int = 3) -> None:
        """Initialize the Discriminator model.

        Parameters
        ----------
        in_channels : int, optional
            Number of input channels (default is 3).
        """
        super(Discriminator, self).__init__()

        def discriminator_block(
            in_filters: int, out_filters: int, first_block: bool = False
        ) -> list[nn.Module]:
            """Create a block for the discriminator model.

            Parameters
            ----------
            in_filters : int
                Number of input filters.
            out_filters : int
                Number of output filters.
            first_block : bool, optional
                Whether this is the first block (default is False).

            Returns
            -------
            list of nn.Module
                List of layers in the block.
            """
            layers = []
            layers.append(
                nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1)
            )
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(
                nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1)
            )
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = in_channels
        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(
                discriminator_block(in_filters, out_filters, first_block=(i == 0))
            )
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Forward pass of the discriminator.

        Parameters
        ----------
        img : torch.Tensor
            Input image tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        return self.model(img)
