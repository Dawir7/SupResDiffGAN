import torch
import torch.nn as nn

# source: https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/esrgan/models.py


class DenseResidualBlock(nn.Module):
    """
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    """

    def __init__(self, filters: int, res_scale: float = 0.2) -> None:
        """Initialize the DenseResidualBlock.

        Parameters
        ----------
        filters : int
            Number of filters for the convolutional layers.
        res_scale : float, optional
            Residual scaling factor (default is 0.2).
        """
        super(DenseResidualBlock, self).__init__()
        self.res_scale = res_scale

        def block(in_features: int, non_linearity: bool = True) -> nn.Sequential:
            """Create a convolutional block.

            Parameters
            ----------
            in_features : int
                Number of input features.
            non_linearity : bool, optional
                Whether to include a non-linearity (default is True).

            Returns
            -------
            nn.Sequential
                Sequential container of layers.
            """
            layers = [nn.Conv2d(in_features, filters, 3, 1, 1, bias=True)]
            if non_linearity:
                layers += [nn.LeakyReLU()]
            return nn.Sequential(*layers)

        self.b1 = block(in_features=1 * filters)
        self.b2 = block(in_features=2 * filters)
        self.b3 = block(in_features=3 * filters)
        self.b4 = block(in_features=4 * filters)
        self.b5 = block(in_features=5 * filters, non_linearity=False)
        self.blocks = [self.b1, self.b2, self.b3, self.b4, self.b5]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the DenseResidualBlock.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        inputs = x
        for block in self.blocks:
            out = block(inputs)
            inputs = torch.cat([inputs, out], 1)
        return out.mul(self.res_scale) + x


class ResidualInResidualDenseBlock(nn.Module):
    def __init__(self, filters: int, res_scale: float = 0.2) -> None:
        """Initialize the ResidualInResidualDenseBlock.

        Parameters
        ----------
        filters : int
            Number of filters for the convolutional layers.
        res_scale : float, optional
            Residual scaling factor (default is 0.2).
        """
        super(ResidualInResidualDenseBlock, self).__init__()
        self.res_scale = res_scale
        self.dense_blocks = nn.Sequential(
            DenseResidualBlock(filters),
            DenseResidualBlock(filters),
            DenseResidualBlock(filters),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the ResidualInResidualDenseBlock.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        return self.dense_blocks(x).mul(self.res_scale) + x


class GeneratorRRDB(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filters: int = 64,
        num_resblocks: int = 23,
        scale_factor: int = 4,
    ) -> None:
        """Initialize the GeneratorRRDB.

        Parameters
        ----------
        in_channels : int
            Number of input channels for the low-resolution image.
        out_channels : int
            Number of output channels for the high-resolution image.
        filters : int, optional
            Number of filters for the convolutional layers (default is 64).
        num_resblocks : int, optional
            Number of residual blocks (default is 23).
        scale_factor : int, optional
            Upsampling scale factor (default is 4).
        """
        super(GeneratorRRDB, self).__init__()
        num_upsample = int(scale_factor ** (1 / 2))

        # First layer
        self.conv1 = nn.Conv2d(in_channels, filters, kernel_size=3, stride=1, padding=1)
        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualInResidualDenseBlock(filters) for _ in range(num_resblocks)]
        )
        # Second conv layer post residual blocks
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
        # Upsampling layers
        upsample_layers = []
        for _ in range(num_upsample):
            upsample_layers += [
                nn.Conv2d(filters, filters * 4, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(),
                nn.PixelShuffle(upscale_factor=num_upsample),
            ]
        self.upsampling = nn.Sequential(*upsample_layers)
        # Final output block
        self.conv3 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(filters, out_channels, kernel_size=3, stride=1, padding=1),
        )
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the GeneratorRRDB.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv3(out)
        out = self.tanh(out)
        return out
