import torch
from diffusers.models.unets import UNet2DModel


class UNet(torch.nn.Module):
    """A UNet for image generation.

    This model architecture is based on a UNet structure, utilizing ResNet blocks
    for both downsampling and upsampling paths. It optionally includes attention
    mechanisms to enhance feature representation.

    Parameters
    ----------
    channels : list of int
        The number of channels for each block in the UNet architecture. Defines the
        depth and capacity of the model.
    """

    def __init__(self, channels: list[int] = [64, 96, 128, 512]):
        super().__init__()
        self.channels = channels
        self.layers_per_block = 2
        self.downblock = "ResnetDownsampleBlock2D"
        self.upblock = "ResnetUpsampleBlock2D"
        self.add_attention = [
            False,
            False,
            True,
            True,
        ]  # Attention added to last two blocks
        self.attention_head_dim = 32

        in_channels = 8  # it is determined by the output of encoder (VAE has 4 channels output so 4*2=8)
        out_channels = 4

        self.unet = UNet2DModel(
            in_channels=in_channels,
            out_channels=out_channels,
            block_out_channels=self.channels,
            layers_per_block=self.layers_per_block,
            down_block_types=tuple(self.downblock for _ in range(len(self.channels))),
            up_block_types=tuple(self.upblock for _ in range(len(self.channels))),
            add_attention=self.add_attention,
            attention_head_dim=self.attention_head_dim,
        )

    def forward(
        self, lr_img: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass of the UNet.

        Parameters
        ----------
        lr_img : torch.Tensor
            The low-resolution image being processed. This tensor should have
            a shape compatible with the input requirements of the UNet model, typically including dimensions
            for batch size, channels, height, and width.
        x_t : torch.Tensor
            The high-resolution image in t step of the diffusion process.
        t : torch.Tensor
            The current timestep or noise level in the diffusion process. This should be a single-dimensional
            tensor with the same batch size as `lr_img`.

        Returns
        -------
        torch.Tensor
            The generated high-resolution image. The output tensor will have the same shape as `lr_img`,
            representing the super-resolved image.
        """
        x = torch.cat([x_t, lr_img], dim=1)
        return self.unet(x, timestep=t.float()).sample
