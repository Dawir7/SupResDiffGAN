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
    low_condition : bool, optional
        Whether to condition the model on low-resolution input images. If `True`,
        the model will accept both low-resolution and high-resolution images as inputs.
        If `False`, the model will only accept high-resolution images as input.
    """

    def __init__(
        self,
        channels: list[int] = [64, 96, 128, 512],
        low_condition: bool = True,
    ):
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
        self.low_condition = low_condition

        in_channels = 6 if low_condition else 3
        out_channels = 3

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

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, x_low=None) -> torch.Tensor:
        """Forward pass of the UNet.

        Parameters
        ----------
        x_t : torch.Tensor
            The generated image at the current timestep.
        t : torch.Tensor
            The current timestep or noise level in the diffusion process.
        x_low : torch.Tensor, optional
            The low-resolution image to condition the model on.
            If `low_condition` is `False`, this argument will be ignored.

        Returns
        -------
        torch.Tensor
            The generated high-resolution image. The output tensor will be representing the super-resolved image.
        """
        x_in = torch.cat([x_t, x_low], dim=1) if self.low_condition else x_t

        return self.unet(x_in, timestep=t.float()).sample
