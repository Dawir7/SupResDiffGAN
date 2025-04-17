import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import vgg19

# source: https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/esrgan/models.py


class FeatureExtractor(nn.Module):
    def __init__(self) -> None:
        """Initialize the FeatureExtractor.

        This class uses the VGG19 model pretrained on ImageNet to extract features
        from images. It uses the first 35 layers of the VGG19 model.
        """
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.vgg19_54 = nn.Sequential(*list(vgg19_model.features.children())[:35])
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Forward pass of the FeatureExtractor.

        Parameters
        ----------
        img : torch.Tensor
            Input image tensor. The tensor should be normalized to the range [-1, 1].

        Returns
        -------
        torch.Tensor
            Output tensor after passing through the VGG19 layers.
        """
        img = (img + 1) / 2

        img = self.normalize(img)
        return self.vgg19_54(img)
