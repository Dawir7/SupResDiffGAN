import torch
import torch.nn as nn
from torchvision.models import vgg19


class FeatureExtractor(nn.Module):
    def __init__(self, device: torch.device) -> None:
        """Initialize the FeatureExtractor.

        This class uses the VGG19 model pretrained on ImageNet to extract features
        from images. It uses the first 35 layers of the VGG19 model.

        Parameters
        ----------
        device : torch.device
            Device to run the model on.
        """
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True).features
        self.layers = {
            "conv1": nn.Sequential(*list(vgg19_model.children())[:2]).to(device),
            "conv2": nn.Sequential(*list(vgg19_model.children())[2:7]).to(device),
            "conv3": nn.Sequential(*list(vgg19_model.children())[7:12]).to(device),
            "conv4": nn.Sequential(*list(vgg19_model.children())[12:21]).to(device),
            "conv5": nn.Sequential(*list(vgg19_model.children())[21:35]).to(device),
        }
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        self.criterion = nn.L1Loss().to(device)
        self.feature_weights = {
            "conv1": 0.1,
            "conv2": 0.1,
            "conv3": 1.0,
            "conv4": 1.0,
            "conv5": 1.0,
        }

    def forward(self, sr_img: torch.Tensor, hr_img: torch.Tensor) -> torch.Tensor:
        """Forward pass of the FeatureExtractor to compute perceptual loss.

        Parameters
        ----------
        sr_img : torch.Tensor
            Super-resolved image tensor. The tensor should be normalized to the range [-1, 1].
        hr_img : torch.Tensor
            High-resolution image tensor. The tensor should be normalized to the range [-1, 1].

        Returns
        -------
        torch.Tensor
            Perceptual loss.
        """
        # Normalize from [-1, 1] to [0, 1]
        sr_img = (sr_img + 1) / 2
        hr_img = (hr_img + 1) / 2

        # Standardize the images
        sr_img = (sr_img - self.mean) / self.std
        hr_img = (hr_img - self.mean) / self.std

        perceptual_loss = 0
        with torch.no_grad():
            for name, layer in self.layers.items():
                sr_img = layer(sr_img)
                hr_img = layer(hr_img)
                perceptual_loss += self.feature_weights[name] * self.criterion(
                    sr_img, hr_img
                )

        return perceptual_loss
