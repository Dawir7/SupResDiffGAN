import torch
import torch.nn as nn
from torchvision.models import vgg19


class VGGLoss(nn.Module):
    """VGGLoss class for perceptual loss using VGG19 features.

    Parameters
    ----------
    device : torch.device, optional
        Device to run the model on (default is CPU).
    """

    def __init__(self, device: torch.device = torch.device("cpu")) -> None:
        super(VGGLoss, self).__init__()
        self.device = device
        self.vgg = vgg19(pretrained=True).features[:36].eval().to(self.device)
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.mse_loss = nn.MSELoss()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Forward pass to compute the VGG perceptual loss.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, C, H, W).
        y : torch.Tensor
            Target tensor of shape (N, C, H, W).

        Returns
        -------
        torch.Tensor
            Computed VGG perceptual loss.
        """
        x_features = self.vgg(x.to(self.device))
        y_features = self.vgg(y.to(self.device))
        loss = self.mse_loss(x_features, y_features)
        return loss
