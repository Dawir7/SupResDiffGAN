import math
from typing import Literal, Optional

import numpy as np
import torch


# Diffusion process
class Diffusion:
    """Implements a diffusion process for generating images.

    Parameters
    ----------
    timesteps : int, optional
        The number of timesteps in the diffusion process. Defaults to 1000.
    beta_type : str, optional
        The type of beta schedule to use. Can be 'cosine' or 'linear'. Defaults to 'cosine'.
    posterior_type : str, optional
        The type of posterior to use. Can be 'ddpm' or 'ddim'. Defaults to 'ddpm'.
    """

    def __init__(
        self,
        timesteps: int = 1000,
        beta_type: Literal["cosine", "linear"] = "cosine",
        posterior_type: Literal["ddpm", "ddim"] = "ddpm",
    ):
        self.posterior_type = posterior_type

        self.beta_type = beta_type
        self.set_timesteps(timesteps)

    def forward(
        self, x_0: torch.Tensor, t: torch.Tensor, epsilon: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Performs the forward pass of the diffusion process, adding noise to an image.

        Parameters
        ----------
        x_0 : torch.Tensor
            The original image tensor.
        t : torch.Tensor
            The tensor containing the timestep `t` for each image in the batch. This determines the
            amount of noise to add to each image.
        epsilon : torch.Tensor, optional
            An external noise tensor. If None, noise is sampled internally using `torch.randn_like`.

        Returns
        -------
        torch.Tensor
            The noised image tensor at timestep `t`.
        """
        if epsilon is None:
            epsilon = torch.randn_like(x_0)

        x_t = apply(np.sqrt(self.alpha_bar), t, x_0) + apply(
            np.sqrt(1 - self.alpha_bar), t, epsilon
        )  # we add noise to x_0
        return x_t

    def compute_loss(
        self, unet: torch.nn.Module, x_0: torch.Tensor, lr_img: torch.Tensor
    ) -> torch.Tensor:
        """Computes the loss for a batch of images using a U-Net model.

        Parameters
        ----------
        unet : UNet
            The U-Net model used for predicting the original images from their noised versions.
        x_0 : torch.Tensor
            The original images tensor.
        lr_img : torch.Tensor
            The low resolution images tensor used as additional input to the U-Net model.

        Returns
        -------
        torch.Tensor
            The computed loss, averaged over the batch.
        """

        timesteps = torch.randint(
            0, self.timesteps, (x_0.shape[0],), device=x_0.device, dtype=torch.long
        )
        epsilon = torch.randn_like(x_0, device=x_0.device, dtype=x_0.dtype)
        x_t = self.forward(x_0, timesteps, epsilon)

        pred_x_0 = unet(torch.cat((x_t, lr_img), dim=1), timesteps)
        mse = (pred_x_0 - x_0).pow(2).mean(dim=(1, 2, 3))  # loss
        loss = apply(np.sqrt(self.alpha_bar), timesteps, mse).mean()

        return loss

    def posterior(
        self,
        x_t: torch.Tensor,
        x_0: torch.Tensor,
        t: torch.Tensor,
        epsilon: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes the posterior estimate of the original image given a noised image at timestep t.

        Parameters
        ----------
        x_t : torch.Tensor
            The noised image tensor at timestep t.
        x_0 : torch.Tensor
            The original image tensor.
        t : torch.Tensor
            The tensor containing the timestep t for each image in the batch.
        epsilon : torch.Tensor, optional
            An external noise tensor. If None, noise is sampled internally.

        Returns
        -------
        torch.Tensor
            The posterior estimate of the original image tensor.
        """
        if epsilon is None:
            epsilon = torch.randn_like(x_0)

        x_prev = (
            apply(
                (np.sqrt(self.alpha_bar_prev) * self.beta) / (1 - self.alpha_bar),
                t,
                x_0,
            )
            + apply(
                (np.sqrt(self.alpha) * (1 - self.alpha_bar_prev))
                / (1 - self.alpha_bar),
                t,
                x_t,
            )
            + apply(
                np.sqrt((1 - self.alpha_bar_prev) / (1 - self.alpha_bar) * self.beta),
                t,
                epsilon,
            )
        )

        return x_prev

    def ddim_posterior(
        self,
        x_t: torch.Tensor,
        x_0: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the DDIM (Denoising Diffusion Implicit Models) posterior.

        Parameters
        ----------
        x_t : torch.Tensor
            The noised image tensor at timestep t.
        x_0 : torch.Tensor
            The original image tensor.
        t : torch.Tensor
            The tensor containing the timestep t for each image in the batch.

        Returns
        -------
        torch.Tensor
            The posterior estimate of the original image tensor.
        """
        x_t_prev = apply(
            np.sqrt((1 - self.alpha_bar_prev) / (1 - self.alpha_bar)), t, x_t
        ) + apply(
            np.sqrt(self.alpha_bar_prev)
            - np.sqrt(
                self.alpha_bar * (1 - self.alpha_bar_prev) / (1 - self.alpha_bar)
            ),
            t,
            x_0,
        )
        return x_t_prev

    def sample(
        self,
        unet: torch.nn.Module,
        lr_img: torch.Tensor,
        sample_size: tuple[int] = (1, 3, 64, 64),
    ) -> torch.Tensor:
        """Generates new images by sampling from the learned distribution using a U-Net model.

        Parameters
        ----------
        unet : UNet
            The U-Net model used for predicting the denoised images at each reverse timestep.
        lr_img : torch.Tensor
            The low resolution images tensor used as additional input to the U-Net model.
        sample_size : tuple of int, optional
            The size of the tensor to be sampled, formatted as (batch_size, channels, height, width).
            Defaults to (1, 3, 64, 64).

        Returns
        -------
        torch.Tensor
            The tensor of generated images, detached and moved to CPU.
        """
        with torch.no_grad():
            x_t = torch.randn(sample_size, device=unet.unet.device)
            timesteps = list(range(self.timesteps))[::-1]

            for i in timesteps:
                t = torch.Tensor([i] * x_t.shape[0]).long().to(x_t.device)
                pred_x_0 = torch.clamp(
                    unet(
                        torch.cat((x_t, lr_img), dim=1),
                        self._scale_timesteps(t),
                    ),
                    -1,
                    1,
                )

                if i > 0:
                    if self.posterior_type == "ddim":
                        x_t = self.ddim_posterior(x_t, pred_x_0, t)
                    elif self.posterior_type == "ddpm":
                        x_t = self.posterior(x_t, pred_x_0, t)
                else:
                    x_t = pred_x_0

        return x_t.detach()

    def set_timesteps(self, timesteps: int) -> None:
        """Set the number of timesteps.

        Set the number of timesteps and update the related parameters
        like alpha, beta, and alpha_bar.
        Mostly for validation purposes to set new timesteps for inference.

        Parameters
        ----------
        timesteps : int
            The number of timesteps to set for the diffusion process.

        Returns
        -------
        None
        """
        self.timesteps = timesteps
        if self.beta_type == "cosine":
            self.beta = beta_schedule_cosine(1000)
        elif self.beta_type == "linear":
            self.beta = beta_schedule_linear(1000)
        self.set_alfas()

        if self.timesteps != 1000:
            self.timestep_map = []
            use_timesteps = np.linspace(
                0, len(self.beta) - 1, self.timesteps, dtype=int
            )

            last_alpha_cumprod = 1.0
            new_betas = []
            for i, alpha_cumprod in enumerate(self.alpha_bar):
                if i in use_timesteps:
                    new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                    last_alpha_cumprod = alpha_cumprod
                    self.timestep_map.append(i)
            betas = np.array(new_betas)
            self.beta = betas
            self.set_alfas()

    def set_alfas(self) -> None:
        """Compute and set the alpha, alpha_bar, and related parameters for the diffusion process.

        This method calculates the following:
        - `alpha`: The complement of beta values (1 - beta).
        - `alpha_bar`: The cumulative product of alpha values.
        - `alpha_bars_torch`: A PyTorch tensor version of `alpha_bar` for use in PyTorch operations.
        - `alpha_bar_prev`: The cumulative product of alpha values shifted by one timestep.

        Returns
        -------
        None
        """
        self.alpha = 1 - self.beta
        self.alpha_bar = np.cumprod(self.alpha)
        self.alpha_bars_torch = torch.tensor(self.alpha_bar, dtype=torch.float32)
        self.alpha_bar_prev = np.concatenate([np.array([1]), self.alpha_bar[:-1]])

    def set_posterior_type(self, posterior_type: Literal["ddpm", "ddim"]) -> None:
        """Set the type of posterior to use.

        Set the type of posterior to use for the diffusion process.
        Mostly for validation purposes to set new posterior type for inference.

        Parameters
        ----------
        posterior_type : str
            The type of posterior to use. Can be 'ddpm' or 'ddim'.

        Returns
        -------
        None
        """
        self.posterior_type = posterior_type

    def _scale_timesteps(self, t: torch.Tensor) -> torch.Tensor:
        """Rescale the timesteps based on the internal timestep map.

        Parameters
        ----------
        t : torch.Tensor
            A tensor of timesteps to be rescaled. The shape should match the batch size.

        Returns
        -------
        torch.Tensor
            A tensor of rescaled timesteps. If `timestep_map` is defined, the timesteps are mapped
            to a reduced range; otherwise, the input timesteps are returned unchanged.

        Notes
        -----
        - `timestep_map` : list[int]
            A mapping of timesteps to a reduced range if the number of timesteps is less than 1000.
            This is used to adjust the timesteps for inference or training with a reduced number
            of timesteps.
        """
        if hasattr(self, "timestep_map"):
            map_tensor = torch.tensor(self.timestep_map, device=t.device, dtype=t.dtype)
            return map_tensor[t]
        return t


# helper functions
def apply(
    factors: np.ndarray, timesteps: torch.Tensor, x: torch.Tensor, multiply: bool = True
) -> torch.Tensor:
    """Extract values from factors with certain timesteps and apply them to x.

    Parameters
    ----------
    factors : numpy.ndarray
        A 1-dimensional numpy array from which values are extracted.
    timesteps : torch.Tensor
        A tensor of indices into the `factors` array to extract values from. These indices correspond
        to specific timesteps in a process.
    x : torch.Tensor
        A tensor to which the extracted values are applied. This tensor should have a larger shape,
        typically with its first dimension (batch dimension) equal to the length of `timesteps`.
    multiply : bool, optional
        A flag determining the mode of application of the extracted values to `x`. If True (default),
        the extracted values are multiplied with `x`. If False, the extracted values are expanded to
        match the shape of `x` without multiplication.

    Returns
    -------
    torch.Tensor
        A tensor resulting from the application of the extracted values to `x`. If `multiply` is True,
        this tensor is the product of `x` and the extracted values, expanded to match the shape of `x`.
        If `multiply` is False, it is the extracted values expanded to match the shape of `x`. The
        returned tensor has the same number of dimensions as `x`, with the first dimension (batch size)
        equal to the length of `timesteps`.
    """
    res = torch.from_numpy(factors).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(x.shape):
        res = res[..., None]
    return (res.expand(x.shape) * x) if multiply else res.expand(x.shape)


# Schedules
def beta_schedule_linear(num_diffusion_timesteps: int) -> np.ndarray:
    """Generates a linear beta schedule for the diffusion process.

    Parameters
    ----------
    num_diffusion_timesteps : int
        The total number of timesteps in the diffusion process. This determines the length of the
        beta schedule array.

    Returns
    -------
    np.ndarray
        An array of beta values for each timestep in the diffusion process. The array has a length
        equal to `num_diffusion_timesteps` and contains float64 values.
    """
    scale = 1000 / num_diffusion_timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return (
        np.linspace(
            beta_start**0.5,
            beta_end**0.5,
            num_diffusion_timesteps,
            dtype=np.float64,
        )
        ** 2
    )


def beta_schedule_cosine(num_diffusion_timesteps: int) -> np.ndarray:
    """Generates a cosine beta schedule for the diffusion process.

    Parameters
    ----------
    num_diffusion_timesteps : int
        The total number of timesteps in the diffusion process. This determines the length of the
        beta schedule array.

    Returns
    -------
    np.ndarray
        An array of beta values for each timestep in the diffusion process. The array has a length
        equal to `num_diffusion_timesteps` and contains float64 values.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        max_beta = 0.999
        t1 = float(i) / num_diffusion_timesteps
        t2 = float(i + 1) / num_diffusion_timesteps
        alpha = lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
        betas.append(min(1 - alpha(t2) / alpha(t1), max_beta))
    return np.array(betas)
