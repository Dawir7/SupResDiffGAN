from functools import partial
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# source: https://github.com/NVlabs/I2SB/tree/master/i2sb


class ISBDiffusion(nn.Module):
    """Implements the ISB Diffusion process for image super-resolution.

    Parameters
    ----------
    n_timestep : int
        Number of timesteps for the diffusion process.
    deterministic : bool, optional
        If True, the sampling process will be deterministic. Default is False.
    """

    def __init__(self, n_timestep: int, deterministic: bool = False) -> None:
        super().__init__()
        self.deterministic = deterministic

        betas = make_isb_beta_schedule(n_timestep)
        self.sample_steps = n_timestep

        # Compute analytic std: eq 11
        std_fwd = np.sqrt(np.cumsum(betas))
        std_bwd = np.sqrt(np.flip(np.cumsum(np.flip(betas))))
        mu_x0, mu_lr_img, var = compute_gaussian_product_coef(std_fwd, std_bwd)
        std_sb = np.sqrt(var)

        # Tensorize everything
        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("std_fwd", to_torch(std_fwd))
        self.register_buffer("std_bwd", to_torch(std_bwd))
        self.register_buffer("std_sb", to_torch(std_sb))
        self.register_buffer("mu_x0", to_torch(mu_x0))
        self.register_buffer("mu_lr_img", to_torch(mu_lr_img))

    def get_std_fwd(
        self, step: int, xdim: Optional[tuple[int, ...]] = None
    ) -> torch.Tensor:
        """Get the forward standard deviation for a given step.

        Parameters
        ----------
        step : int
            The current timestep.
        xdim : tuple of int, optional
            The dimensions of the input tensor. If provided, the standard deviation
            will be unsqueezed to match these dimensions.

        Returns
        -------
        torch.Tensor
            The forward standard deviation for the given step.
        """
        std_fwd = self.std_fwd[step]
        return std_fwd if xdim is None else unsqueeze_xdim(std_fwd, xdim)

    def forward(
        self, step: int, x0: torch.Tensor, lr_img: torch.Tensor, ot_ode: bool = False
    ) -> torch.Tensor:
        """Sample q(x_t | x_0, x_1), i.e., Equation 11.

        Parameters
        ----------
        step : int
            The current timestep.
        x0 : torch.Tensor
            The high-resolution image tensor.
        lr_img : torch.Tensor
            The low-resolution image tensor.
        ot_ode : bool, optional
            If True, the sampling process will be deterministic. Default is False.

        Returns
        -------
        torch.Tensor
            The sampled tensor at the current timestep.
        """
        assert x0.shape == lr_img.shape
        batch, *xdim = x0.shape

        mu_x0 = unsqueeze_xdim(self.mu_x0[step], xdim)
        mu_lr_img = unsqueeze_xdim(self.mu_lr_img[step], xdim)
        std_sb = unsqueeze_xdim(self.std_sb[step], xdim)

        xt = mu_x0 * x0 + mu_lr_img * lr_img
        if not ot_ode:
            xt = xt + std_sb * torch.randn_like(xt)
        return xt.detach()

    def p_posterior(
        self,
        nprev: int,
        n: int,
        x_n: torch.Tensor,
        x0: torch.Tensor,
        ot_ode: bool = False,
    ) -> torch.Tensor:
        """Sample p(x_{nprev} | x_n, x_0), i.e., Equation 4.

        Parameters
        ----------
        nprev : int
            The previous timestep.
        n : int
            The current timestep.
        x_n : torch.Tensor
            The tensor at the current timestep.
        x0 : torch.Tensor
            The high-resolution image tensor.
        ot_ode : bool, optional
            If True, the sampling process will be deterministic. Default is False.

        Returns
        -------
        torch.Tensor
            The sampled tensor at the previous timestep.
        """
        assert nprev < n
        std_n = self.std_fwd[n]
        std_nprev = self.std_fwd[nprev]
        std_delta = (std_n**2 - std_nprev**2).sqrt()

        mu_x0, mu_xn, var = compute_gaussian_product_coef(std_nprev, std_delta)

        xt_prev = mu_x0 * x0 + mu_xn * x_n
        if not ot_ode and nprev > 0:
            xt_prev = xt_prev + var.sqrt() * torch.randn_like(xt_prev)

        return xt_prev

    def compute_label(
        self, step: int, x0: torch.Tensor, xt: torch.Tensor
    ) -> torch.Tensor:
        """Compute the label for training, i.e., Equation 12.

        Parameters
        ----------
        step : int
            The current timestep.
        x0 : torch.Tensor
            The high-resolution image tensor.
        xt : torch.Tensor
            The tensor at the current timestep.

        Returns
        -------
        torch.Tensor
            The computed label tensor.
        """
        std_fwd = self.get_std_fwd(step, xdim=x0.shape[1:])
        label = (xt - x0) / std_fwd
        return label.detach()

    def compute_pred_x0(
        self,
        step: int,
        xt: torch.Tensor,
        net_out: torch.Tensor,
        clip_denoise: bool = False,
    ) -> torch.Tensor:
        """Recover x0 from the network output. This should be the inverse of Equation 12.

        Parameters
        ----------
        step : int
            The current timestep.
        xt : torch.Tensor
            The tensor at the current timestep.
        net_out : torch.Tensor
            The network output tensor.
        clip_denoise : bool, optional
            If True, the predicted x0 will be clipped to the range [-1, 1]. Default is False.

        Returns
        -------
        torch.Tensor
            The predicted x0 tensor.
        """
        std_fwd = self.get_std_fwd(step, xdim=xt.shape[1:])
        pred_x0 = xt - std_fwd * net_out
        if clip_denoise:
            pred_x0.clamp_(-1.0, 1.0)
        return pred_x0

    def compute_loss(
        self, model: nn.Module, x0: torch.Tensor, lr_img: torch.Tensor
    ) -> torch.Tensor:
        """Compute the loss for the diffusion process.

        Parameters
        ----------
        model : nn.Module
            The model used for prediction.
        x0 : torch.Tensor
            The high-resolution image tensor.
        lr_img : torch.Tensor
            The low-resolution image tensor.

        Returns
        -------
        torch.Tensor
            The computed loss value.
        """
        step = torch.randint(0, len(self.betas), (x0.shape[0],), device=x0.device)

        xt = self.forward(step, x0, lr_img, ot_ode=self.deterministic)
        label = self.compute_label(step, x0, xt)

        pred = model(xt, step, lr_img)

        loss = F.mse_loss(pred, label)

        return loss

    def sample(
        self,
        model: nn.Module,
        lr_img: torch.Tensor,
        log_steps: Optional[np.ndarray] = None,
        verbose: bool = False,
    ) -> torch.Tensor:
        """Perform sampling to generate a high-resolution image.

        Parameters
        ----------
        model : nn.Module
            The model used for prediction.
        lr_img : torch.Tensor
            The low-resolution image tensor.
        log_steps : np.ndarray, optional
            The steps at which to log progress. Default is None.
        verbose : bool, optional
            If True, display progress using tqdm. Default is False.

        Returns
        -------
        torch.Tensor
            The generated high-resolution image tensor.
        """
        steps = np.linspace(0, len(self.betas) - 1, self.sample_steps, dtype=int)

        xt = lr_img.detach().to(lr_img.device)
        log_steps = log_steps or steps
        assert steps[0] == log_steps[0] == 0

        steps = steps[::-1]

        pair_steps = zip(steps[1:], steps[:-1])
        pair_steps = (
            tqdm(pair_steps, desc="DDPM sampling", total=len(steps) - 1)
            if verbose
            else pair_steps
        )
        for prev_step, step in pair_steps:
            assert prev_step < step, f"{prev_step=}, {step=}"

            model_step = torch.full(
                (xt.shape[0],), step, device=lr_img.device, dtype=torch.long
            )
            out = model(xt, model_step, lr_img)
            pred_x0 = self.compute_pred_x0(model_step, xt, out, clip_denoise=True)

            xt = self.p_posterior(
                prev_step, step, xt, pred_x0, ot_ode=self.deterministic
            )

        return pred_x0

    def set_timesteps(self, timesteps: int) -> None:
        """Set the number of timesteps.

        Parameters
        ----------
        timesteps : int
            The number of timesteps to set for the diffusion process.

        Returns
        -------
        None
        """
        self.sample_steps = timesteps


def compute_gaussian_product_coef(
    sigma1: np.ndarray, sigma2: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the coefficients for the Gaussian product.

    Given p1 = N(x_t|x_0, sigma_1**2) and p2 = N(x_t|x_1, sigma_2**2)
    return p1 * p2 = N(x_t| coef1 * x0 + coef2 * lr_img, var)

    Parameters
    ----------
    sigma1 : np.ndarray
        The standard deviation of the first Gaussian.
    sigma2 : np.ndarray
        The standard deviation of the second Gaussian.

    Returns
    -------
    tuple of np.ndarray
        Coefficients for the Gaussian product and the variance.
    """
    denom = sigma1**2 + sigma2**2
    coef1 = sigma2**2 / denom
    coef2 = sigma1**2 / denom
    var = (sigma1**2 * sigma2**2) / denom
    return coef1, coef2, var


def unsqueeze_xdim(z: torch.Tensor, xdim: tuple[int, ...]) -> torch.Tensor:
    """Unsqueeze a tensor to match the given dimensions.

    Parameters
    ----------
    z : torch.Tensor
        The input tensor.
    xdim : tuple of int
        The target dimensions.

    Returns
    -------
    torch.Tensor
        The unsqueezed tensor.
    """
    bc_dim = (...,) + (None,) * len(xdim)
    return z[bc_dim]


def make_isb_beta_schedule(
    n_timestep: int = 1000, linear_start: float = 1e-4, linear_end: float = 2e-2
) -> np.ndarray:
    """Create a beta schedule for the ISB diffusion process.

    Parameters
    ----------
    n_timestep : int, optional
        The number of timesteps. Default is 1000.
    linear_start : float, optional
        The starting value for the linear schedule. Default is 1e-4.
    linear_end : float, optional
        The ending value for the linear schedule. Default is 2e-2.

    Returns
    -------
    np.ndarray
        The beta schedule.
    """
    betas = (
        torch.linspace(
            linear_start**0.5, linear_end**0.5, n_timestep, dtype=torch.float64
        )
        ** 2
    ).numpy()

    betas = np.concatenate(
        [betas[: n_timestep // 2], np.flip(betas[: n_timestep // 2])]
    )
    return betas
