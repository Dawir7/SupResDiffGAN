import math
from typing import Generator, Optional

import numpy as np
import torch
from tqdm.auto import tqdm

# based on: https://github.com/zsyOAOA/ResShift


def mean_flat(tensor: torch.Tensor) -> torch.Tensor:
    """Compute the mean across all dimensions except the batch dimension.

    Parameters
    ----------
    tensor : torch.Tensor
        The input tensor.

    Returns
    -------
    torch.Tensor
        The mean of the tensor across all dimensions except the batch dimension.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def get_named_eta_schedule(
    num_diffusion_timesteps: int,
    min_noise_level: float = 0.2,
    etas_end: float = 0.99,
    kappa: float = 2.0,
    power: float = 0.3,
) -> np.ndarray:
    """Generate a named eta schedule for the diffusion process.

    Parameters
    ----------
    num_diffusion_timesteps : int
        The number of diffusion timesteps.
    min_noise_level : float, optional
        The minimum noise level. Default is 0.2.
    etas_end : float, optional
        The ending eta value. Default is 0.99.
    kappa : float, optional
        A scaling factor for the variance of the diffusion kernel. Default is 2.0.
    power : float, optional
        The power for the timestep scaling. Default is 0.3.

    Returns
    -------
    np.ndarray
        The eta schedule as a 1-D numpy array.
    """
    etas_start = min(min_noise_level / kappa, min_noise_level)
    increaser = math.exp(
        1 / (num_diffusion_timesteps - 1) * math.log(etas_end / etas_start)
    )
    base = (
        np.ones(
            [
                num_diffusion_timesteps,
            ]
        )
        * increaser
    )
    power_timestep = np.linspace(0, 1, num_diffusion_timesteps, endpoint=True) ** power
    power_timestep *= num_diffusion_timesteps - 1
    sqrt_etas = np.power(base, power_timestep) * etas_start

    return sqrt_etas


def _extract_into_tensor(
    arr: np.ndarray, timesteps: torch.Tensor, broadcast_shape: tuple[int, ...]
) -> torch.Tensor:
    """Extract values from a 1-D numpy array for a batch of indices.

    Parameters
    ----------
    arr : np.ndarray
        The 1-D numpy array.
    timesteps : torch.Tensor
        A tensor of indices into the array to extract.
    broadcast_shape : tuple of int
        A larger shape of K dimensions with the batch dimension equal to the length of timesteps.

    Returns
    -------
    torch.Tensor
        A tensor of shape [batch_size, 1, ...] where the shape has K dimensions.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


class Diffusion:
    """
    Utilities for training and sampling diffusion models.

    Parameters
    ----------
    n_timestep : int
        Number of timesteps for the diffusion process.
    temperature : float, optional
        A scaling factor controlling the variance of the diffusion kernel. Default is 2.0.
    clip_denoised : bool, optional
        If True, clip the denoised signal into [-1, 1]. Default is True.

    Notes
    -----
    - `sqrt_etas` : np.ndarray
        A 1-D numpy array of etas for each diffusion timestep, starting at T and going to 1.
    - `kappa` : float
        A scalar controlling the variance of the diffusion kernel.
    - `model_mean_type` : str
        Determines what the model outputs.
    - `loss_type` : str
        Determines the loss function to use.
    - The timesteps are rescaled internally to match the original paper's range (0 to 1000).
    """

    def __init__(
        self,
        n_timestep: int,
        temperature: float = 2.0,
        clip_denoised: bool = True,
    ):
        self.kappa = temperature
        self.clip_denoised = clip_denoised
        self.n_timestep = n_timestep

        sqrt_etas = get_named_eta_schedule(
            num_diffusion_timesteps=1000, kappa=temperature
        )

        self.setup_scheduler(sqrt_etas)

        if n_timestep != 1000:
            self.timestep_map = []
            use_timesteps = np.linspace(
                0, len(sqrt_etas) - 1, self.n_timestep, dtype=int
            )
            new_sqrt_etas = []

            for i, etas_current in enumerate(sqrt_etas):
                if i in use_timesteps:
                    new_sqrt_etas.append(etas_current)
                    self.timestep_map.append(i)

            sqrt_etas = np.array(new_sqrt_etas)

            self.setup_scheduler(sqrt_etas)

    def setup_scheduler(self, sqrt_etas: np.ndarray) -> None:
        """Set up the scheduler for the diffusion process.

        Parameters
        ----------
        sqrt_etas : np.ndarray
            The square root of eta values for each timestep.
            A 1-D numpy array of etas for each diffusion timestep, starting at T and going to 1.

        Returns
        -------
        None
        """
        self.sqrt_etas = sqrt_etas

        # Use float64 for accuracy.
        self.etas = self.sqrt_etas**2
        assert len(self.etas.shape) == 1, "etas must be 1-D"
        assert (self.etas > 0).all() and (self.etas <= 1).all()

        self.num_timesteps = int(self.etas.shape[0])
        self.etas_prev = np.append(0.0, self.etas[:-1])
        self.alpha = self.etas - self.etas_prev

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.kappa**2 * self.etas_prev / self.etas * self.alpha
        )
        self.posterior_variance_clipped = np.append(
            self.posterior_variance[1], self.posterior_variance[1:]
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(self.posterior_variance_clipped)
        self.posterior_mean_coef1 = self.etas_prev / self.etas
        self.posterior_mean_coef2 = self.alpha / self.etas

        # weight for the mse loss
        weight_loss_mse = (
            0.5 / self.posterior_variance_clipped * (self.alpha / self.etas) ** 2
        )

        # self.weight_loss_mse = np.append(weight_loss_mse[1],  weight_loss_mse[1:])
        self.weight_loss_mse = weight_loss_mse

    def q_mean_variance(
        self, x_start: torch.Tensor, y: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get the distribution q(x_t | x_0).

        Parameters
        ----------
        x_start : torch.Tensor
            The [N x C x ...] tensor of noiseless inputs.
        y : torch.Tensor
            The [N x C x ...] tensor of degraded inputs.
        t : torch.Tensor
            The number of diffusion steps (minus 1). Here, 0 means one step.

        Returns
        -------
        tuple of torch.Tensor
            A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            _extract_into_tensor(self.etas, t, x_start.shape) * (y - x_start) + x_start
        )
        variance = _extract_into_tensor(self.etas, t, x_start.shape) * self.kappa**2
        log_variance = variance.log()
        return mean, variance, log_variance

    def forward(
        self,
        x_start: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        Parameters
        ----------
        x_start : torch.Tensor
            The initial data batch.
        y : torch.Tensor
            The [N x C x ...] tensor of degraded inputs.
        t : torch.Tensor
            The number of diffusion steps (minus 1). Here, 0 means one step.
        noise : torch.Tensor, optional
            If specified, the split-out normal noise. Default is None.

        Returns
        -------
        torch.Tensor
            A noisy version of x_start.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.etas, t, x_start.shape) * (y - x_start)
            + x_start
            + _extract_into_tensor(self.sqrt_etas * self.kappa, t, x_start.shape)
            * noise
        )

    def q_posterior_mean_variance(
        self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the mean and variance of the diffusion posterior: q(x_{t-1} | x_t, x_0).

        Parameters
        ----------
        x_start : torch.Tensor
            The [N x C x ...] tensor of noiseless inputs.
        x_t : torch.Tensor
            The [N x C x ...] tensor at time t.
        t : torch.Tensor
            The number of diffusion steps (minus 1). Here, 0 means one step.

        Returns
        -------
        tuple of torch.Tensor
            A tuple (posterior_mean, posterior_variance, posterior_log_variance_clipped),
            all of x_start's shape.
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_t
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_start
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self,
        model: torch.nn.Module,
        x_t: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor,
        clip_denoised: bool = True,
    ) -> dict[str, torch.Tensor]:
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of the initial x, x_0.

        Parameters
        ----------
        model : torch.nn.Module
            The model, which takes a signal and a batch of timesteps as input.
        x_t : torch.Tensor
            The [N x C x ...] tensor at time t.
        y : torch.Tensor
            The [N x C x ...] tensor of degraded inputs.
        t : torch.Tensor
            A 1-D tensor of timesteps.
        clip_denoised : bool, optional
            If True, clip the denoised signal into [-1, 1]. Default is True.

        Returns
        -------
        dict
            A dictionary with the following keys:
            - 'mean': the model mean output.
            - 'variance': the model variance output.
            - 'log_variance': the log of 'variance'.
        - 'pred_xstart': the prediction for x_0.
        """
        B, C = x_t.shape[:2]
        assert t.shape == (B,)
        model_output = model(
            self._scale_input(x_t, t),
            self.rescale_timesteps(t),
            y,
        )

        model_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        model_log_variance = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )

        pred_xstart = model_output.clamp(-1, 1) if clip_denoised else model_output

        model_mean, _, _ = self.q_posterior_mean_variance(
            x_start=pred_xstart, x_t=x_t, t=t
        )

        assert (
            model_mean.shape
            == model_log_variance.shape
            == pred_xstart.shape
            == x_t.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def p_sample(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor,
        clip_denoised: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Sample x_{t-1} from the model at the given timestep.

        Parameters
        ----------
        model : torch.nn.Module
            The model to sample from.
        x : torch.Tensor
            The current tensor at x_t.
        y : torch.Tensor
            The [N x C x ...] tensor of degraded inputs.
        t : torch.Tensor
            The value of t, starting at 0 for the first diffusion step.
        clip_denoised : bool, optional
            If True, clip the x_start prediction to [-1, 1]. Default is True.

        Returns
        -------
        dict
            A dictionary containing the following keys:
            - 'sample': a random sample from the model.
            - 'pred_xstart': a prediction of x_0.
            - 'mean': the model mean output.
        """
        out = self.p_mean_variance(
            model,
            x,
            y,
            t,
            clip_denoised=clip_denoised,
        )
        noise = torch.randn_like(x)

        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = (
            out["mean"] + nonzero_mask * torch.exp(1.5 * out["log_variance"]) * noise
        )
        return {
            "sample": sample,
            "pred_xstart": out["pred_xstart"],
            "mean": out["mean"],
        }

    def sample(
        self,
        model: torch.nn.Module,
        y: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        progress: bool = False,
    ) -> torch.Tensor:
        """Generate samples from the model.

        Parameters
        ----------
        model : torch.nn.Module
            The model module.
        y : torch.Tensor
            The [N x C x ...] tensor of degraded inputs.
        noise : torch.Tensor, optional
            If specified, the noise from the encoder to sample. Should be of the same shape as `y`.
            Default is None.
        device : torch.device, optional
            If specified, the device to create the samples on. If not specified, use the device of `y`.
            Default is None.
        progress : bool, optional
            If True, show a tqdm progress bar. Default is False.

        Returns
        -------
        torch.Tensor
            A non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive(
            model,
            y,
            noise=noise,
            device=device,
            progress=progress,
        ):
            final = sample["sample"]

        return final

    def p_sample_loop_progressive(
        self,
        model: torch.nn.Module,
        y: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        progress: bool = False,
    ) -> Generator[dict[str, torch.Tensor], None, None]:
        """
        Generate samples from the model and yield intermediate samples from each timestep of diffusion.

        Parameters
        ----------
        model : torch.nn.Module
            The model module.
        y : torch.Tensor
            The [N x C x ...] tensor of degraded inputs.
        noise : torch.Tensor, optional
            If specified, the noise from the encoder to sample. Default is None.
        device : torch.device, optional
            If specified, the device to create the samples on. Default is None.
        progress : bool, optional
            If True, show a tqdm progress bar. Default is False.

        Yields
        ------
        dict
            A dictionary containing intermediate samples and predictions at each timestep.
        """
        with torch.no_grad():
            if device is None:
                device = y.device

            # generating noise
            if noise is None:
                noise = torch.randn_like(y)

            z_sample = self.prior_sample(y, noise)

            indices = list(range(self.num_timesteps))
            indices = indices[::-1]

            indices = tqdm(indices) if progress else indices

            for i in indices:
                t = torch.tensor([i] * y.shape[0], device=device)
                out = self.p_sample(
                    model,
                    z_sample,
                    y,
                    t,
                    clip_denoised=self.clip_denoised,
                )
                yield out
                z_sample = out["sample"]

    def prior_sample(
        self, y: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Generate samples from the prior distribution, i.e., q(x_T|x_0) ~= N(x_T|y, ~).

        Parameters
        ----------
        y : torch.Tensor
            The [N x C x ...] tensor of degraded inputs.
        noise : torch.Tensor, optional
            The [N x C x ...] tensor of Gaussian noise. If None, random noise is generated. Default is None.

        Returns
        -------
        torch.Tensor
            The sampled tensor from the prior distribution.
        """
        if noise is None:
            noise = torch.randn_like(y)

        t = torch.tensor(
            [
                self.num_timesteps - 1,
            ]
            * y.shape[0],
            device=y.device,
        ).long()

        return y + _extract_into_tensor(self.kappa * self.sqrt_etas, t, y.shape) * noise

    def compute_loss(
        self,
        model: torch.nn.Module,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Compute training losses for a single timestep.

        Parameters
        ----------
        model : torch.nn.Module
            The model to evaluate the loss on.
        x_0 : torch.Tensor
            The [N x C x ...] tensor of inputs (ground truth).
        x_1 : torch.Tensor
            The [N x C x ...] tensor of degraded inputs.

        Returns
        -------
        dict
            A dictionary containing:
            - "loss": The computed loss value.
            - "pred_x0": The predicted x_0 tensor.
            - "weights": The weights for the loss computation.

        Notes
        -----
        - `t` : torch.Tensor
            A batch of timestep indices randomly sampled for each input in the batch.
        - `noise` : torch.Tensor
            Gaussian noise added to the input tensor `x_0`.
        - `z_t` : torch.Tensor
            The noisy version of `x_0` after applying the forward diffusion process.
        - `model_output` : torch.Tensor
            The model's prediction for the denoised input.
        - `target` : torch.Tensor
            The ground truth tensor `x_0` used for loss computation.
        """
        t = torch.randint(0, len(self.etas), (x_0.shape[0],), device=x_0.device)

        noise = torch.randn_like(x_0)

        z_t = self.forward(x_0, x_1, t, noise=noise)

        model_output = model(self._scale_input(z_t, t), self.rescale_timesteps(t), x_1)

        target = x_0

        loss = ((target - model_output) ** 2).mean()

        sqrt_etas_torch = torch.from_numpy(self.sqrt_etas).to(device=t.device).float()

        return {
            "loss": loss,
            "pred_x0": model_output,
            "weights": 1 / sqrt_etas_torch[t],
        }

    def _scale_input(self, inputs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Scale the input tensor based on the timestep.

        Parameters
        ----------
        inputs : torch.Tensor
            The input tensor to be scaled.
        t : torch.Tensor
            The timesteps corresponding to the input tensor.

        Returns
        -------
        torch.Tensor
            The scaled input tensor.
        """
        inputs_max = (
            _extract_into_tensor(self.sqrt_etas, t, inputs.shape) * self.kappa * 3 + 1
        )
        return inputs / inputs_max

    def rescale_timesteps(self, t: torch.Tensor) -> torch.Tensor:
        """Rescale the timesteps based on the internal timestep map.

        Parameters
        ----------
        t : torch.Tensor
            The timesteps to be rescaled.

        Returns
        -------
        torch.Tensor
            The rescaled timesteps.
        """
        if hasattr(self, "timestep_map"):
            map_tensor = torch.tensor(self.timestep_map, device=t.device, dtype=t.dtype)
            return map_tensor[t]
        return t
