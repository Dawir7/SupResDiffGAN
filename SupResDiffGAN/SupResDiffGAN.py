import time

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb
from matplotlib import pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


class SupResDiffGAN(pl.LightningModule):
    """SupResDiffGAN class for Super-Resolution Diffusion Generative Adversarial Network.

    Parameters
    ----------
    ae : nn.Module
        Autoencoder model.
    discriminator : nn.Module
        Discriminator model.
    unet : nn.Module
        UNet generator model.
    diffusion : nn.Module
        Diffusion model.
    learning_rate : float, optional
        Learning rate for the optimizers (default is 1e-4).
    alfa_perceptual : float, optional
        Weight for the perceptual loss (default is 1e-3).
    alfa_adv : float, optional
        Weight for the adversarial loss (default is 1e-2).
    vgg_loss : nn.Module | None, optional
        The VGG loss module for perceptual loss (default is None).
        If None, the perceptual loss will not be used.
    """

    def __init__(
        self,
        ae: nn.Module,
        discriminator: nn.Module,
        unet: nn.Module,
        diffusion: nn.Module,
        learning_rate: float = 1e-4,
        alfa_perceptual: float = 1e-3,
        alfa_adv: float = 1e-2,
        vgg_loss: nn.Module | None = None,
    ) -> None:
        super(SupResDiffGAN, self).__init__()
        self.ae = ae
        self.discriminator = discriminator
        self.generator = unet
        self.diffusion = diffusion

        self.vgg_loss = vgg_loss
        self.content_loss = nn.MSELoss()
        self.adversarial_loss = nn.BCELoss()
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(
            self.device
        )

        self.lr = learning_rate
        self.alfa_adv = alfa_adv
        self.alfa_perceptual = alfa_perceptual
        self.betas = (0.9, 0.999)

        for param in self.ae.parameters():
            param.requires_grad = False

        self.automatic_optimization = False  # Disable automatic optimization

        self.test_step_outputs = []
        self.ema_weight = 0.97
        self.ema_mean = 0.5
        self.s = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the SupResDiffGAN model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (batch_size, channels, height, width).

        Returns
        -------
        torch.Tensor
            Output tensor with shape (batch_size, channels, height, width).
        """
        with torch.no_grad():
            x_lat = (
                self.ae.encode(x).latent_dist.mode().detach()
                * self.ae.config.scaling_factor
            )

        x = self.diffusion.sample(self.generator, x_lat, x_lat.shape)

        with torch.no_grad():
            x_out = self.ae.decode(x / self.ae.config.scaling_factor).sample
        x_out = torch.clamp(x_out, -1, 1)
        return x_out

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> dict[str, torch.Tensor]:
        """Training step for the SupResDiffGAN model.

        Parameters
        ----------
        batch : tuple of torch.Tensor
            Batch of data containing low-resolution and high-resolution images.
        batch_idx : int
            Index of the batch.

        Returns
        -------
        dict
            Dictionary containing the generator and discriminator losses.
        """
        lr_img, hr_img = batch["lr"], batch["hr"]

        # Get optimizers
        optimizer_g, optimizer_d = self.optimizers()

        # Going into the latent space
        with torch.no_grad():
            lr_lat = (
                self.ae.encode(lr_img).latent_dist.mode().detach()
                * self.ae.config.scaling_factor
            )
            x0_lat = (
                self.ae.encode(hr_img).latent_dist.mode().detach()
                * self.ae.config.scaling_factor
            )

        # Forward diffusion process
        timesteps = torch.randint(
            0,
            self.diffusion.timesteps,
            (x0_lat.shape[0],),
            device=x0_lat.device,
            dtype=torch.long,
        )

        x_t = self.diffusion.forward(x0_lat, timesteps)

        # Generating new image
        alfa_bars = self.diffusion.alpha_bars_torch.to(timesteps.device)[timesteps]
        x_gen_0 = self.generator(lr_lat, x_t, alfa_bars)

        # Calculating noised images for discriminator with EMA calculated noise step
        s_tensor = torch.tensor(self.s, device=x0_lat.device, dtype=torch.long)
        s_tensor = s_tensor.expand(x0_lat.shape[0])

        x_s = self.diffusion.forward(x0_lat, s_tensor)
        x_gen_s = self.diffusion.forward(x_gen_0, s_tensor)

        # Going back to pixel space
        with torch.no_grad():
            sr_img = self.ae.decode(x_gen_0 / self.ae.config.scaling_factor).sample
            sr_img = torch.clamp(sr_img, -1, 1)

            hr_s_img = self.ae.decode(x_s / self.ae.config.scaling_factor).sample
            hr_s_img = torch.clamp(hr_s_img, -1, 1)
            sr_s_img = self.ae.decode(x_gen_s / self.ae.config.scaling_factor).sample
            sr_s_img = torch.clamp(sr_s_img, -1, 1)

        if batch_idx % 2 == 0:
            # Generator training
            self.toggle_optimizer(optimizer_g)
            optimizer_g.zero_grad()
            g_loss = self.generator_loss(
                x0_lat,
                x_gen_0,
                hr_img,
                sr_img,
                hr_s_img,
                sr_s_img,
            )
            self.manual_backward(g_loss)
            optimizer_g.step()
            self.untoggle_optimizer(optimizer_g)
            self.log(
                "train/g_loss",
                g_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )
            return {"g_loss": g_loss}

        else:
            # Discriminator training
            self.toggle_optimizer(optimizer_d)
            optimizer_d.zero_grad()
            d_loss = self.discriminator_loss(hr_s_img, sr_s_img.detach())
            self.manual_backward(d_loss)
            optimizer_d.step()
            self.untoggle_optimizer(optimizer_d)
            self.log(
                "train/d_loss",
                d_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )
            return {"d_loss": d_loss}

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Validation step for the SupResDiffGAN model.

        Parameters
        ----------
        batch : tuple of torch.Tensor
            Batch of data containing low-resolution and high-resolution images.
        batch_idx : int
            Index of the batch.
        """
        lr_img, hr_img = batch["lr"], batch["hr"]
        lr_img, hr_img = lr_img.to(self.device), hr_img.to(self.device)
        sr_img = self(lr_img)
        padding_info = {}
        padding_info["lr"] = batch["padding_data_lr"]
        padding_info["hr"] = batch["padding_data_hr"]

        # Plot HR, LR, and SR images
        if batch_idx == 0:
            title = f"Epoch {self.current_epoch}"
            img_array = self.plot_images(hr_img, lr_img, sr_img, padding_info, title)
            self.logger.experiment.log(
                {
                    f"Validation epoch: {self.current_epoch}": [
                        wandb.Image(img_array, caption=f"Epoch {self.current_epoch}")
                    ]
                }
            )

        # Compute and log metrics
        metrics = {"PSNR": [], "SSIM": [], "MSE": []}

        for i in range(lr_img.shape[0]):
            hr_img_np = hr_img[i].detach().cpu().numpy().transpose(1, 2, 0)
            sr_img_np = sr_img[i].detach().cpu().numpy().transpose(1, 2, 0)

            hr_img_np = (hr_img_np + 1) / 2
            sr_img_np = (sr_img_np + 1) / 2

            hr_img_np = hr_img_np[
                : padding_info["hr"][i][1], : padding_info["hr"][i][0], :
            ]
            sr_img_np = sr_img_np[
                : padding_info["hr"][i][1], : padding_info["hr"][i][0], :
            ]

            psnr = peak_signal_noise_ratio(hr_img_np, sr_img_np, data_range=1.0)
            ssim = structural_similarity(
                hr_img_np, sr_img_np, channel_axis=-1, data_range=1.0
            )
            mse = np.mean((hr_img_np - sr_img_np) ** 2)

            metrics["PSNR"].append(psnr)
            metrics["SSIM"].append(ssim)
            metrics["MSE"].append(mse)

        lpips = self.lpips(hr_img, sr_img).cpu().item()

        self.log(
            "val/PSNR",
            np.mean(metrics["PSNR"]),
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "val/SSIM",
            np.mean(metrics["SSIM"]),
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "val/MSE",
            np.mean(metrics["MSE"]),
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "val/LPIPS",
            lpips,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> dict:
        """Test step for the SupResDiffGAN model, for model evaluation.

        Parameters
        ----------
        batch : tuple of torch.Tensor
            Batch of data containing low-resolution and high-resolution images.
        batch_idx : int
            Index of the batch.

        Returns
        -------
        dict
            Dictionary containing the metrics for the batch.
        """
        lr_img, hr_img = batch["lr"], batch["hr"]
        lr_img, hr_img = lr_img.to(self.device), hr_img.to(self.device)
        padding_info = {}
        padding_info["lr"] = batch["padding_data_lr"]
        padding_info["hr"] = batch["padding_data_hr"]

        start_time = time.perf_counter()
        sr_img = self(lr_img)
        elapsed_time = time.perf_counter() - start_time

        # Plot HR, LR, and SR images for the first batch
        if batch_idx == 0:
            img_array = self.plot_images(
                hr_img,
                lr_img,
                sr_img,
                padding_info,
                title=f"Test Images: timesteps: {self.diffusion.timesteps}, posterior: {self.diffusion.posterior_type}",
            )
            self.logger.experiment.log(
                {f"Test images": [wandb.Image(img_array, caption=f"Test Images")]}
            )

        # Compute metrics
        metrics = {"PSNR": [], "SSIM": [], "MSE": []}

        for i in range(lr_img.shape[0]):
            hr_img_np = hr_img[i].detach().cpu().numpy().transpose(1, 2, 0)
            sr_img_np = sr_img[i].detach().cpu().numpy().transpose(1, 2, 0)

            hr_img_np = (hr_img_np + 1) / 2
            sr_img_np = (sr_img_np + 1) / 2

            hr_img_np = hr_img_np[
                :, : padding_info["hr"][i][1], : padding_info["hr"][i][0]
            ]
            sr_img_np = sr_img_np[
                :, : padding_info["hr"][i][1], : padding_info["hr"][i][0]
            ]

            psnr = peak_signal_noise_ratio(hr_img_np, sr_img_np, data_range=1.0)
            ssim = structural_similarity(
                hr_img_np, sr_img_np, channel_axis=-1, data_range=1.0
            )
            mse = np.mean((hr_img_np - sr_img_np) ** 2)

            metrics["PSNR"].append(psnr)
            metrics["SSIM"].append(ssim)
            metrics["MSE"].append(mse)

        lpips = self.lpips(hr_img, sr_img).cpu().item()

        result = {
            "PSNR": np.mean(metrics["PSNR"]),
            "SSIM": np.mean(metrics["SSIM"]),
            "MSE": np.mean(metrics["MSE"]),
            "LPIPS": lpips,
            "time": elapsed_time,
        }

        self.test_step_outputs.append(result)

        return result

    def on_test_epoch_end(self) -> None:
        """Aggregate the metrics for all batches at the end of the test epoch."""
        avg_psnr = np.mean([x["PSNR"] for x in self.test_step_outputs])
        avg_ssim = np.mean([x["SSIM"] for x in self.test_step_outputs])
        avg_mse = np.mean([x["MSE"] for x in self.test_step_outputs])
        avg_lpips = np.mean([x["LPIPS"] for x in self.test_step_outputs])
        avg_time = np.mean([x["time"] for x in self.test_step_outputs])

        self.log(
            "test/PSNR",
            avg_psnr,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "test/SSIM",
            avg_ssim,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "test/MSE",
            avg_mse,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "test/LPIPS",
            avg_lpips,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "test/time",
            avg_time,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        # Clear test_step_outputs
        self.test_step_outputs.clear()

    def discriminator_loss(
        self,
        hr_s_img: torch.Tensor,
        sr_s_img: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the discriminator loss.

        The input to the discriminator is a concatenation of two images along the channel dimension.
        The tensors are concatenated in random order to prevent the discriminator from learning the pattern.
        The concatenated tensor has shape (N, 2*C, H, W), where N is the batch size, 2*C is the number
        of channels (since two images are concatenated), H is the height, and W is the width.

        Parameters
        ----------
        hr_s_img : torch.Tensor
            Tensor representing the high-resolution image in noise step s in pixel space.
            Shape: (N, C, H, W), where N is the batch size, C is the number of channels,
            H is the height, and W is the width.
        sr_s_img : torch.Tensor
            Tensor representing the super-resolution generated image in noise step s in pixel space.
            Shape: (N, C, H, W), where N is the batch size, C is the number of channels,
            H is the height, and W is the width.

        Returns
        -------
        torch.Tensor
            Discriminator loss.
        """
        # x = (real,fake), y = 0
        # x = (fake,real), y = 1
        y = torch.randint(0, 2, (hr_s_img.shape[0],), device=hr_s_img.device)
        y = y.view(-1, 1, 1, 1)
        y_expanded = y.expand(
            -1, hr_s_img.shape[1], hr_s_img.shape[2], hr_s_img.shape[3]
        )
        first = torch.where(y_expanded == 0, hr_s_img, sr_s_img)
        second = torch.where(y_expanded == 0, sr_s_img, hr_s_img)

        x = torch.cat([first, second], dim=1)

        prediction = self.discriminator(x)
        d_loss = self.adversarial_loss(prediction, y.float())

        self.calculate_ema_noise_step(prediction, y.float())

        return d_loss

    def generator_loss(
        self,
        x0: torch.Tensor,
        x_gen: torch.Tensor,
        hr_img: torch.Tensor,
        sr_img: torch.Tensor,
        hr_s_img: torch.Tensor,
        sr_s_img: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the generator loss.

        The generator loss consists of three components:
        - content loss: MSE loss between the generated image and the original high-resolution image,
        - perceptual loss: VGG perceptual loss between the super-resolution generated image and
                            the original high-resolution image,
        - adversarial loss: BCE loss between the discriminator prediction and the target label.

        Parameters
        ----------
        x0 : torch.Tensor
            Tensor representing the original high-resolution image in latent space.
            Shape: (N, C, H, W), where N is the batch size, C is the number of channels,
            H is the height, and W is the width.
        x_gen : torch.Tensor
            Tensor representing the generated image in latent space.
            Shape: (N, C, H, W), where N is the batch size, C is the number of channels,
            H is the height, and W is the width.
        hr_img : torch.Tensor
            Tensor representing the original high-resolution image in pixel space.
            Shape: (N, C, H, W), where N is the batch size, C is the number of channels,
            H is the height, and W is the width.
        sr_img : torch.Tensor
            Tensor representing the super-resolution generated image in pixel space.
            Shape: (N, C, H, W), where N is the batch size, C is the number of channels,
            H is the height, and W is the width.
        hr_s_img : torch.Tensor
            Tensor representing the high-resolution image in noise step s in pixel space.
        sr_s_img : torch.Tensor
            Tensor representing the super-resolution generated image in noise step s in pixel space.

        Returns
        -------
        torch.Tensor
            Generator loss.
        """
        # x = (real,fake), y = 1
        # x = (fake,real), y = 0
        content_loss = self.content_loss(x_gen, x0)

        if self.vgg_loss is not None:
            perceptual_loss = self.vgg_loss(sr_img, hr_img)
        else:
            perceptual_loss = 0

        y = torch.randint(0, 2, (hr_s_img.shape[0],), device=hr_s_img.device)
        y = y.view(-1, 1, 1, 1)
        y_expanded = y.expand(
            -1, hr_s_img.shape[1], hr_s_img.shape[2], hr_s_img.shape[3]
        )
        first = torch.where(y_expanded == 0, sr_s_img, hr_s_img)
        second = torch.where(y_expanded == 0, hr_s_img, sr_s_img)

        x = torch.cat([first, second], dim=1)

        prediction = self.discriminator(x)
        adversarial_loss = self.adversarial_loss(prediction, y.float())

        y_reversed = 1 - y.float()
        self.calculate_ema_noise_step(prediction, y_reversed)

        g_loss = (
            content_loss
            + self.alfa_perceptual * perceptual_loss
            + self.alfa_adv * adversarial_loss
        )

        self.log(
            "train/g_content_loss",
            content_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "train/g_adv_loss",
            adversarial_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        if self.vgg_loss is not None:
            self.log(
                "train/g_perceptual_loss",
                perceptual_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )
        return g_loss

    def configure_optimizers(
        self,
    ) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
        """Configure optimizers for the generator and discriminator.

        Returns
        -------
        tuple
            Tuple containing the generator and discriminator optimizers.
        """
        # ToDo: Consider use of learning rate scheduler (remember to step it in training loop)
        opt_g = torch.optim.Adam(
            self.generator.parameters(), lr=self.lr, betas=self.betas
        )
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.lr, betas=self.betas
        )
        return [opt_g, opt_d]

    def plot_images(
        self,
        hr_img: torch.Tensor,
        lr_img: torch.Tensor,
        sr_img: torch.Tensor,
        padding_info: dict,
        title: str,
    ) -> np.ndarray:
        """Plotting results method.

        Plots 5 random triples of high-resolution (HR), low-resolution (LR),
        and super-resolution (SR) images, as form of validation. Returns the
        array representing the plotted images.

        Parameters
        ----------
        hr_img : torch.Tensor
            Tensor representing the high-resolution images.
        lr_img : torch.Tensor
            Tensor representing the low-resolution images.
        sr_img : torch.Tensor
            Tensor representing the super-resolution generated images.
        title : str
            Title for the plot.
        padding_info : dict
            Dictionary containing padding information for the images.


        Returns
        -------
        np.ndarray
            Array representing the plotted images.
        """
        fig, axs = plt.subplots(3, 5, figsize=(10, 4))
        for i in range(5):
            num = np.random.randint(0, lr_img.shape[0])

            sr_img_plot = torch.clip(sr_img[num], -1, 1)
            sr_img_plot = sr_img_plot.detach().cpu().numpy().transpose(1, 2, 0)
            sr_img_plot = (sr_img_plot + 1) / 2  # Normalize to [0, 1]
            sr_img_plot = sr_img_plot[
                : padding_info["hr"][num][1], : padding_info["hr"][num][0], :
            ]

            hr_img_true = hr_img[num].detach().cpu().numpy().transpose(1, 2, 0)
            hr_img_true = (hr_img_true + 1) / 2  # Normalize to [0, 1]
            hr_img_true = hr_img_true[
                : padding_info["hr"][num][1], : padding_info["hr"][num][0], :
            ]

            lr_img_true = lr_img[num].detach().cpu().numpy().transpose(1, 2, 0)
            lr_img_true = (lr_img_true + 1) / 2  # Normalize to [0, 1]
            lr_img_true = lr_img_true[
                : padding_info["lr"][num][1], : padding_info["lr"][num][0], :
            ]

            axs[0, i].imshow(hr_img_true)
            axs[0, i].set_title("Ground Truth HR image")
            axs[0, i].set_xticks([])
            axs[0, i].set_yticks([])
            axs[1, i].imshow(lr_img_true)
            axs[1, i].set_title("Low resolution image")
            axs[1, i].set_xticks([])
            axs[1, i].set_yticks([])
            axs[2, i].imshow(sr_img_plot)
            axs[2, i].set_title("Predicted SR image")
            axs[2, i].set_xticks([])
            axs[2, i].set_yticks([])

        plt.suptitle(f"{title}")
        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)

        return img_array

    def calculate_ema_noise_step(self, pred: torch.Tensor, y: torch.Tensor) -> None:
        """
        Calculate the Exponential Moving Average (EMA) noise step.

        This function calculates the accuracy of the predictions compared to the ground truth,
        updates the EMA of the accuracy, and adjusts the noise scale based on the updated EMA.

        Parameters
        ----------
        pred : torch.Tensor
            The predicted tensor.
        y : torch.Tensor
            The ground truth tensor.
        """
        pred_binary = (pred > 0.5).float()
        acc = (pred_binary == y).float().mean().cpu().item()
        self.ema_mean = acc * (1 - self.ema_weight) + self.ema_mean * self.ema_weight

        self.s = int(
            torch.clamp(
                torch.tensor((self.ema_mean - 0.5) * 2 * self.diffusion.timesteps),
                0,
                self.diffusion.timesteps - 1,
            ).item()
        )
        self.log(
            "train/ema_noise_step",
            self.s,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "train/ema_accuracy",
            acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
