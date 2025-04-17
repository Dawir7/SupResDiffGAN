import time

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb
from matplotlib import pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.optim import Adam
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


class RealESRGAN(pl.LightningModule):
    def __init__(
        self,
        discriminator: nn.Module,
        generator: nn.Module,
        feature_extractor: nn.Module,
        learning_rate: float = 2e-4,
    ) -> None:
        """Initialize the RealESRGAN (Enhanced Super-Resolution Generative Adversarial Networks) model.

        Parameters
        ----------
        discriminator : nn.Module
            The discriminator network.
        generator : nn.Module
            The generator network.
        feature_extractor : nn.Module
            The feature extractor network.
        learning_rate : float
            Learning rate for the optimizers (default is 2e-4).
        """
        super(RealESRGAN, self).__init__()
        # Classes
        self.discriminator = discriminator
        self.generator = generator

        # Loss functions
        self.adversarial_loss = nn.BCEWithLogitsLoss().to(self.device)
        self.perceptual_loss = feature_extractor
        self.pixel_loss = nn.L1Loss().to(self.device)

        # Hyperparameters
        self.lr = learning_rate
        self.betas = (0.9, 0.999)
        self.warmup_batches = 500
        self.lambda_adv = 0.1

        # Other attributes
        self.automatic_optimization = False
        self.test_step_outputs = []
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(
            self.device
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the generator.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        return self.generator(x)

    def configure_optimizers(self) -> list[torch.optim.Optimizer]:
        """Configure optimizers for the generator and discriminator.

        Returns
        -------
        list of torch.optim.Optimizer
            List containing the optimizers for the generator and discriminator.
        """
        optimizer_g = Adam(self.generator.parameters(), lr=self.lr, betas=self.betas)
        optimizer_d = Adam(
            self.discriminator.parameters(), lr=self.lr, betas=self.betas
        )
        return [optimizer_g, optimizer_d]

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> dict:
        """Training step for ESRGAN.

        Parameters
        ----------
        batch : tuple[torch.Tensor, torch.Tensor]
            Batch of data containing low-resolution and high-resolution images.
        batch_idx : int
            Index of the batch.

        Returns
        -------
        dict
            Dictionary containing the loss and logs.
        """
        lr_images, hr_images = batch["lr"], batch["hr"]

        optimizer_g, optimizer_d = self.optimizers()

        if batch_idx % 2 == 0:
            # Train generator
            self.toggle_optimizer(optimizer_g)
            sr_images = self(lr_images)  # generator forward pass
            g_loss = self.generator_loss(sr_images, hr_images)

            # Backward pass and optimization for generator
            optimizer_g.zero_grad()
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
            # Train discriminator
            self.toggle_optimizer(optimizer_d)
            sr_images = self.generator(lr_images).detach()
            d_loss = self.discriminator_loss(sr_images, hr_images)

            # Backward pass and optimization for discriminator
            optimizer_d.zero_grad()
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
    ) -> dict:
        """Validation step for ESRGAN.

        Parameters
        ----------
        batch : tuple[torch.Tensor, torch.Tensor]
            Batch of data containing low-resolution and high-resolution images.
        batch_idx : int
            Index of the batch.

        Returns
        -------
        dict
            Dictionary containing the validation loss.
        """
        lr_img, hr_img = batch["lr"], batch["hr"]
        lr_img, hr_img = lr_img.to(self.device), hr_img.to(self.device)
        sr_img = self.generator(lr_img)

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

        # Compute metrics
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
        """Test step of the model, for model evaluation.

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
        sr_img = self.generator(lr_img)
        elapsed_time = time.perf_counter() - start_time

        # Plot HR, LR, and SR images for the first batch
        if batch_idx == 0:
            img_array = self.plot_images(
                hr_img, lr_img, sr_img, padding_info, title="Test Images"
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

    def generator_loss(
        self,
        sr_images: torch.Tensor,
        hr_images: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the generator loss.

        Parameters
        ----------
        sr_images : torch.Tensor
            Tensor representing the super-resolved images generated by the generator.
            Shape: (N, C, H, W), where N is the batch size, C is the number of channels,
            H is the height, and W is the width.
        hr_images : torch.Tensor
            Tensor representing the high-resolution ground truth images.
            Shape: (N, C, H, W), where N is the batch size, C is the number of channels,
            H is the height, and W is the width.

        Returns
        -------
        torch.Tensor
            Generator loss.
        """
        loss_pixel = self.pixel_loss(sr_images, hr_images)

        if self.global_step < self.warmup_batches:
            # Warm-up (pixel-wise loss only)
            return loss_pixel

        real_output = self.discriminator(hr_images).detach()
        fake_output = self.discriminator(sr_images)

        real = torch.ones_like(real_output, device=self.device, requires_grad=False)

        adversarial_loss = self.adversarial_loss(
            fake_output - real_output.mean(0, keepdim=True), real
        )

        perceptual_loss = self.perceptual_loss(sr_images, hr_images)

        g_loss = perceptual_loss + self.lambda_adv * adversarial_loss + loss_pixel

        return g_loss

    def discriminator_loss(
        self,
        sr_images: torch.Tensor,
        hr_images: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the discriminator loss.

        Parameters
        ----------
        sr_images : torch.Tensor
            Tensor representing the super-resolved images generated by the generator.
            Shape: (N, C, H, W), where N is the batch size, C is the number of channels,
            H is the height, and W is the width.
        hr_images : torch.Tensor
            Tensor representing the high-resolution ground truth images.
            Shape: (N, C, H, W), where N is the batch size, C is the number of channels,
            H is the height, and W is the width.

        Returns
        -------
        torch.Tensor
            Discriminator loss.
        """
        real_output = self.discriminator(hr_images)
        fake_output = self.discriminator(sr_images.detach())

        real = torch.ones_like(real_output, device=self.device, requires_grad=False)
        fake = torch.zeros_like(fake_output, device=self.device, requires_grad=False)

        real_loss = self.adversarial_loss(
            real_output - fake_output.mean(0, keepdim=True), real
        )
        fake_loss = self.adversarial_loss(
            fake_output - real_output.mean(0, keepdim=True), fake
        )

        d_loss = (real_loss + fake_loss) / 2

        return d_loss

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
