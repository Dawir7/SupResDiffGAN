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


# SRGAN
class SRGAN(pl.LightningModule):
    """SRGAN class for Super-Resolution Generative Adversarial Network.

    Parameters
    ----------
    discriminator : Discriminator
        The discriminator network.
    generator : Generator
        The generator network.
    vgg_loss : VGGLoss
        The VGG perceptual loss.
    learning_rate : float, optional
        Learning rate for the optimizers (default is 1e-4).
    """

    def __init__(
        self,
        discriminator: torch.nn.Module,
        generator: torch.nn.Module,
        vgg_loss: torch.nn.Module,
        learning_rate: float = 1e-4,
    ) -> None:
        super(SRGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.vgg_loss = vgg_loss

        self.content_loss = nn.MSELoss()
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(
            self.device
        )

        self.lr = learning_rate

        # Set automatic optimization to False
        self.automatic_optimization = False

        self.test_step_outputs = []

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
        optimizer_g = Adam(self.generator.parameters(), lr=self.lr)
        optimizer_d = Adam(self.discriminator.parameters(), lr=self.lr)
        return [optimizer_g, optimizer_d]

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> dict:
        """Training step for SRGAN.

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

        # Run at the start of training to evaluate initial metrics.
        # if batch_idx == 0 and self.current_epoch == 0:
        #     self.log_start_metrics(batch)

        # Get optimizers
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
        """Validation step for SRGAN.

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
        # g_loss = self.generator_loss(sr_img, hr_img)
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
        # self.log(
        #     "val/g_loss",
        #     g_loss,
        #     on_step=False,
        #     on_epoch=True,
        #     prog_bar=True,
        #     logger=True,
        #     sync_dist=True,
        # )

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

        # g_loss = self.generator_loss(sr_img, hr_img)

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
            # "g_loss": g_loss.cpu().item(),
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
        # avg_g_loss = np.mean([x["g_loss"] for x in self.test_step_outputs])
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
        # self.log(
        #     "test/g_loss",
        #     avg_g_loss,
        #     on_epoch=True,
        #     on_step=False,
        #     prog_bar=True,
        #     logger=True,
        #     sync_dist=True,
        # )
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
        self, sr_images: torch.Tensor, hr_images: torch.Tensor
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
        content_loss = self.content_loss(sr_images, hr_images)
        perceptual_loss = self.vgg_loss(sr_images, hr_images)
        fake_output = self.discriminator(sr_images)
        adversarial_loss = 1 - fake_output.mean()
        g_loss = content_loss + 1e-3 * perceptual_loss + 1e-3 * adversarial_loss
        return g_loss

    def discriminator_loss(
        self, sr_images: torch.Tensor, hr_images: torch.Tensor
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
        fake_output = self.discriminator(sr_images)
        d_loss = 1 - real_output.mean() + fake_output.mean()
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

    def log_start_metrics(self, batch: tuple[torch.Tensor, torch.Tensor]) -> None:
        """
        Computes and logs various evaluation metrics at the start of training for a given batch.

        This function is typically used to establish a baseline for model performance before training begins.
        It evaluates metrics like PSNR (Peak Signal-to-Noise Ratio), SSIM (Structural Similarity Index),
        MSE (Mean Squared Error), and LPIPS (Learned Perceptual Image Patch Similarity) on a batch of
        validation data.

        Args:
            batch (tuple): A tuple containing low-resolution (lr_img) and high-resolution (hr_img) images.
                        Each of these tensors has shape (batch_size, channels, height, width) and is
                        normalized to a range of [-1, 1].
        """
        with torch.no_grad():
            # Fetch a batch of validation data
            lr_img, hr_img = batch["lr"], batch["hr"]
            lr_img, hr_img = lr_img.to(self.device), hr_img.to(self.device)

            # Calculate metrics
            metrics = {"PSNR": [], "SSIM": [], "MSE": []}
            for i in range(hr_img.shape[0]):
                hr_img_np = hr_img[i].cpu().numpy().transpose(1, 2, 0)
                lr_img_np = lr_img[i].cpu().numpy().transpose(1, 2, 0)

                # Rescale images to [0, 1]
                hr_img_np = (hr_img_np + 1) / 2
                lr_img_np = (lr_img_np + 1) / 2

                # Compute metrics
                psnr = peak_signal_noise_ratio(hr_img_np, lr_img_np, data_range=1.0)
                ssim = structural_similarity(
                    hr_img_np, lr_img_np, channel_axis=-1, data_range=1.0
                )
                mse = np.mean((hr_img_np - lr_img_np) ** 2)

                metrics["PSNR"].append(psnr)
                metrics["SSIM"].append(ssim)
                metrics["MSE"].append(mse)

            # Compute LPIPS metric
            lpips = self.lpips(hr_img, lr_img.to(self.device)).cpu().item()

            # Log metrics at the start
            self.log(
                "start/PSNR",
                np.mean(metrics["PSNR"]),
                sync_dist=True,
                prog_bar=True,
                logger=True,
            )
            self.log(
                "start/SSIM",
                np.mean(metrics["SSIM"]),
                sync_dist=True,
                prog_bar=True,
                logger=True,
            )
            self.log(
                "start/MSE",
                np.mean(metrics["MSE"]),
                sync_dist=True,
                prog_bar=True,
                logger=True,
            )
            self.log("start/LPIPS", lpips, sync_dist=True, prog_bar=True, logger=True)
