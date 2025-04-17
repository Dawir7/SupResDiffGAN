import time

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


class SR3(pl.LightningModule):
    """
    A PyTorch Lightning module for training and validating a super-resolution
    model (SR3) with a diffusion process.

    Parameters
    ----------
    unet_model : UNet
        The U-Net model to be trained.
    diffusion : Diffusion
        The diffusion process to be used for generating and denoising images.
    lr : float, optional
        The learning rate for the optimizer. Default is 1e-4.
    """

    def __init__(
        self, unet_model: torch.nn.Module, diffusion: torch.nn.Module, lr: float = 1e-4
    ):
        super().__init__()
        self.unet = unet_model
        self.diffusion = diffusion

        self.lr = lr
        self.betas = (0.9, 0.999)

        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(
            self.device
        )

        self.test_step_outputs = []

    def forward(self, lr_img: torch.Tensor) -> torch.Tensor:
        """Forward pass of the SR3 model.

        This method generates a super-resolution image from a low-resolution input image
        using the diffusion process and the U-Net model.

        Parameters
        ----------
        lr_img : torch.Tensor
            A tensor representing the low-resolution input image. The tensor should have
            shape `(batch_size, channels, height, width)` and values normalized to the range [-1, 1].

        Returns
        -------
        torch.Tensor
            A tensor representing the super-resolution output image. The tensor will have
            the same shape as the input tensor and values normalized to the range [-1, 1].
        """
        return self.diffusion.sample(self.unet, lr_img, sample_size=lr_img.shape)

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Performs a single training step.

        Parameters
        ----------
        batch : tuple
            A tuple containing the input batch. It includes the cropped images and the original images.
        batch_idx : int
            The index of the current batch.

        Returns
        -------
        torch.Tensor
            The computed loss for the batch.
        """
        lr_img, x_0 = batch["lr"], batch["hr"]

        # Run at the start of training to evaluate initial metrics.
        if batch_idx == 0 and self.current_epoch == 0:
            self.log_start_metrics(batch)

        loss = self.diffusion.compute_loss(self.unet, x_0, lr_img)
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> dict:
        """Performs a validation step and visualizes predictions.

        This method generates images using the diffusion process and compares them with the ground
        truth images. It visualizes a subset of the generated and ground truth images.

        Parameters
        ----------
        batch : tuple[torch.Tensor, torch.Tensor]
            The input and target tensors for the batch.
        batch_idx : int
            The index of the batch.

        Returns
        -------
        dict
            A dictionary with the loss for the step.
        """
        lr_img, x_0 = batch["lr"], batch["hr"]
        lr_img, x_0 = lr_img.to(self.device), x_0.to(self.device)
        pred_x_0 = self.diffusion.sample(self.unet, lr_img, sample_size=lr_img.shape)

        padding_info = {}
        padding_info["lr"] = batch["padding_data_lr"]
        padding_info["hr"] = batch["padding_data_hr"]

        # Plot HR, LR, and SR images
        if batch_idx == 0:
            title = f"Epoch {self.current_epoch}"
            img_array = self.plot_images(x_0, lr_img, pred_x_0, padding_info, title)
            self.logger.experiment.log(
                {
                    f"Validation epoch: {self.current_epoch}": [
                        wandb.Image(img_array, caption=f"Epoch {self.current_epoch}")
                    ]
                }
            )

        metrics = {"PSNR": [], "SSIM": [], "MSE": []}
        for i in range(lr_img.shape[0]):
            x_0_np = x_0[i].detach().cpu().numpy().transpose(1, 2, 0)
            pred_x_0_np = pred_x_0[i].detach().cpu().numpy().transpose(1, 2, 0)

            x_0_np = (x_0_np + 1) / 2
            pred_x_0_np = (pred_x_0_np + 1) / 2

            x_0_np = x_0_np[: padding_info["hr"][i][1], : padding_info["hr"][i][0], :]
            pred_x_0_np = pred_x_0_np[
                : padding_info["hr"][i][1], : padding_info["hr"][i][0], :
            ]

            psnr = peak_signal_noise_ratio(x_0_np, pred_x_0_np, data_range=1.0)
            ssim = structural_similarity(
                x_0_np, pred_x_0_np, channel_axis=-1, data_range=1.0
            )
            mse = np.mean((x_0_np - pred_x_0_np) ** 2)

            metrics["PSNR"].append(psnr)
            metrics["SSIM"].append(ssim)
            metrics["MSE"].append(mse)

        lpips = self.lpips(x_0, pred_x_0.to(self.device)).cpu().item()

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
        lr_img, x_0 = batch["lr"], batch["hr"]
        lr_img, x_0 = lr_img.to(self.device), x_0.to(self.device)
        padding_info = {}
        padding_info["lr"] = batch["padding_data_lr"]
        padding_info["hr"] = batch["padding_data_hr"]

        start_time = time.perf_counter()
        pred_x_0 = self.diffusion.sample(self.unet, lr_img, sample_size=lr_img.shape)
        elapsed_time = time.perf_counter() - start_time

        # Plot HR, LR, and SR images for the first batch
        if batch_idx == 0:
            img_array = self.plot_images(
                x_0, lr_img, pred_x_0, padding_info, title="Test Images"
            )
            self.logger.experiment.log(
                {f"Test images": [wandb.Image(img_array, caption=f"Test Images")]}
            )

        # Compute metrics
        metrics = {"PSNR": [], "SSIM": [], "MSE": []}
        for i in range(lr_img.shape[0]):
            x_0_np = x_0[i].detach().cpu().numpy().transpose(1, 2, 0)
            pred_x_0_np = pred_x_0[i].detach().cpu().numpy().transpose(1, 2, 0)

            x_0_np = (x_0_np + 1) / 2
            pred_x_0_np = (pred_x_0_np + 1) / 2

            x_0_np = x_0_np[: padding_info["hr"][i][1], : padding_info["hr"][i][0], :]
            pred_x_0_np = pred_x_0_np[
                : padding_info["hr"][i][1], : padding_info["hr"][i][0], :
            ]

            psnr = peak_signal_noise_ratio(x_0_np, pred_x_0_np, data_range=1.0)
            ssim = structural_similarity(
                x_0_np, pred_x_0_np, channel_axis=-1, data_range=1.0
            )
            mse = np.mean((x_0_np - pred_x_0_np) ** 2)

            metrics["PSNR"].append(psnr)
            metrics["SSIM"].append(ssim)
            metrics["MSE"].append(mse)

        lpips = self.lpips(x_0, pred_x_0.to(self.device)).cpu().item()

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

    def configure_optimizers(self) -> dict:
        """Configures the optimizer for the model.

        Returns
        -------
        dict
            The configured optimizer.
        """
        optimizer = torch.optim.Adam(
            self.unet.parameters(), lr=self.lr, betas=self.betas
        )

        return optimizer

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
