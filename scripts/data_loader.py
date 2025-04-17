import os
from pathlib import Path

import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import numpy as np
import math


class PairedImagesDataset(Dataset):
    """A PyTorch Dataset for loading paired images from directories.

    This Dataset loads paired images (low and high resolution) from directories.
    It applies a transform to the images if provided.

    Parameters
    ----------
    lr_dir : Path
        The directory containing the low resolution images.
    hr_dir : Path
        The directory containing the high resolution images.
    transform_hr : callable, optional
        An optional transform to apply to the images.
    transform_lr : callable, optional
        An optional transform to apply to the images.
    """

    def __init__(
        self, lr_dir: Path, hr_dir: Path, transform_hr=None, transform_lr=None
    ):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.transform_hr = transform_hr
        self.transform_lr = transform_lr

        self.file_pairs = [
            filename for filename in sorted(os.listdir(os.path.join(lr_dir)))
        ]

    def __len__(self) -> int:
        """Return the number of images in the dataset.

        Returns
        -------
        int
            The number of images in the dataset.
        """
        return len(self.file_pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the low and high resolution images at the given index.

        Parameters
        ----------
        idx : int
            The index of the images to return.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            The low and high resolution images at the given index.
        """
        # Load the low and high resolution images
        filename = self.file_pairs[idx]

        # Load the low and high resolution images
        img_lr = Image.open(self.lr_dir / filename)
        img_hr = Image.open(self.hr_dir / filename)

        if img_lr.mode != "RGB":
            img_lr = img_lr.convert("RGB")
        if img_hr.mode != "RGB":
            img_hr = img_hr.convert("RGB")

        # Apply the transformations
        if self.transform_hr:
            img_hr = self.transform_hr(img_hr)
        if self.transform_lr:
            img_lr = self.transform_lr(img_lr)

        return img_lr, img_hr


class PairedImagesDataModule(pl.LightningDataModule):
    """
    A PyTorch Lightning DataModule for loading paired images from directories.

    This DataModule loads paired images (low and high resolution) from directories
    for training, validation, and testing.
    It applies a transform to the images and provides DataLoaders for the datasets.

    Parameters
    ----------
    cfg : DictConfig
        The configuration object containing model parameters and settings.
    lr_dir : Path
        The directory containing the low resolution images.
    hr_dir : Path
        The directory containing the high resolution images.
    batch_size : int, optional
        The batch size for the DataLoaders, by default 32.
    scale : int, optional
        The scaling factor for resizing images. If None, the scale is taken from the configuration, by default None.
    """

    def __init__(
        self, cfg, lr_dir: Path, hr_dir: Path, batch_size: int = 32, scale: int = None
    ):
        super().__init__()
        self.cfg = cfg
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.batch_size = batch_size

        if scale is None:
            self.scale = self.cfg.dataset.scale
        else:
            self.scale = scale

        transform_list = [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        if self.cfg.dataset.resize:
            self.transform_hr = transforms.Compose(transform_list)

            resize = transforms.Lambda(
                lambda img: self.resize_by_scale(img, self.scale)
            )
            transfer_list_lr = [resize] + transform_list
            self.transform_lr = transforms.Compose(transfer_list_lr)
        else:
            self.transform_hr = transforms.Compose(transform_list)
            self.transform_lr = transforms.Compose(transform_list)

    def resize_by_scale(self, img: Image, scale: float) -> Image:
        """Resize an image by a given scaling factor.

        Parameters
        ----------
        img : PIL.Image
            The image to be resized.
        scale : float
            The scaling factor to resize the image.

        Returns
        -------
        PIL.Image
            The resized image.
        """
        new_size = (int(img.height * scale), int(img.width * scale))
        return transforms.functional.resize(
            img,
            new_size,
            interpolation=transforms.InterpolationMode.BILINEAR,
        )

    def setup(self, stage=None, val_split=0.3):
        """Prepare the datasets for the given stage.

        This method is called by PyTorch Lightning during the setup stage.
        It prepares the datasets for the given stage (either 'fit', 'test', or None).

        Parameters
        ----------
        stage : str, optional
            The stage for which to prepare the datasets. If 'fit', the method prepares
            the training and validation datasets.
            If 'test', it prepares the test dataset. If None, it prepares all datasets.
        """
        if stage == "fit" or stage is None:
            self.paired_images_train = PairedImagesDataset(
                self.lr_dir / "train",
                self.hr_dir / "train",
                transform_hr=self.transform_hr,
                transform_lr=self.transform_lr,
            )
            self.paired_images_val = PairedImagesDataset(
                self.lr_dir / "val",
                self.hr_dir / "val",
                transform_hr=self.transform_hr,
                transform_lr=self.transform_lr,
            )

        if stage == "test" or stage is None:
            self.paired_images_test = PairedImagesDataset(
                self.lr_dir / "test",
                self.hr_dir / "test",
                transform_hr=self.transform_hr,
                transform_lr=self.transform_lr,
            )

        elif stage == "train_test":
            self.paired_images_train = PairedImagesDataset(
                Path(str(self.lr_dir).replace("|train_test|", "train")),
                Path(str(self.hr_dir).replace("|train_test|", "train")),
                transform_hr=self.transform_hr,
                transform_lr=self.transform_lr,
            )

            val_size = int(len(self.paired_images_train) * val_split)
            train_size = len(self.paired_images_train) - val_size

            self.paired_images_train, self.paired_images_val = random_split(
                self.paired_images_train, [train_size, val_size]
            )

            self.paired_images_test = PairedImagesDataset(
                Path(str(self.lr_dir).replace("|train_test|", "test")),
                Path(str(self.hr_dir).replace("|train_test|", "test")),
                transform_hr=self.transform_hr,
                transform_lr=self.transform_lr,
            )

        elif stage == "train_val_test":
            self.paired_images_train = PairedImagesDataset(
                Path(str(self.lr_dir).replace("|train_val_test|", "train")),
                Path(str(self.hr_dir).replace("|train_val_test|", "train")),
                transform_hr=self.transform_hr,
                transform_lr=self.transform_lr,
            )

            self.paired_images_val = PairedImagesDataset(
                Path(str(self.lr_dir).replace("|train_val_test|", "val")),
                Path(str(self.hr_dir).replace("|train_val_test|", "val")),
                transform_hr=self.transform_hr,
                transform_lr=self.transform_lr,
            )

            self.paired_images_test = PairedImagesDataset(
                Path(str(self.lr_dir).replace("|train_val_test|", "test")),
                Path(str(self.hr_dir).replace("|train_val_test|", "test")),
                transform_hr=self.transform_hr,
                transform_lr=self.transform_lr,
            )

        elif stage == "only_test":
            self.paired_images_test = PairedImagesDataset(
                self.lr_dir,
                self.hr_dir,
                transform_hr=self.transform_hr,
                transform_lr=self.transform_lr,
            )

    def train_dataloader(self):
        """Return a DataLoader for the training dataset.

        Returns
        -------
        DataLoader
            A DataLoader for the training dataset.
        """
        return DataLoader(
            self.paired_images_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=min(2, os.cpu_count()),
            collate_fn=self.collate_cropping_fn,
        )

    def val_dataloader(self):
        """Return a DataLoader for the validation dataset.

        Returns
        -------
        DataLoader
            A DataLoader for the validation dataset.
        """
        return DataLoader(
            self.paired_images_val,
            batch_size=self.batch_size,
            num_workers=min(2, os.cpu_count()),
            collate_fn=self.collate_padding_fn,
        )

    def test_dataloader(self):
        """Return a DataLoader for the test dataset.

        Returns
        -------
        DataLoader
            A DataLoader for the test dataset.
        """
        return DataLoader(
            self.paired_images_test,
            batch_size=self.batch_size,
            num_workers=min(2, os.cpu_count()),
            collate_fn=self.collate_padding_fn,
        )

    def collate_cropping_fn(self, batch):
        """
        Collates a batch of images by cropping them to the minimum height and width
        that are multiples of `k` (64). If resizing is enabled in the configuration,
        the images are randomly cropped to the minimum size. Otherwise, the images
        are cropped from the top-left corner.

        Parameters
        ----------
        batch : list of tuples
            A batch of tuples where each tuple contains a low-resolution image (img_lr)
            and a high-resolution image (img_hr).

        Returns
        -------
        dict
            A dictionary containing:
            - "lr" (torch.Tensor): A tensor of cropped low-resolution images.
            - "hr" (torch.Tensor): A tensor of cropped high-resolution images.
            - "original_size" (list of tuples): A list of original sizes of the low-resolution images.
        """
        k = 64
        if self.cfg.dataset.resize:
            min_height = min(img.size(1) for img, _ in batch) // k * k
            min_width = min(img.size(2) for img, _ in batch) // k * k

            cropped_lr = []
            cropped_hr = []
            original_size = []
            for img_lr, img_hr in batch:
                start_height = np.random.randint(0, img_lr.size(1) - min_height + 1)
                start_width = np.random.randint(0, img_lr.size(2) - min_width + 1)
                end_height = min(start_height + min_height, img_lr.size(1))
                end_width = min(start_width + min_width, img_lr.size(2))
                cropped_lr.append(
                    img_lr[:, start_height:end_height, start_width:end_width]
                )
                cropped_hr.append(
                    img_hr[:, start_height:end_height, start_width:end_width]
                )
                original_size.append((img_lr.size(2), img_lr.size(1)))
        else:
            min_height_lr = min(img.size(1) for img, _ in batch) // k * k
            min_width_lr = min(img.size(2) for img, _ in batch) // k * k
            min_height_hr = min_height_lr * self.cfg.dataset.scale
            min_width_hr = min_width_lr * self.cfg.dataset.scale

            cropped_lr = []
            cropped_hr = []
            original_size = []
            for img_lr, img_hr in batch:

                cropped_lr.append(img_lr[:, 0:min_height_lr, 0:min_width_lr])
                cropped_hr.append(img_hr[:, 0:min_height_hr, 0:min_width_hr])
                original_size.append((img_lr.size(2), img_lr.size(1)))

        res = {
            "lr": torch.stack(cropped_lr),
            "hr": torch.stack(cropped_hr),
            "original_size": original_size,
        }
        return res

    def collate_padding_fn(self, batch):
        """
        Collates a batch of images by padding them to the maximum height and width
        that are multiples of `k` (64). If resizing is enabled in the configuration,
        the images are padded to the maximum size. Otherwise, the images are padded
        from the top-left corner.

        Parameters
        ----------
        batch : list of tuples
            A batch of tuples where each tuple contains a low-resolution image (img_lr)
            and a high-resolution image (img_hr).

        Returns
        -------
        dict
            A dictionary containing:
            - "lr" (torch.Tensor): A tensor of padded low-resolution images.
            - "hr" (torch.Tensor): A tensor of padded high-resolution images.
            - "padding_data_lr" (list of tuples): A list of original sizes of the low-resolution images.
            - "padding_data_hr" (list of tuples): A list of original sizes of the high-resolution images.
        """
        k = 64

        if self.cfg.dataset.resize:
            max_height_lr = math.ceil(max(img.size(1) for img, _ in batch) / k) * k
            max_width_lr = math.ceil(max(img.size(2) for img, _ in batch) / k) * k
            max_height_hr, max_width_hr = max_height_lr, max_width_lr

        else:
            max_height_lr = math.ceil(max(img.size(1) for img, _ in batch) / k) * k
            max_width_lr = math.ceil(max(img.size(2) for img, _ in batch) / k) * k
            max_height_hr = max_height_lr * self.cfg.dataset.scale
            max_width_hr = max_width_lr * self.cfg.dataset.scale

        padded_lr = []
        padded_hr = []
        padding_data_hr = []
        padding_data_lr = []
        for img_lr, img_hr in batch:

            pad_lr = torch.nn.functional.pad(
                img_lr,
                (0, max_width_lr - img_lr.size(2), 0, max_height_lr - img_lr.size(1)),
            )
            pad_hr = torch.nn.functional.pad(
                img_hr,
                (0, max_width_hr - img_hr.size(2), 0, max_height_hr - img_hr.size(1)),
            )

            padded_lr.append(pad_lr)
            padded_hr.append(pad_hr)
            padding_data_lr.append((img_lr.size(2), img_lr.size(1)))
            padding_data_hr.append((img_hr.size(2), img_hr.size(1)))
        res = {
            "lr": torch.stack(padded_lr),
            "hr": torch.stack(padded_hr),
            "padding_data_lr": padding_data_lr,
            "padding_data_hr": padding_data_hr,
        }
        return res


def train_val_test_loader(cfg):
    """Create and return DataLoaders for training, validation, and testing datasets.

    Parameters
    ----------
    cfg : DictConfig
        The configuration object containing model parameters and settings.

    Returns
    -------
    tuple
        A tuple containing the training, validation and test DataLoaders.

    Raises
    ------
    ValueError
        If the specified dataset is not found in the configuration.
    """
    if cfg.dataset.name in ["div2k", "realsr_canon", "realsr_nikon"]:
        hr_dir = Path(f"data/{cfg.dataset.name}/|train_test|/X{cfg.dataset.scale}/HR")
        lr_dir = Path(f"data/{cfg.dataset.name}/|train_test|/X{cfg.dataset.scale}/LR")

        data_module = PairedImagesDataModule(
            cfg=cfg, lr_dir=lr_dir, hr_dir=hr_dir, batch_size=cfg.dataset.batch_size
        )
        data_module.setup(stage="train_test", val_split=0.3)

        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        test_loader = data_module.test_dataloader()

    elif cfg.dataset.name in ["celeb", "imagenet"]:
        hr_dir = Path(
            f"data/{cfg.dataset.name}/|train_val_test|/X{cfg.dataset.scale}/HR"
        )
        lr_dir = Path(
            f"data/{cfg.dataset.name}/|train_val_test|/X{cfg.dataset.scale}/LR"
        )

        data_module = PairedImagesDataModule(
            cfg=cfg, lr_dir=lr_dir, hr_dir=hr_dir, batch_size=cfg.dataset.batch_size
        )
        data_module.setup("train_val_test")

        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        test_loader = data_module.test_dataloader()

    elif cfg.dataset.name in ["Set14", "urban100"]:
        hr_dir = Path(f"data/{cfg.dataset.name}/X{cfg.dataset.scale}/HR")
        lr_dir = Path(f"data/{cfg.dataset.name}/X{cfg.dataset.scale}/LR")

        data_module = PairedImagesDataModule(
            cfg=cfg, lr_dir=lr_dir, hr_dir=hr_dir, batch_size=cfg.dataset.batch_size
        )
        data_module.setup("only_test")

        train_loader = None
        val_loader = None
        test_loader = data_module.test_dataloader()

    else:
        raise ValueError("Wrong Dataset name")

    return train_loader, val_loader, test_loader
