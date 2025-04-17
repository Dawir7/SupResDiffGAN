import hydra
import torch
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

from scripts.data_loader import train_val_test_loader
from scripts.exceptions import (
    EvaluateFreshInitializedModelException,
    UnknownModeException,
)
from scripts.model_config import model_selection
from scripts.utilis import model_path


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg) -> None:
    """Main function to train or test the model based on the provided configuration.

    This function initializes the model, data loaders, logger, and trainer,
    and performs training, testing, or both based on the specified mode.

    Parameters
    ----------
    cfg : OmegaConf
        Configuration object containing all settings for model training and testing.

    Returns
    -------
    None

    Raises
    ------
    EvaluateFreshInitializedModelException
        If no pre-trained model is specified in the configuration when in "test" mode.
    UnknownModeException
        If an unsupported mode is specified in the configuration.
    """

    final_model_path = model_path(cfg)
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    logger = WandbLogger(
        project=cfg.wandb.project,
        name=final_model_path.split("/")[-1],
        save_dir="logs",
        config=config_dict,
        entity=cfg.wandb.entity,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_gpus = torch.cuda.device_count()

    model = model_selection(cfg=cfg, device=device)
    train_loader, val_loader, test_loader = train_val_test_loader(cfg=cfg)

    checkpoint_callback = ModelCheckpoint(
        monitor=cfg.checkpoint.monitor,
        dirpath=cfg.checkpoint.dirpath,
        filename=cfg.model.name + "-{epoch:02d}-{train_loss:.2f}",
        save_top_k=cfg.checkpoint.save_top_k,
        mode=cfg.checkpoint.mode,
    )

    model = model.to(device)

    trainer = Trainer(
        max_epochs=cfg.trainer.max_epochs,
        max_steps=cfg.trainer.max_steps,
        devices=num_gpus,
        callbacks=[checkpoint_callback],
        logger=logger,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        limit_val_batches=cfg.trainer.limit_val_batches,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        strategy=DDPStrategy(find_unused_parameters=True),
    )

    if cfg.mode == "train":
        trainer.fit(model, train_loader, val_loader)
        torch.save(model.state_dict(), f"{final_model_path}.pth")

    elif cfg.mode == "train-test":
        trainer.fit(model, train_loader, val_loader)
        torch.save(model.state_dict(), f"{final_model_path}.pth")
        model = adjust_model_for_testing(cfg, model)
        trainer.test(model, test_loader)

    elif cfg.mode == "test":
        if cfg.model.load_model is None:
            raise EvaluateFreshInitializedModelException()
        model = adjust_model_for_testing(cfg, model)
        trainer.test(model, test_loader)

    else:
        raise UnknownModeException()


def adjust_model_for_testing(cfg, model) -> object:
    """Adjust the model configuration for testing.

    This function sets the validation timesteps and posterior type
    for the diffusion process based on the provided configuration.

    Parameters
    ----------
    cfg : object
        The configuration object containing the validation settings.
    model : object
        The model object to be adjusted.

    Returns
    -------
    object
        The adjusted model object.
    """
    # List of supported models that may require adjustments
    models_with_diffusion = {
        "SupResDiffGAN_v3",
        "SR3",
        "SupResDiffGAN_no_perceptual",
        "I2SB",
    }

    # Check if the model requires adjustments of diffusion parameters
    if cfg.model.name in models_with_diffusion:
        if cfg.diffusion.validation_timesteps is not None:
            model.diffusion.set_timesteps(cfg.diffusion.validation_timesteps)

        if cfg.diffusion.validation_posterior_type is not None:
            model.diffusion.set_posterior_type(cfg.diffusion.validation_posterior_type)

    return model


if __name__ == "__main__":
    main()
