from .model_config_imports import *


def model_selection(cfg, device):
    """Select and initialize the model based on the configuration.

    Parameters
    ----------
    cfg : DictConfig
        The configuration object containing model parameters and settings.
    device : torch.device
        The device on which the model will be loaded (e.g., 'cuda' or 'cpu').

    Returns
    -------
    torch.nn.Module
        The initialized model.

    Raises
    ------
    ValueError
        If the specified model is not found in the configuration.
    """

    if cfg.model.name == "SR3":
        unet = UNet_sr3(channels=cfg.unet)
        diffusion = Diffusion_sr3(
            timesteps=cfg.diffusion.timesteps,
            beta_type=cfg.diffusion.beta_type,
            posterior_type=cfg.diffusion.posterior_type,
        )

        if cfg.model.load_model is not None:
            model_path = cfg.model.load_model
            _, ext = os.path.splitext(model_path)
            if ext == ".pth":
                model = SR3(unet_model=unet, diffusion=diffusion, lr=cfg.model.lr)
                model.load_state_dict(torch.load(model_path, map_location=device))
            elif ext == ".ckpt":
                model = SR3.load_from_checkpoint(
                    model_path,
                    map_location=device,
                    unet_model=unet,
                    diffusion=diffusion,
                    lr=cfg.model.lr,
                )
            else:
                raise ValueError(f"Unsupported file extension: {ext}")
        else:
            model = SR3(unet_model=unet, diffusion=diffusion, lr=cfg.model.lr)

        return model

    elif cfg.model.name == "SRGAN":
        discriminator = Discriminator_srgan(
            in_channels=cfg.discriminator.in_channels,
            channels=cfg.discriminator.channels,
        )
        generator = Generator_srgan(
            in_channels=cfg.generator.in_channels,
            out_channels=cfg.generator.out_channels,
            scale_factor=cfg.generator.scale_factor,
            num_resblocks=cfg.generator.num_resblocks,
        )
        vgg_loss = VGGLoss_srgan(device)

        if cfg.model.load_model is not None:
            model_path = cfg.model.load_model
            _, ext = os.path.splitext(model_path)
            if ext == ".pth":
                model = SRGAN(
                    discriminator=discriminator,
                    generator=generator,
                    vgg_loss=vgg_loss,
                    learning_rate=cfg.model.lr,
                )
                model.load_state_dict(torch.load(model_path, map_location=device))
            elif ext == ".ckpt":
                model = SRGAN.load_from_checkpoint(
                    model_path,
                    map_location=device,
                    discriminator=discriminator,
                    generator=generator,
                    vgg_loss=vgg_loss,
                    learning_rate=cfg.model.lr,
                )
            else:
                raise ValueError(f"Unsupported file extension: {ext}")

        else:
            model = SRGAN(
                discriminator=discriminator,
                generator=generator,
                vgg_loss=vgg_loss,
                learning_rate=cfg.model.lr,
            )

        return model

    elif cfg.model.name == "SupResDiffGAN":
        return initialize_supresdiffgan(
            cfg, device, SupResDiffGAN, use_discriminator=True
        )

    elif cfg.model.name == "ESRGAN":
        discriminator = Discriminator_esrgan(in_channels=cfg.discriminator.in_channels)
        generator = Generator_esrgan(
            in_channels=cfg.generator.in_channels,
            out_channels=cfg.generator.out_channels,
            num_resblocks=cfg.generator.num_resblocks,
            scale_factor=cfg.generator.scale_factor,
        )
        feature_extractor = FeatureExtractor_esrgan()

        if cfg.model.load_model is not None:
            model_path = cfg.model.load_model
            _, ext = os.path.splitext(model_path)
            if ext == ".pth":
                model = ESRGAN(
                    discriminator=discriminator,
                    generator=generator,
                    feature_extractor=feature_extractor,
                    learning_rate=cfg.model.lr,
                )
                model.load_state_dict(torch.load(model_path, map_location=device))
            elif ext == ".ckpt":
                model = ESRGAN.load_from_checkpoint(
                    model_path,
                    map_location=device,
                    discriminator=discriminator,
                    generator=generator,
                    feature_extractor=feature_extractor,
                    learning_rate=cfg.model.lr,
                )
            else:
                raise ValueError(f"Unsupported file extension: {ext}")

        else:
            model = ESRGAN(
                discriminator=discriminator,
                generator=generator,
                feature_extractor=feature_extractor,
                learning_rate=cfg.model.lr,
            )

        return model

    elif cfg.model.name == "I2SB":
        unet = UNet_i2sb(channels=cfg.unet)
        diffusion = Diffusion_i2sb(
            n_timestep=cfg.diffusion.timesteps,
        )

        if cfg.model.load_model is not None:
            model_path = cfg.model.load_model
            _, ext = os.path.splitext(model_path)
            if ext == ".pth":
                model = I2SB(unet_model=unet, diffusion=diffusion, lr=cfg.model.lr)
                model.load_state_dict(torch.load(model_path, map_location=device))
            elif ext == ".ckpt":
                model = I2SB.load_from_checkpoint(
                    model_path,
                    map_location=device,
                    unet_model=unet,
                    diffusion=diffusion,
                    lr=cfg.model.lr,
                )
            else:
                raise ValueError(f"Unsupported file extension: {ext}")
        else:
            model = I2SB(unet_model=unet, diffusion=diffusion, lr=cfg.model.lr)

        return model

    elif cfg.model.name == "RealESRGAN":
        discriminator = Discriminator_realesrgan(
            in_channels=cfg.discriminator.in_channels
        )
        generator = Generator_realesrgan(
            in_channels=cfg.generator.in_channels,
            out_channels=cfg.generator.out_channels,
            num_resblocks=cfg.generator.num_resblocks,
            scale_factor=cfg.generator.scale_factor,
        )

        feature_extractor = FeatureExtractor_realesrgan(device)

        if cfg.model.load_model is not None:
            model_path = cfg.model.load_model
            _, ext = os.path.splitext(model_path)
            if ext == ".pth":
                model = RealESRGAN(
                    discriminator=discriminator,
                    generator=generator,
                    feature_extractor=feature_extractor,
                    learning_rate=cfg.model.lr,
                )
                model.load_state_dict(torch.load(model_path, map_location=device))
            elif ext == ".ckpt":
                model = RealESRGAN.load_from_checkpoint(
                    model_path,
                    map_location=device,
                    discriminator=discriminator,
                    generator=generator,
                    feature_extractor=feature_extractor,
                    learning_rate=cfg.model.lr,
                )
            else:
                raise ValueError(f"Unsupported file extension: {ext}")

        else:
            model = RealESRGAN(
                discriminator=discriminator,
                generator=generator,
                feature_extractor=feature_extractor,
                learning_rate=cfg.model.lr,
            )

        return model

    elif cfg.model.name == "ResShift":
        unet = UNet_resshift(channels=cfg.unet)
        diffusion = Diffusion_resshift(
            n_timestep=cfg.diffusion.timesteps,
        )

        if cfg.model.load_model is not None:
            model_path = cfg.model.load_model
            _, ext = os.path.splitext(model_path)
            if ext == ".pth":
                model = ResShift(unet_model=unet, diffusion=diffusion, lr=cfg.model.lr)
                model.load_state_dict(torch.load(model_path, map_location=device))
            elif ext == ".ckpt":
                model = ResShift.load_from_checkpoint(
                    model_path,
                    map_location=device,
                    unet_model=unet,
                    diffusion=diffusion,
                    lr=cfg.model.lr,
                )
            else:
                raise ValueError(f"Unsupported file extension: {ext}")
        else:
            model = ResShift(unet_model=unet, diffusion=diffusion, lr=cfg.model.lr)

        return model

    elif cfg.model.name == "SupResDiffGAN_without_adv":
        return initialize_supresdiffgan(
            cfg, device, SupResDiffGAN_without_adv, use_discriminator=False
        )

    elif cfg.model.name == "SupResDiffGAN_no_perceptual":
        if cfg.autoencoder == "VAE":
            model_id = "stabilityai/stable-diffusion-2-1"
            autoencoder = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(
                device
            )

        discriminator = Discriminator_supresdiffgan(
            in_channels=cfg.discriminator.in_channels,
            channels=cfg.discriminator.channels,
        )
        unet = UNet_supresdiffgan(cfg.unet)
        diffusion = Diffusion_supresdiffgan(
            timesteps=cfg.diffusion.timesteps,
            beta_type=cfg.diffusion.beta_type,
            posterior_type=cfg.diffusion.posterior_type,
        )

        if cfg.model.load_model is not None:
            model_path = cfg.model.load_model
            _, ext = os.path.splitext(model_path)
            if ext == ".pth":
                model = SupResDiffGAN_no_perceptual(
                    ae=autoencoder,
                    discriminator=discriminator,
                    unet=unet,
                    diffusion=diffusion,
                    learning_rate=cfg.model.lr,
                )
                model.load_state_dict(torch.load(model_path, map_location=device))
            elif ext == ".ckpt":
                model = SupResDiffGAN_no_perceptual.load_from_checkpoint(
                    model_path,
                    map_location=device,
                    ae=autoencoder,
                    discriminator=discriminator,
                    unet=unet,
                    diffusion=diffusion,
                    learning_rate=cfg.model.lr,
                )

            else:
                raise ValueError(f"Unsupported file extension: {ext}")

        else:
            model = SupResDiffGAN_no_perceptual(
                ae=autoencoder,
                discriminator=discriminator,
                unet=unet,
                diffusion=diffusion,
                learning_rate=cfg.model.lr,
            )

        return model

    elif cfg.model.name == "SupResDiffGAN_simple_gan":
        return initialize_supresdiffgan(
            cfg, device, SupResDiffGAN_simple_gan, use_discriminator=True
        )

    else:
        raise ValueError("Model not found")


def initialize_supresdiffgan(cfg, device, model_class, use_discriminator=True):
    """Helper function to initialize SupResDiffGAN variants.

    Parameters
    ----------
    cfg : DictConfig
        The configuration object containing model parameters and settings.
    device : torch.device
        The device on which the model will be loaded (e.g., 'cuda' or 'cpu').
    model_class : class
        The class of the model to initialize (e.g., SupResDiffGAN, SupResDiffGAN_without_adv).
    use_discriminator : bool, optional
        Whether to include the discriminator in the model initialization.
    use_vgg_loss : bool, optional
        Whether to include the VGG loss in the model initialization.

    Returns
    -------
    torch.nn.Module
        The initialized model.
    """
    if cfg.autoencoder == "VAE":
        model_id = "stabilityai/stable-diffusion-2-1"
        autoencoder = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(
            device
        )

    discriminator = (
        Discriminator_supresdiffgan(
            in_channels=cfg.discriminator.in_channels,
            channels=cfg.discriminator.channels,
        )
        if use_discriminator
        else None
    )

    unet = UNet_supresdiffgan(cfg.unet)

    diffusion = Diffusion_supresdiffgan(
        timesteps=cfg.diffusion.timesteps,
        beta_type=cfg.diffusion.beta_type,
        posterior_type=cfg.diffusion.posterior_type,
    )

    if cfg.use_perceptual_loss:
        if cfg.feature_extractor:
            vgg_loss = FeatureExtractor_supresdiffgan(device)
        else:
            vgg_loss = VGGLoss_supresdiffgan(device)
    else:
        vgg_loss = None

    if cfg.model.load_model is not None:
        model_path = cfg.model.load_model
        _, ext = os.path.splitext(model_path)
        if ext == ".pth":
            model = model_class(
                ae=autoencoder,
                discriminator=discriminator,
                unet=unet,
                diffusion=diffusion,
                learning_rate=cfg.model.lr,
                alfa_perceptual=cfg.model.alfa_perceptual,
                alfa_adv=cfg.model.alfa_adv,
                vgg_loss=vgg_loss,
            )
            model.load_state_dict(torch.load(model_path, map_location=device))
        elif ext == ".ckpt":
            model = model_class.load_from_checkpoint(
                model_path,
                map_location=device,
                ae=autoencoder,
                discriminator=discriminator,
                unet=unet,
                diffusion=diffusion,
                learning_rate=cfg.model.lr,
                alfa_perceptual=cfg.model.alfa_perceptual,
                alfa_adv=cfg.model.alfa_adv,
                vgg_loss=vgg_loss,
            )

        else:
            raise ValueError(f"Unsupported file extension: {ext}")

    else:
        model = model_class(
            ae=autoencoder,
            discriminator=discriminator,
            unet=unet,
            diffusion=diffusion,
            learning_rate=cfg.model.lr,
            alfa_perceptual=cfg.model.alfa_perceptual,
            alfa_adv=cfg.model.alfa_adv,
            vgg_loss=vgg_loss,
        )

    return model
