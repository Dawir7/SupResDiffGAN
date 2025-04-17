# Configuration Files Documentation

This document provides an explanation of the parameters used in the configuration files located in the `conf/` directory. Each configuration file corresponds to a specific model or use case and contains parameters for training, evaluation, and logging. While there is a general pattern across the configuration files, some parameters are specific to certain models.

---

## General Structure of Configuration Files

All configuration files follow a similar structure with the following sections:

- **Mode**: Specifies the mode of operation (e.g., `train`, `test`, or `train-test`).
- **Model**: Contains model-specific parameters such as learning rate, loss weights, and model name.
- **W&B Logger**: Configuration for Weights & Biases (W&B) experiment tracking.
- **Trainer**: Parameters for training, such as the number of epochs and logging frequency.
- **Dataset**: Parameters for dataset loading and preprocessing.
- **Evaluation**: Parameters for evaluation, such as the number of steps and posterior types.
- **Checkpoint**: Configuration for saving model checkpoints.
- **Additional Sections**: Some models include additional sections like `autoencoder`, `unet`, `diffusion`, or `discriminator`.

---

## Parameters Overview

### **1. Mode**

- **`mode`**: Specifies the mode of operation.
  - `train`: Training mode.
  - `test`: Testing mode.
  - `train-test`: Combined training and testing mode. After the training, the model will be tested (if the training is not completed, the model will not be tested).

---

### **2. Model**

Defines the model-specific parameters.

| Parameter              | Description                                                                                     | Example Values                |
|------------------------|-------------------------------------------------------------------------------------------------|--------------------------------|
| `name`                | Name of the model.                                                                              | `SupResDiffGAN`, `SRGAN`      |
| `lr`                  | Learning rate for the optimizer.                                                                | `0.0001`, `0.0002`            |
| `alfa_perceptual`     | Weight for the perceptual loss (specific to SupResDiffGAN).                                        | `0.001`                       |
| `alfa_adv`            | Weight for the adversarial loss (specific to SupResDiffGAN).                                       | `0.01`                        |
| `use_perceptual_loss` | Whether to use perceptual loss during (specific to SupResDiffGAN).                                                  | `True`, `False`               |
| `load_model`          | Path to a pre-trained model checkpoint (if applicable).                                         | `models/checkpoints/model.ckpt` |

---

### **3. W&B Logger**

Configuration for Weights & Biases (W&B) experiment tracking.

| Parameter      | Description                                      | Example Values         |
|----------------|--------------------------------------------------|------------------------|
| `project`      | Name of the W&B project.                        | `your_project`         |
| `entity`       | Name of the W&B entity (organization or user).  | `your_entity`          |

---

### **4. Trainer**

Defines training-related parameters.

| Parameter                  | Description                                                                                     | Example Values                |
|----------------------------|-------------------------------------------------------------------------------------------------|--------------------------------|
| `max_epochs`              | Maximum number of training epochs.                                                              | `200`, `300`                  |
| `max_steps`               | Maximum number of training steps.                                                               | `330000`                      |
| `check_val_every_n_epoch` | Frequency (in epochs) to run validation.                                                        | `5`                           |
| `limit_val_batches`       | Number of validation batches to process.                                                        | `1`                           |
| `log_every_n_steps`       | Frequency (in steps) to log metrics.                                                            | `1`                           |

---

### **5. Dataset**

Defines dataset-related parameters.

| Parameter      | Description                                                                                     | Example Values                |
|----------------|-------------------------------------------------------------------------------------------------|--------------------------------|
| `name`         | Name of the dataset.                                                                            | `celeb`, `imagenet`           |
| `batch_size`   | Batch size for training and validation.                                                         | `8`, `16`, `32`               |
| `resize`       | Whether to resize images (specific to diffusion or GAN models).                                 | `True`, `False`               |
| `scale`        | Scale factor for super-resolution.                                                              | `4`                           |

---

### **6. Evaluation**

Defines evaluation-related parameters.

| Parameter              | Description                                                                                     | Example Values                |
|------------------------|-------------------------------------------------------------------------------------------------|--------------------------------|
| `mode`                | Evaluation mode.                                                                                | `steps`, `posterior`, `all`   |
| `steps`               | List of inference steps for evaluation.                                                         | `[100, 500]`                  |
| `posteriors`          | List of posterior types for evaluation.                                                         | `['ddpm', 'ddim']`            |
| `save_results`        | Whether to save evaluation results to a file.                                                   | `True`, `False`               |
| `results_file`        | Path to the file where evaluation results will be stored.                                       | `evaluation_results/results.csv` |

---

### **7. Checkpoint**

Defines parameters for saving model checkpoints.

| Parameter      | Description                                                                                     | Example Values                |
|----------------|-------------------------------------------------------------------------------------------------|--------------------------------|
| `monitor`      | Metric to monitor for saving the best checkpoint.                                               | `val/LPIPS`                   |
| `dirpath`      | Directory where checkpoints will be saved.                                                      | `models/checkpoints/`         |
| `save_top_k`   | Number of top checkpoints to save.                                                              | `1`                           |
| `mode`         | Mode for monitoring the metric (`min` or `max`).                                                | `min`, `max`                  |

---

### **8. Additional Sections**

Some models include additional sections for specific configurations.

#### **Autoencoder**

| Parameter          | Description                                                                                     | Example Values                |
|--------------------|-------------------------------------------------------------------------------------------------|--------------------------------|
| `autoencoder`      | Type of autoencoder used in the model.                                                          | `VAE`                         |
| `feature_extractor`| Whether to use a feature extractor.                                                             | `True`, `False`               |

#### **UNet**

| Parameter      | Description                                                                                     | Example Values                |
|----------------|-------------------------------------------------------------------------------------------------|--------------------------------|
| `unet`         | List of channel sizes for the UNet architecture.                                                | `[64, 96, 128, 512]`          |

#### **Diffusion**

| Parameter                  | Description                                                                                     | Example Values                |
|----------------------------|-------------------------------------------------------------------------------------------------|--------------------------------|
| `timesteps`               | Number of diffusion timesteps.                                                                  | `1000`                        |
| `beta_type`               | Type of beta schedule for diffusion.                                                            | `cosine`                      |
| `posterior_type`          | Type of posterior used during inference.                                                        | `ddpm`, `ddim`                |
| `validation_timesteps`    | Number of timesteps used during validation.                                                     | `1000`                        |
| `validation_posterior_type` | Posterior type used during validation.                                                        | `ddpm`                        |

#### **Discriminator**

| Parameter      | Description                                                                                     | Example Values                |
|----------------|-------------------------------------------------------------------------------------------------|--------------------------------|
| `in_channels`  | Number of input channels for the discriminator.                                                 | `3`, `6`                      |
| `channels`     | List of channel sizes for the discriminator architecture.                                        | `[64, 128, 256, 512]`         |

---

## Model-Specific Configurations

### **SupResDiffGAN**

- Includes parameters for `alfa_perceptual` and `alfa_adv` to control loss weights.
- Uses `autoencoder`, `discriminator`, `unet` and `diffusion` sections for hybrid architecture.

### **SRGAN**

- Includes `generator` and `discriminator` sections for GAN-specific architecture.
- Does not use `diffusion` or `autoencoder`.

### **ESRGAN**

- Similar to SRGAN.

### **RealESRGAN**

- Similar to SRGAN.

### **I2SB**

- Includes `unet` and `diffusion` sections for diffusion-based super-resolution.

### **ResShift**

- Includes `unet` and `diffusion` sections.

---

## Notes

- Parameters not listed in a specific configuration file will fall back to defaults in `config.yaml`.
- Refer to the individual configuration files for exact parameter values and usage.
