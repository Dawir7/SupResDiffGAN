# SupResDiffGAN a new approach for the Super-Resolution task 🚀✨

<div align="center">
  <table>
    <tr>
      <td>
        <img src="assets/wust_logo.png" alt="WUST Logo" width="85">
      </td>
      <td align="center">
        <b><a href="https://github.com/Dawir7">Dawid Kopeć<sup>1</sup></a></b>,
        <b><a href="https://github.com/WojciechKoz">Wojciech Kozłowski<sup>1</sup></a></b>,
        <b><a href="https://github.com/mwizer">Maciej Wizerkaniuk<sup>1</sup></a></b>,
        <b><a href="https://github.com/dkrutul">Dawid Krutul<sup>1</sup></a></b>,
        <b><a href="https://scholar.google.com/citations?user=pmQHb5IAAAAJ">Jan Kocoń<sup>1</sup></a></b>, and
        <b><a href="https://scholar.google.com/citations?user=XmOBJZYAAAAJ">Maciej Zięba<sup>1</sup></a></b><br>
        WUST, Wybrzeże Stanisława Wyspiańskiego 27, 50-370 Wrocław, Poland<br>
        {wojciech.kozlowski, jan.kocon, maciej.zieba}@pwr.edu.pl
      </td>
    </tr>
  </table>
</div>

<div align="center">

[![Conference](https://img.shields.io/badge/ICCS%202025-Published-brightgreen)](https://www.iccs-meeting.org/iccs2025/)&nbsp;
[![arXiv](https://img.shields.io/badge/arXiv%20paper-2504.13622-b31b1b.svg)](https://arxiv.org/abs/2504.13622)&nbsp;
[![PapersWithCode](https://img.shields.io/badge/Papers%20with%20Code-SupResDiffGAN-32B1B4?logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB3aWR0aD0iNjA2IiBoZWlnaHQ9IjYwNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgb3ZlcmZsb3c9ImhpZGRlbiI%2BPGRlZnM%2BPGNsaXBQYXRoIGlkPSJjbGlwMCI%2BPHJlY3QgeD0iLTEiIHk9Ii0xIiB3aWR0aD0iNjA2IiBoZWlnaHQ9IjYwNiIvPjwvY2xpcFBhdGg%2BPC9kZWZzPjxnIGNsaXAtcGF0aD0idXJsKCNjbGlwMCkiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDEgMSkiPjxyZWN0IHg9IjUyOSIgeT0iNjYiIHdpZHRoPSI1NiIgaGVpZ2h0PSI0NzMiIGZpbGw9IiM0NEYyRjYiLz48cmVjdCB4PSIxOSIgeT0iNjYiIHdpZHRoPSI1NyIgaGVpZ2h0PSI0NzMiIGZpbGw9IiM0NEYyRjYiLz48cmVjdCB4PSIyNzQiIHk9IjE1MSIgd2lkdGg9IjU3IiBoZWlnaHQ9IjMwMiIgZmlsbD0iIzQ0RjJGNiIvPjxyZWN0IHg9IjEwNCIgeT0iMTUxIiB3aWR0aD0iNTciIGhlaWdodD0iMzAyIiBmaWxsPSIjNDRGMkY2Ii8%2BPHJlY3QgeD0iNDQ0IiB5PSIxNTEiIHdpZHRoPSI1NyIgaGVpZ2h0PSIzMDIiIGZpbGw9IiM0NEYyRjYiLz48cmVjdCB4PSIzNTkiIHk9IjE3MCIgd2lkdGg9IjU2IiBoZWlnaHQ9IjI2NCIgZmlsbD0iIzQ0RjJGNiIvPjxyZWN0IHg9IjE4OCIgeT0iMTcwIiB3aWR0aD0iNTciIGhlaWdodD0iMjY0IiBmaWxsPSIjNDRGMkY2Ii8%2BPHJlY3QgeD0iNzYiIHk9IjY2IiB3aWR0aD0iNDciIGhlaWdodD0iNTciIGZpbGw9IiM0NEYyRjYiLz48cmVjdCB4PSI0ODIiIHk9IjY2IiB3aWR0aD0iNDciIGhlaWdodD0iNTciIGZpbGw9IiM0NEYyRjYiLz48cmVjdCB4PSI3NiIgeT0iNDgyIiB3aWR0aD0iNDciIGhlaWdodD0iNTciIGZpbGw9IiM0NEYyRjYiLz48cmVjdCB4PSI0ODIiIHk9IjQ4MiIgd2lkdGg9IjQ3IiBoZWlnaHQ9IjU3IiBmaWxsPSIjNDRGMkY2Ii8%2BPC9nPjwvc3ZnPg%3D%3D)](https://paperswithcode.com/paper/supresdiffgan-a-new-approach-for-the-super)

</div>

This repository contains the implementation of our novel super-resolution (SR) method, as presented in our paper published at the **ICCS 2025**. The repository is designed with modularity and flexibility in mind, leveraging **PyTorch Lightning** for training, **Hydra** for configuration management, and **Weights & Biases (W&B)** for experiment tracking.

---

## Abstract 📜

> In this work, we present SupResDiffGAN, a novel hybrid architecture that combines the strengths of Generative Adversarial Networks (GANs) and diffusion models for super-resolution tasks. By leveraging latent space representations and reducing the number of diffusion steps, SupResDiffGAN achieves significantly faster inference times than other diffusion-based super-resolution models while maintaining competitive perceptual quality. To prevent discriminator overfitting, we propose adaptive noise corruption, ensuring a stable balance between the generator and the discriminator during training. Extensive experiments on benchmark datasets show that our approach outperforms traditional diffusion models such as SR3 and I<sup>2</sup>SB in efficiency and image quality. This work bridges the performance gap between diffusion- and GAN-based methods, laying the foundation for real-time applications of diffusion models in high-resolution image generation.

![The training process of our proposed model](assets/training.png)

---

## Table of Contents 📚

- [SupResDiffGAN a new approach for the Super-Resolution task 🚀✨](#supresdiffgan-a-new-approach-for-the-super-resolution-task-)
  - [Abstract 📜](#abstract-)
  - [Table of Contents 📚](#table-of-contents-)
  - [Results 📊](#results-)
    - [Comparison of Time of Batch Inference in Seconds](#comparison-of-time-of-batch-inference-in-seconds)
    - [Visual Results](#visual-results)
  - [Getting Started 🛠️](#getting-started-️)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [Configuration Management with Hydra](#configuration-management-with-hydra)
      - [Example: Override Parameters](#example-override-parameters)
      - [Configuration Files](#configuration-files)
    - [Experiment Tracking with W\&B](#experiment-tracking-with-wb)
  - [Datasets 📂](#datasets-)
    - [Prerequisites](#prerequisites-1)
    - [Download and Prepare Datasets](#download-and-prepare-datasets)
    - [Examples:](#examples)
  - [Usage ▶️](#usage-️)
    - [Training](#training)
    - [Evaluation](#evaluation)
    - [More info](#more-info)
  - [Model Weights](#model-weights)
    - [Download Link](#download-link)
    - [Usage](#usage)
  - [Citation 📖](#citation-)
  - [Acknowledgement 🙏](#acknowledgement-)
  - [License 📜](#license-)
    - [Third-Party Code and Licenses](#third-party-code-and-licenses)

---

## Results 📊

The best and second-best results are highlighted in **bold** and <ins>underline</ins>, respectively. Methods are categorized into _Diffusion-based_ and _GAN-based_ to reflect their distinct architectural frameworks.

| **Model / Dataset**       | **Imagenet** | **Celeb** | **Div2k** | **RealSR-nikon** | **RealSR-canon** | **Set14** | **Urban100** |
|---------------------------|--------------|-----------|-----------|------------------|------------------|-----------|--------------|
| **Metric**                | **LPIPS ↓**  | **LPIPS ↓** | **LPIPS ↓** | **LPIPS ↓**       | **LPIPS ↓**       | **LPIPS ↓** | **LPIPS ↓**  |
| **GAN-based methods**      |              |           |           |                  |                  |           |              |
| **SRGAN**                 | 0.3452       | 0.2441    | 0.3327    | 0.3464           | <ins>0.3050</ins>    | 0.2901    | 0.3156       |
| **ESRGAN**                | <ins>0.2320</ins>| <ins>0.1903</ins>| <ins>0.2649</ins>| <ins>0.3380</ins>    | 0.3053           | <ins>0.2375</ins>| <ins>0.2408</ins> |
| **Real-ESRGAN**           | **0.2123**   | **0.1690**| **0.2562**| **0.3309**       | **0.3020**       | **0.2301**| **0.2285**   |
|   **Diffusion-based methods**|              |           |           |                  |                  |           |              |
| **SR3**                   | <ins>0.3519</ins>| 0.2229    | 0.3396    | <ins>0.4018</ins>    | 0.4008           | <ins>0.3015</ins>| **0.2428**   |
| **I<sup>2</sup>SB**                  | 0.3755       | <ins>0.2221</ins>| <ins>0.3309</ins>| 0.4069           | <ins>0.3867</ins>    | 0.3169    | 0.2635       |
| **ResShift**              | 0.5360       | 0.3275    | 0.4724    | 0.4959           | 0.4671           | 0.4832    | 0.4822       |
| **SupResDiffGAN**         | **0.3079**   | **0.1875**| **0.2876**| **0.3970**       | **0.3853**       | **0.2789**| <ins>0.2570</ins> |

### Comparison of Time of Batch Inference in Seconds

The best and second-best results are highlighted in **bold** and <ins>underline</ins>, respectively. Methods are categorized into _Diffusion-based_ and _GAN-based_ to reflect their distinct architectural frameworks.

| **Model / Dataset**       | **Imagenet** | **Celeb** | **Div2k** | **RealSR-nikon** | **RealSR-canon** | **Set14** | **Urban100** |
|---------------------------|--------------|-----------|-----------|------------------|------------------|-----------|--------------|
| **Metric**                | **Time per batch [s]** | **Time per batch [s]** | **Time per batch [s]** | **Time per batch [s]** | **Time per batch [s]** | **Time per batch [s]** | **Time per batch [s]** |
| **GAN-based methods**      |              |           |           |                  |                  |           |              |
| **SRGAN**                 | **0.0671**   | **0.0109**| **0.0193**| **0.0367**       | **0.0113**       | **0.0888**| **0.0070**   |
| **ESRGAN**                | 0.2188       | 0.0870    | 0.2316    | 0.2711           | 0.1504           | <ins>0.2049</ins>| <ins>0.0821</ins> |
| **Real-ESRGAN**           | <ins>0.1392</ins>| <ins>0.0816</ins>| <ins>0.1899</ins>| <ins>0.2468</ins>    | <ins>0.1427</ins>    | 0.2361    | 0.1013       |
|   **Diffusion-based methods**|              |           |           |                  |  
| **SR3**                   | 1.9953       | 0.3072    | 7.6377    | 8.4242           | 3.6420           | 0.8627    | 1.5028       |
| **I<sup>2</sup>SB**                  | <ins>1.6776</ins>| **0.1184**| <ins>6.7292</ins>| <ins>7.0910</ins>    | <ins>3.1629</ins>    | 1.8049    | <ins>1.2395</ins> |
| **ResShift**              | 2.2466       | 0.4394    | 8.6647    | 8.9677           | 4.1880           | <ins>0.5983</ins>| 1.6762       |
| **SupResDiffGAN**         | **0.2954**   | <ins>0.1832</ins>| **0.9333**| **1.0021**       | **0.6114**       | **0.3542**| **0.3206**   |

### Visual Results

![Visual Results](assets/results.png)

Two representative SupResDiffGAN outputs: (top) 4× face superresolution at 128×128→512×512 pixels (bottom) 4× natural image super-resolution at 125×93→500×372 pixels.

![Visual Comparison](assets/visual_comparison.png)

Qualitative comparison of visual performance on two example images from ImageNet. Low-quality inputs are on the left, while results from bicubic upscale and seven SR models: SRGAN, ESRGAN, Real-ESRGAN, SR3, ResShift, I<sup>2</sup>SB, and Ours are on the right.

---

## Getting Started 🛠️

### Prerequisites

- Python >= 3.9,
- PyTorch Lightning == 2.2.2
- CUDA-enabled GPU (recommended for training)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Dawir7/SupResDiffGAN.git
   cd SupResDiffGAN
   ```

2. Create a virtual environment and activate it:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements-gpu.txt
   ```

### Configuration Management with Hydra

This repository uses **Hydra** for managing configurations. Configuration files are located in the `conf/` directory. You can override any configuration parameter directly from the command line.

#### Example: Override Parameters

```bash
python train_model.py model.name=ESRGAN dataset.batch_size=16 trainer.max_epochs=50
```

More information about overriding parameters in `Hydra` documentation [Basic Override syntax](https://hydra.cc/docs/advanced/override_grammar/basic/)

#### Configuration Files

- `config.yaml`: Default configuration file.
- `config_srgan.yaml`: Configuration for SRGAN.
- `config_esrgan.yaml`: Configuration for ESRGAN.
- `config_real_esrgan`: Configuration for Real-ESRGAN.
- `config_sr3.yaml`: Configuration for SR3.
- `config_i2sb.yaml`: Configuration for I<sup>2</sup>SB.
- `config_resshift.yaml`: Configuration for ResShift.
- `config_supresdiffgan.yaml`: Configuration for SupResDiffGAN.
- `config_supresdiffgan_without_adv.yaml`: Configuration for SupResDiffGAN model without a discriminator or adversarial loss.
- `config_supresdiffgan_simple_gan.yaml`: Configuration for SupResDiffGAN model with a discriminator but without Gaussian noise augmentation.

### Experiment Tracking with W&B

This repository integrates **Weights & Biases (W&B)** for experiment tracking. Follow these steps to get started:

1. **Login to W&B**:

   ```bash
   wandb login
   ```

2. **Track Experiments**:

   - Metrics, losses, and visualizations are automatically logged to your W&B project.
   - Customize the W&B project name in the configuration file in use f.e.:

     ```yaml
     wandb_logger:
        project: 'your_project' # your wandb project
        entity: 'your_entity' # your wandb entity
     ```

3. **View Results**:
   - Visit [https://wandb.ai](https://wandb.ai) and navigate to your project to view experiment results.

---

## Datasets 📂

This section outlines how to download the necessary datasets for training and evaluating the SupResDiffGAN model. We provide a convenient bash script to automate the download process.

### Prerequisites

- Activated virtual environment (as described in the [Installation](#installation) section).
- **Note:** If you haven't installed all GPU requirements using `requirements-gpu.txt`, the minimal libraries required for downloading the CelebA and ImageNet datasets are listed in `requirements-data.txt`. You can install these specifically using:
  ```bash
  pip install -r requirements-data.txt
  ```

### Download and Prepare Datasets

The `get_data.sh` script will download the specified datasets to the appropriate directories (the exact locations are defined within the script). Please ensure you have sufficient disk space before running the script.

**Notes:** 
- The specific implementation and sources for each dataset download are defined within the `get_data.sh` script. Refer to the script for more details on the download process for each dataset.
- Due to the potentially long download and processing times for some datasets, especially ImageNet and large RealSR variants, it is highly recommended to run the script within a terminal multiplexer such as `tmux` or `screen`. This will allow the process to continue even if your SSH connection is interrupted.
- **Crucially, datasets are subjects to its own license terms and conditions.** By using any of datasets, **you are solely responsible for understanding and complying with the respective dataset's license.** We, as the authors of this code repository, assume no responsibility for your usage of these datasets or any potential license violations. It is your responsibility to ensure your use adheres to the terms set forth by the dataset providers. \
We strongly recommend that you familiarize yourself with the licensing terms of any dataset you choose to use before downloading and incorporating it into your workflow. Links to the official licenses are typically available on the dataset providers' websites.

1. **Ensure you are in the repository's root directory:**

   ```bash
   cd SupResDiffGAN
   ```

2. **Run the `get_data.sh` script with the desired dataset flags.** The script accepts the following flags:

   - `-i` or `--imagenet`: Downloads the ImageNet dataset.
   - `-c` or `--celeba`: Downloads the CelebA dataset.
   - `-d` or `--div2k`: Downloads the Div2k dataset.
   - `-r` or `--realsr`: Downloads the RealSR dataset.
   - `-s` or `--set14`: Downloads the Set14 dataset.
   - `-u` or `--urban100`: Downloads the Urban100 dataset.

### Examples:

- Download the ImageNet dataset:

  ```bash
  bash get_data.sh -i
  ```

- Download the ImageNet and CelebA datasets:

  ```bash
  bash get_data.sh -i -c
  ```

- Download supported datasets using full names:

  ```bash
  bash get_data.sh --celeba --div2k
  ```

---

## Usage ▶️

### Training

To train a model, use the `train_model.py` script. Example:

```bash
python train_model.py -cn "config_supresdiffgan"
```

### Evaluation

To evaluate a trained model, use the `evaluate_model.py` script. Example:

```bash
python evaluate_model.py "config_supresdiffgan"
```

### More info

More about configs in `CONFIGS.md`.

More about usage of `Hydra` flags: [Hydra documentation](https://hydra.cc/docs/advanced/hydra-command-line-flags/)

---

## Model Weights

We provide pre-trained weights for SupResDiffGAN to facilitate evaluation and fine-tuning. These weights are trained on ImageNet and can be used for inference or as a starting point for further training.

### Download Link

[Download](https://drive.google.com/drive/folders/1dQz_mSEiEdf8aEGT4lEijz8psBfLQ1Qe?usp=sharing)

### Usage

To use a pre-trained model, specify the path to the checkpoint file in the `load_model` field of the configuration file. For example, in `config.yaml`:

```yaml
model:
  load_model: 'path/to/your/checkpoint_file.pth'  # Path to the pre-trained model checkpoint
```

---

## Citation 📖

If you use this repository in your research, please cite our paper:

```text
@inproceedings{kopec2025supresdiffgan,
  title={SupResDiffGAN: A New Approach for the Super-Resolution Task},
  author={Kope{\'c}, Dawid and Koz{\l}owski, Wojciech and Wizerkaniuk, Maciej and Krutul, Dawid and Koco{\'n}, Jan and Zi{\k{e}}ba, Maciej},
  booktitle={Proceedings of the International Conference on Computational Science (ICCS)},
  year={2025}
}
```

---

## Acknowledgement 🙏

We would like to acknowledge the following repositories and works that served as inspiration or baselines for our research:

- [PyTorch-GAN](https://github.com/eriklindernoren/PyTorch-GAN): A collection of PyTorch implementations of GANs.
- [Real-ESRGAN-bicubic](https://github.com/final-0/Real-ESRGAN-bicubic): A bicubic version of Real-ESRGAN for super-resolution tasks.
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN): A practical algorithm for general image restoration.
- [ResShift](https://github.com/zsyOAOA/ResShift): A novel approach for image super-resolution.
- [I<sup>2</sup>SB](https://github.com/NVlabs/I2SB): A diffusion-based method for image-to-image super-resolution.

We are grateful for the contributions of these projects to the field of super-resolution and deep learning.

---

## License 📜

This repository is licensed under the **Academic Free License (AFL) v3.0**. See the [LICENSE.txt](LICENSE.txt) file for the full license text.

By using this repository, you agree to comply with the terms of the Academic Free License and any applicable third-party licenses.

### Third-Party Code and Licenses

Some parts of this repository are modified or adapted from other open-source projects mentioned in the [Acknowledgement 🙏](#acknowledgement-) section. These parts retain their original licenses, which are included in their respective directories. Please refer to the following for more details:

- **[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)**: Licensed under the **BSD 3-Clause License**. See the [RealESRGAN/LICENSE.txt](RealESRGAN/LICENSE.txt) file for the full license text.
- **[ResShift](https://github.com/zsyOAOA/ResShift)**: Licensed under the **S-Lab License 1.0**. See the [ResShift/LICENSE.txt](ResShift/LICENSE.txt) file for the full license text.
- **[I<sup>2</sup>SB](https://github.com/NVlabs/I2SB)**: Licensed under the **NVIDIA Source Code License**. See the [I<sup>2</sup>SB/LICENSE.txt](I2SB/LICENSE.txt) file for the full license text.
