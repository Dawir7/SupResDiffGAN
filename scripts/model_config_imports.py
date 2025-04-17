"""File containing imports for all models in the repo for model_config.py."""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from diffusers import AutoencoderKL

from ESRGAN.ESRGAN import ESRGAN
from ESRGAN.modules.discriminator import Discriminator as Discriminator_esrgan
from ESRGAN.modules.feature_extractor import FeatureExtractor as FeatureExtractor_esrgan
from ESRGAN.modules.generator import GeneratorRRDB as Generator_esrgan
from I2SB.I2SB import I2SB
from I2SB.modules.Diffusion import ISBDiffusion as Diffusion_i2sb
from I2SB.modules.UNet import UNet as UNet_i2sb
from RealESRGAN.modules.discriminator import Discriminator as Discriminator_realesrgan
from RealESRGAN.modules.feature_extractor import (
    FeatureExtractor as FeatureExtractor_realesrgan,
)
from RealESRGAN.modules.generator import Generator as Generator_realesrgan
from RealESRGAN.RealESRGAN import RealESRGAN
from ResShift.modules.Diffusion import Diffusion as Diffusion_resshift
from ResShift.modules.UNet import UNet as UNet_resshift
from ResShift.ResShift import ResShift
from SR3.modules.Diffusion import Diffusion as Diffusion_sr3
from SR3.modules.UNet import UNet as UNet_sr3
from SR3.SR3 import SR3
from SRGAN.modules.discriminator import Discriminator as Discriminator_srgan
from SRGAN.modules.generator import Generator as Generator_srgan
from SRGAN.modules.VggLoss import VGGLoss as VGGLoss_srgan
from SRGAN.SRGAN import SRGAN
from SupResDiffGAN.modules.Diffusion import Diffusion as Diffusion_supresdiffgan
from SupResDiffGAN.modules.Discriminator import (
    Discriminator as Discriminator_supresdiffgan,
)
from SupResDiffGAN.modules.FeatureExtractor import (
    FeatureExtractor as FeatureExtractor_supresdiffgan,
)
from SupResDiffGAN.modules.UNet import UNet as UNet_supresdiffgan
from SupResDiffGAN.modules.VggLoss import VGGLoss as VGGLoss_supresdiffgan
from SupResDiffGAN.SupResDiffGAN import SupResDiffGAN
from SupResDiffGAN.SupResDiffGAN_no_perceptual import SupResDiffGAN_no_perceptual
from SupResDiffGAN.SupResDiffGAN_simple_gan import SupResDiffGAN_simple_gan
from SupResDiffGAN.SupResDiffGAN_without_adv import SupResDiffGAN_without_adv
