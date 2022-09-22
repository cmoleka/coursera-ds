"""
    lets have some fun with stable diffusion
"""
import os
# import cv2
import numpy as np
# from IPYTHON.display import HTML
from base64 import b64encode
import torch
from torch import autocast
from torch.nn import functional as F
from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel, PNDMScheduler, LMSDiscreteScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
# from transformers import


device = 'cuda'

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    revision='fp16',
    torch_dtype=torch.float16,
    use_auth_token=True
)

pipe = pipe.to(device)

prompt = "a photo of an astronaut riding a horse on mars"
with autocast(device):
    image = pipe(prompt)["sample"][0]

image.save("astronaut_rides_horse.png")
