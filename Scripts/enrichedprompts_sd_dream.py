# -*- coding: utf-8 -*-

"""Imports"""

import pandas as pd

import re
import os
import gc

import json

from PIL import Image
import urllib.request

import torch
from diffusers import StableDiffusionPipeline

"""Functions"""

def stable_diffusion_dreamlike(prompt_dict, save_path, model_name, n):
  '''
    prompt_dict -> dictionary containing link to the COCO images and the corresponding enriched caption
    save_path -> path to the root folder where to save the images
    model_name -> name to give to the folder where the images will be saved
  '''

  # 1) Create model folder if not exists
  modelpath = "".join([save_path, model_name, '/'])
  if not os.path.exists(modelpath):
    os.mkdir(modelpath)

  # 2) Iterate over the URLs
  for URL in prompt_dict.keys():

    # Free up GPU memory
    gc.collect()
    torch.cuda.empty_cache()

    # 2.1) Creates URL folder
    urlpath = "".join([modelpath, str(re.split(r'[_.]', URL)[-2]), '/'])
    if not os.path.exists(urlpath):
      os.mkdir(urlpath)

    # 2.2) Extract the prompts
    prompt = prompt_dict[URL]

    print(URL, prompt)

    # 2.3) Generate n images per prompt
    for i in range(0, n):
      image = pipe(prompt, generator=generator).images[0]
      imgname = str(re.split(r'[_.]', URL)[-2] + '_' + str(i))
      image.save("".join([urlpath, imgname, '.jpg']))

      # Free up GPU memory
      gc.collect()
      torch.cuda.empty_cache()

"""Load Enriched prompts dictionary"""

### HAND-CRAFTED PROMPTS

# with open('Scrivania/Tesi Cavallaro/Files/enriched_prompts.json') as json_file:
#   enriched_prompts = json.load(json_file)

with open('Scrivania/Tesi Cavallaro/Files/enriched_prompts_5099.json') as json_file:
  enriched_prompts = json.load(json_file)

"""Generation w/ Stable Diffusion"""

# pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16, use_auth_token=True)
# pipe = pipe.to("cuda")
# pipe.enable_attention_slicing()

# generator = torch.Generator("cuda").manual_seed(3052023)

# # Setup paths and call the generation function
# imgsavepath = 'Immagini/Enriched Prompts/'
# stable_diffusion_dreamlike(enriched_prompts, imgsavepath, 'StableDiffusion2.1', 10)

"""Generation w/ DreamlikeArt"""

pipe = StableDiffusionPipeline.from_pretrained("dreamlike-art/dreamlike-photoreal-2.0", torch_dtype=torch.float16, use_auth_token=True)
pipe = pipe.to("cuda")
pipe.enable_attention_slicing()

generator = torch.Generator("cuda").manual_seed(3052023)

# Setup paths and call the generation function
imgsavepath = 'Immagini/Enriched Prompts/'
stable_diffusion_dreamlike(enriched_prompts, imgsavepath, 'DreamlikeArt', 10)