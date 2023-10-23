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

from kandinsky2 import get_kandinsky2

"""Model definition"""

model = get_kandinsky2('cuda', 
                       task_type='text2img', 
                       cache_dir='/tmp/kandinsky2', 
                       model_version='2.1', 
                       use_flash_attention=False)

"""Functions"""

def er_kandinsky(prompt_dict, save_path, model_name, n):
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
      image = model.generate_text2img(prompt=prompt,
                                        num_steps=100, 
                                        batch_size=1, 
                                        guidance_scale=4,
                                        h=256, w=256,
                                        sampler='p_sampler', 
                                        prior_cf_scale=4,
                                        prior_steps="5",)  
      imgname = str(re.split(r'[_.]', URL)[-2] + '_' + str(i))
      image[0].save("".join([urlpath, imgname, '.jpg']))

      # Free up GPU memory
      gc.collect()
      torch.cuda.empty_cache()
      
"""Load Enriched prompts dictionary"""

### HAND-CRAFTED PROMPTS

# with open('Scrivania/Tesi Cavallaro/Files/enriched_prompts.json') as json_file:
#   enriched_prompts = json.load(json_file)

with open('Scrivania/Tesi Cavallaro/Files/enriched_prompts_5099.json') as json_file:
  enriched_prompts = json.load(json_file)
  
  """Generation w/ Kandinsky"""
  
# Setup paths and call the generation function
imgsavepath = 'Immagini/Enriched Prompts/'
er_kandinsky(enriched_prompts, imgsavepath, 'Kandinsky', 10)