# -*- coding: utf-8 -*-

"""Setup"""

# Click on the link that appears in the terminal, copy the Wandb key and press Enter

# Model references
# DALLE-MEGA
# DALLE_MODEL = "dalle-mini/dalle-mini/mega-1-fp16:latest"
DALLE_COMMIT_ID = None

# DALLE-MINI (smaller model)
DALLE_MODEL = "dalle-mini/dalle-mini/mini-1:v0"

# VQGAN model
VQGAN_REPO = "dalle-mini/vqgan_imagenet_f16_16384"
VQGAN_COMMIT_ID = "e93a26e7707683d349bf5d5c41c5b0ef69b677a9"


# Environment setup
import jax
import jax.numpy as jnp

# check how many devices are available
jax.local_device_count()

# Model setup
# Load models & tokenizer
from dalle_mini import DalleBart, DalleBartProcessor
from vqgan_jax.modeling_flax_vqgan import VQModel
from transformers import CLIPProcessor, FlaxCLIPModel

# Load dalle-mini
model, params = DalleBart.from_pretrained(
    DALLE_MODEL, revision=DALLE_COMMIT_ID, dtype=jnp.float16, _do_init=False
)

# Load VQGAN
vqgan, vqgan_params = VQModel.from_pretrained(
    VQGAN_REPO, revision=VQGAN_COMMIT_ID, _do_init=False
)

# Replication of model parameters for faster inference
from flax.jax_utils import replicate

params = replicate(params)
vqgan_params = replicate(vqgan_params)

# Parallelization of model functions for faster inference
from functools import partial

# model inference
@partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(3, 4, 5, 6))
def p_generate(
    tokenized_prompt, key, params, top_k, top_p, temperature, condition_scale
):
    return model.generate(
        **tokenized_prompt,
        prng_key=key,
        params=params,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        condition_scale=condition_scale,
    )

# decode image
@partial(jax.pmap, axis_name="batch")
def p_decode(indices, params):
    return vqgan.decode_code(indices, params=params)

# Import tokenizer
from dalle_mini import DalleBartProcessor
processor = DalleBartProcessor.from_pretrained(DALLE_MODEL, revision=DALLE_COMMIT_ID)

# Set random seed to each device
import random

# create a random key
seed = random.randint(0, 2**32 - 1)
key = jax.random.PRNGKey(seed)

"""Imports"""

import pandas as pd
import json
import re
import os
import gc
import torch as th
 
from flax.training.common_utils import shard_prng_key
import numpy as np
from PIL import Image
from tqdm import tqdm
import urllib.request

"""Load Enriched prompts dictionary"""

### HAND-CRAFTED PROMPTS

# with open('Scrivania/Tesi Cavallaro/Files/enriched_prompts.json') as json_file:
#   enriched_prompts = json.load(json_file)

with open('Scrivania/Tesi Cavallaro/Files/enriched_prompts_5099.json') as json_file:
  enriched_prompts = json.load(json_file)

"""Generation"""

def er_dalle(prompt_dict, save_path, model_name, n_predictions, gen_top_k, gen_top_p, temperature, cond_scale, key):
    
  # 1) Create model folder if not exists
  modelpath = "".join([save_path, model_name, '/'])
  if not os.path.exists(modelpath):
    os.mkdir(modelpath)

  # Iterate over the URLs
  for URL in prompt_dict.keys():
      
    # Free up GPU memory
    gc.collect()
    th.cuda.empty_cache()
    
    # 2.1) Creates URL folder
    urlpath = "".join([modelpath, str(re.split(r'[_.]', URL)[-2]), '/'])
    if not os.path.exists(urlpath):
      os.mkdir(urlpath)

    # 2.2) Extract the prompts
    prompt = prompt_dict[URL]

    print(URL, prompt)

    # Preprocess the prompt
    tokenized_prompts = processor([prompt])
    tokenized_prompt = replicate(tokenized_prompts)

    for i in tqdm(range(max(n_predictions // jax.device_count(), 1))):
      # get a new key
      key, subkey = jax.random.split(key)
      # generate images
      encoded_images = p_generate(
        tokenized_prompt,
        shard_prng_key(subkey),
        params,
        gen_top_k,
        gen_top_p,
        temperature,
        cond_scale,
      )
      # remove BOS
      encoded_images = encoded_images.sequences[..., 1:]
      # decode images
      decoded_images = p_decode(encoded_images, vqgan_params)
      decoded_images = decoded_images.clip(0.0, 1.0).reshape((-1, 256, 256, 3))
      for decoded_img in decoded_images:
        img = Image.fromarray(np.asarray(decoded_img * 255, dtype=np.uint8))
        imgname = str(re.split(r'[_.]', URL)[-2] + '_' + str(i))
        img.save("".join([urlpath, imgname, '.jpg']))
        
# Define inference parameters (see https://huggingface.co/blog/how-to-generate)
n_predictions = 10
gen_top_k = None
gen_top_p = None
temperature = None
cond_scale = 10.0

# Setup paths and call the generation function
imgsavepath = 'Immagini/Enriched Prompts/'
er_dalle(enriched_prompts, imgsavepath, 'Dalle', n_predictions, gen_top_k, gen_top_p, temperature, cond_scale, key)