# -*- coding: utf-8 -*-

"""Imports"""

import pandas as pd
import json
import re
import os
import gc

from PIL import Image
import urllib.request

import torch as th
import torchvision.transforms as T

from glide_text2im.download import load_checkpoint
from glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler
)

"""Setup"""
has_cuda = th.cuda.is_available()
device = th.device('cpu' if not has_cuda else 'cuda')

"""Model creation"""

# BASE MODEL
options = model_and_diffusion_defaults()
options['use_fp16'] = has_cuda
options['timestep_respacing'] = '100' # 100 diffusion steps for fast sampling
model, diffusion = create_model_and_diffusion(**options)
model.eval()
if has_cuda:
    model.convert_to_fp16()
model.to(device)
model.load_state_dict(load_checkpoint('base', device))
print('Total base parameters', sum(x.numel() for x in model.parameters()))

# UPSAMPLER MODEL
options_up = model_and_diffusion_defaults_upsampler()
options_up['use_fp16'] = has_cuda
options_up['timestep_respacing'] = 'fast27' # 27 diffusion steps for very fast sampling
model_up, diffusion_up = create_model_and_diffusion(**options_up)
model_up.eval()
if has_cuda:
    model_up.convert_to_fp16()
model_up.to(device)
model_up.load_state_dict(load_checkpoint('upsample', device))
print('Total upsampler parameters', sum(x.numel() for x in model_up.parameters()))

"""Function definition"""

def generate_sample(prompt, batch_size, guidance_scale):
  # Create the text tokens to feed to the model.
  tokens = model.tokenizer.encode(prompt)
  tokens, mask = model.tokenizer.padded_tokens_and_mask(
      tokens, options['text_ctx']
      )

  # Create the classifier-free guidance tokens (empty)
  full_batch_size = batch_size * 2
  uncond_tokens, uncond_mask = model.tokenizer.padded_tokens_and_mask(
      [], options['text_ctx']
      )

  # Pack the tokens together into model kwargs.
  model_kwargs = dict(
      tokens=th.tensor(
          [tokens] * batch_size + [uncond_tokens] * batch_size, device=device
          ),
          mask=th.tensor(
              [mask] * batch_size + [uncond_mask] * batch_size,
              dtype=th.bool,
              device=device,
          ),
      )

  # Create a classifier-free guidance sampling function
  def model_fn(x_t, ts, **kwargs):
    half = x_t[: len(x_t) // 2]
    combined = th.cat([half, half], dim=0)
    model_out = model(combined, ts, **kwargs)
    eps, rest = model_out[:, :3], model_out[:, 3:]
    cond_eps, uncond_eps = th.split(eps, len(eps) // 2, dim=0)
    half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
    eps = th.cat([half_eps, half_eps], dim=0)
    return th.cat([eps, rest], dim=1)

  # Sample from the base model.
  model.del_cache()
  samples = diffusion.p_sample_loop(
      model_fn,
      (full_batch_size, 3, options["image_size"], options["image_size"]),
      device=device,
      clip_denoised=True,
      progress=True,
      model_kwargs=model_kwargs,
      cond_fn=None,
  )[:batch_size]
  model.del_cache()
  return samples

def upsample(samples, prompt, batch_size, upsample_temp):
  tokens = model_up.tokenizer.encode(prompt)
  tokens, mask = model_up.tokenizer.padded_tokens_and_mask(
      tokens, options_up['text_ctx']
      )

  # Create the model conditioning dict.
  model_kwargs = dict(
      # Low-res image to upsample.
      low_res=((samples+1)*127.5).round()/127.5 - 1,

      # Text tokens
      tokens=th.tensor(
          [tokens] * batch_size, device=device
      ),
      mask=th.tensor(
          [mask] * batch_size,
          dtype=th.bool,
          device=device,
      ),
  )

  # Sample from the base model.
  model_up.del_cache()
  up_shape = (batch_size, 3, options_up["image_size"], options_up["image_size"])
  up_samples = diffusion_up.ddim_sample_loop(
      model_up,
      up_shape,
      noise=th.randn(up_shape, device=device) * upsample_temp,
      device=device,
      clip_denoised=True,
      progress=True,
      model_kwargs=model_kwargs,
      cond_fn=None,
  )[:batch_size]
  model_up.del_cache()
  return up_samples

def save_image(batch: th.Tensor, path):
    scaled = ((batch + 1)*127.5).round().clamp(0,255).to(th.uint8).cpu()
    reshaped = scaled.permute(2, 0, 3, 1).reshape([batch.shape[2], -1, 3])
    img = Image.fromarray(reshaped.numpy())
    img.save("".join([path, '.jpg']))

def er_glide(prompt_dict, save_path, model_name, batch_size, guidance_scale, upsample_temp, n):
  '''
    prompt_dict -> dictionary containing link to the COCO images and the corresponding enriched caption
    save_path -> path to the root folder where to save the images
    model_name -> name to give to the folder where the images will be saved
    batch_size, guidance_scale, upscale_temp -> generation parameters
  '''

  # 1) Create model folder if not exists
  modelpath = "".join([save_path, model_name, '/'])
  if not os.path.exists(modelpath):
    os.mkdir(modelpath)

  # 2) Iterate over the URLs
  for URL in prompt_dict.keys():
      
    # 2.1) Creates URL folder
    urlpath = "".join([modelpath, str(re.split(r'[_.]', URL)[-2]), '/'])
    if not os.path.exists(urlpath):
      os.mkdir(urlpath)

    # Free up GPU memory
    gc.collect()
    th.cuda.empty_cache()

    # 2.2) Extract the prompts
    prompt = prompt_dict[URL]

    print(URL, prompt)

    # 2.3) Generate n images per prompt
    for i in range(0, n):
        image = generate_sample(prompt, batch_size, guidance_scale)
        upsample_image = upsample(image, prompt, batch_size, upsample_temp)
        imgname = str(re.split(r'[_.]', URL)[-2]  + '_' + str(i))
        image_path = "".join([urlpath, imgname])
        save_image(upsample_image, image_path)
        
        # Free up GPU memory
        gc.collect()
        th.cuda.empty_cache()
        
"""Load Enriched prompts dictionary"""

### HAND-CRAFTED PROMPTS

# with open('Scrivania/Tesi Cavallaro/Files/enriched_prompts.json') as json_file:
#   enriched_prompts = json.load(json_file)

with open('Scrivania/Tesi Cavallaro/Files/enriched_prompts_5099.json') as json_file:
  enriched_prompts = json.load(json_file)
  
"""Generation"""

batch_size = 1
guidance_scale = 5.0
upsample_temp = 0.998

imgsavepath = 'Immagini/Enriched Prompts/'
er_glide(enriched_prompts, imgsavepath, 'Glide', batch_size, guidance_scale, upsample_temp, 10)