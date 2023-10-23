# -*- coding: utf-8 -*-

"""Imports"""

import pandas as pd
import re
import os
import gc

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

"""Function definition"""

def prompt_to_image(dataset, save_path, model_name, n):
  '''
    dataset -> dataset containing link to the COCO images and the corresponding captions
    save_path -> path to the root folder where to save the images
    model_name -> name to give to the folder where the images will be saved
  '''
  
  # 1) Extract the list of URLs in the dataset (each URL correspond to a different image)
  url_list = dataset['Image_URL'].tolist()

  # 2) Create model folder if not exists
  modelpath = "".join([save_path, model_name, '/'])
  if not os.path.exists(modelpath):
    os.mkdir(modelpath)

  # 3) Iterate over the URLs
  for URL in url_list:
    
    # Free up GPU memory
    gc.collect()
    torch.cuda.empty_cache()

    # 3.1) Print the processed URL to keep track of the running
    print(URL)
    
    # 3.2) Creates URL folder
    urlpath = "".join([modelpath, str(re.split(r'[_.]', URL)[-2]), '/'])
    if not os.path.exists(urlpath):
      os.mkdir(urlpath)
    
    # 3.3) Extract a dictionary with the I2T models as keys and the corresponding prompts as values
    prompt_dict = dataset.loc[dataset['Image_URL']==URL, dataset.columns!='Image_URL'].to_dict(orient='records')[0]

    # 3.4) Iterate over I2T models
    for key in prompt_dict.keys():
      
      # 3.4.1) Creates a folder for each prompt (i.e. I2T model)
      promptpath = "".join([urlpath, str(key), '/'])
      if not os.path.exists(promptpath):
        os.mkdir(promptpath)
      
      # 3.4.2) Extract the current prompt
      prompt = prompt_dict[key]
      print(key, prompt)
      
      # 3.4.3) Generate n images per prompt
      for i in range(0, n):
        image = model.generate_text2img(prompt=prompt,
                                        num_steps=100, 
                                        batch_size=1, 
                                        guidance_scale=4,
                                        h=256, w=256,
                                        sampler='p_sampler', 
                                        prior_cf_scale=4,
                                        prior_steps="5",)
        
        imgname = str(re.split(r'[_.]', URL)[-2] + '_' + key + '_' + str(i))
        image[0].save("".join([promptpath, imgname, '.jpg']))
        
        # Free up GPU memory
        gc.collect()
        torch.cuda.empty_cache()

"""Dataset"""

all_captions = pd.read_csv('Scrivania/Tesi Cavallaro/Files/all_captions.csv')

# Choose which caption to process
# If needed, choose a subset to process
subset = all_captions.loc[50:99, :]

# Setup paths and call the generation function
imgsavepath = 'Immagini/'
prompt_to_image(subset, imgsavepath, 'Kandinsky2.1', 10)
