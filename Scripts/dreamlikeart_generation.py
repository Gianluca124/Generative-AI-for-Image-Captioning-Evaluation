# -*- coding: utf-8 -*-

# If errors are raised while running the script make sure you are logged in HuggingFace:
#   1) In terminal run: huggingface-cli login
#   2) Copy the authorization token in present in the given link and paste it in the terminal

"""Imports"""

import pandas as pd
import re
import os
import gc

from PIL import Image
import urllib.request

import torch
from diffusers import StableDiffusionPipeline

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
        image = pipe(prompt, generator=generator).images[0]
        imgname = str(re.split(r'[_.]', URL)[-2] + '_' + key + '_' + str(i))
        image.save("".join([promptpath, imgname, '.jpg']))
        
        # Free up GPU memory
        gc.collect()
        torch.cuda.empty_cache()

"""Dataset"""

all_captions = pd.read_csv('Scrivania/Tesi Cavallaro/Files/all_captions.csv')

# Choose which caption to process
# If needed, choose a subset to process
subset = all_captions.loc[96:96, :]

"""# Dreamlike-art photoreal"""

pipe = StableDiffusionPipeline.from_pretrained("dreamlike-art/dreamlike-photoreal-2.0", torch_dtype=torch.float16, use_auth_token=True)
pipe = pipe.to("cuda")
pipe.enable_attention_slicing()

generator = torch.Generator("cuda").manual_seed(3052023)

# Setup paths and call the generation function
imgsavepath = 'Immagini/'
prompt_to_image(subset, imgsavepath, 'DreamlikeArt', 10)