# -*- coding: utf-8 -*-

"""Imports"""

import pandas as pd
import numpy as np
from scipy import stats

import re
import os
import gc
import itertools

from PIL import Image
import urllib.request

import torch
import clip

import pickle

"""Average CLIP score computation"""

def get_clip_score(original_image_path, generated_image_path):
  '''
    original_image_path -> path to the original image (i.e. COCO URL)
    generated_image_path -> path to the generated image (i.e. path to a local folder)
  '''
  # Load the pre-trained CLIP model and the image
  model, preprocess = clip.load('ViT-B/32')

  with urllib.request.urlopen(original_image_path) as url:
      original_image = Image.open(url)
  generated_image = Image.open(generated_image_path)

  # Preprocess the image and tokenize the text
  original_image_input = preprocess(original_image).unsqueeze(0)
  generated_image_input = preprocess(generated_image).unsqueeze(0)

  # Move the inputs to GPU if available
  device = "cuda" if torch.cuda.is_available() else "cpu"
  original_image_input = original_image_input.to(device)
  generated_image_input = generated_image_input.to(device)
  model = model.to(device)

  # Generate embeddings for the image and text
  with torch.no_grad():
    original_image_features = model.encode_image(original_image_input)
    generated_image_features = model.encode_image(generated_image_input)

    # Normalize the features
    original_image_features = original_image_features / original_image_features.norm(dim=-1, keepdim=True)
    generated_image_features = generated_image_features / generated_image_features.norm(dim=-1, keepdim=True)

    # Calculate the cosine similarity to get the CLIP score
    clip_score = torch.matmul(original_image_features, generated_image_features.T).item()

  return clip_score

def averageCLIPcalculator(gen_image_path, dataset, model_name):
  '''
    gen_image_path -> path to the folder containing the generated images
    dataset -> dataset containing the URLs to COCO images
    model_name -> name of the T2I model to process
  '''
  
  # Creates the path to the current model (i.e. Immagini/modello/)
  model_path = "".join([gen_image_path, model_name,'/'])

  # Creates the dictionary where to store the scores for each image
  model_dict = {}

  # Creates the dictionary where to store every single CLIP score
  full_scores = {}

  # Extract the list of the URLs in the dataset
  url_list = dataset['Image_URL'].tolist()

  for URL in url_list:
    
    print(URL)
    
    # Extract original image ID in order to choose the correct directory (i.e. Immagini/modello/codice/)
    original_image_id = re.split(r'[_.]', URL)[-2]
    dir_path = "".join([model_path, original_image_id,'/'])

    # Dictionary where to store average score
    scores_dict = {}

    # Dictionary where to store every score
    every_score = {}

    # Iterate over the 8 T2I models
    for file in os.listdir(dir_path):
      curr_path = os.path.join(dir_path, file) # (i.e. Immagini/modello/codice/OFA_Huge/)
      mod_scores = []
      # Iterate over the 10 images generated for each model
      for filename in os.scandir(curr_path):
        mod_scores.append(get_clip_score(URL, filename.path))
      every_score[filename.name.split('_', maxsplit=1)[1].rsplit('_', maxsplit=1)[0]] = mod_scores
      # print(every_score)
      scores_dict[filename.name.split('_', maxsplit=1)[1].rsplit('_', maxsplit=1)[0]] = np.average(mod_scores)

    full_scores[original_image_id] = every_score
    model_dict[URL] = scores_dict
    print(model_dict)

  # Convert to dataframe
  df = pd.DataFrame.from_dict(model_dict, orient='index')
  return df, full_scores

# Loading the dataset

all_captions = pd.read_csv('Scrivania/Tesi Cavallaro/Files/all_captions.csv')
# If needed, choose a subset to process
subset = all_captions.loc[50:99, :]

# Saving the results
# We are saving a 5 .csv files, one for each T2I model, containing the average CLIP scores for each captioning model
# We are also saving 5 dictionaries containing ALL the CLIP scores (i.e. the score for each image with respect to the original)

# imgsavepath = 'Immagini/'
# averageCLIP_SD, full_scores_dict_SD = averageCLIPcalculator(imgsavepath, subset, 'StableDiffusion2.1')
# averageCLIP_SD.to_csv("Scrivania/Tesi Cavallaro/Files/Average CLIP scores/average_SD_5099.csv")

# # # Save dictionary to pkl file
# with open('Scrivania/Tesi Cavallaro/Files/Average CLIP scores/full_scores_SD_5099.pkl', 'wb') as fp:
#     pickle.dump(full_scores_dict_SD, fp)

##########################################################################################

# imgsavepath = 'Immagini/'
# averageCLIP_Kan, full_scores_dict_Kan = averageCLIPcalculator(imgsavepath, subset, 'Kandinsky2.1')
# averageCLIP_Kan.to_csv("Scrivania/Tesi Cavallaro/Files/Average CLIP scores/average_Kandinsky_5099.csv")

# # Save dictionary to pkl file
# with open('Scrivania/Tesi Cavallaro/Files/Average CLIP scores/full_scores_Kandinsky_5099.pkl', 'wb') as fp:
#     pickle.dump(full_scores_dict_Kan, fp)

##########################################################################################

# imgsavepath = 'Immagini/'
# averageCLIP_Gli, full_scores_dict_Gli = averageCLIPcalculator(imgsavepath, subset, 'Glide')
# averageCLIP_Gli.to_csv("Scrivania/Tesi Cavallaro/Files/Average CLIP scores/average_Glide_5099.csv")

# # Save dictionary to pkl file
# with open('Scrivania/Tesi Cavallaro/Files/Average CLIP scores/full_scores_Glide_5099.pkl', 'wb') as fp:
#     pickle.dump(full_scores_dict_Gli, fp)

##########################################################################################

# imgsavepath = 'Immagini/'
# averageCLIP_Dal, full_scores_dict_Dal = averageCLIPcalculator(imgsavepath, subset, 'Dalle')
# averageCLIP_Dal.to_csv("Scrivania/Tesi Cavallaro/Files/Average CLIP scores/average_Dalle_5099.csv")

# # Save dictionary to pkl file
# with open('Scrivania/Tesi Cavallaro/Files/Average CLIP scores/full_scores_Dalle_5099.pkl', 'wb') as fp:
#     pickle.dump(full_scores_dict_Dal, fp)

##########################################################################################

imgsavepath = 'Immagini/'
averageCLIP_Dream, full_scores_dict_Dream = averageCLIPcalculator(imgsavepath, subset, 'DreamlikeArt')
averageCLIP_Dream.to_csv("Scrivania/Tesi Cavallaro/Files/Average CLIP scores/average_DreamlikeArt_5099.csv")

# Save dictionary to pkl file
with open('Scrivania/Tesi Cavallaro/Files/Average CLIP scores/full_scores_DreamlikeArt_5099.pkl', 'wb') as fp:
    pickle.dump(full_scores_dict_Dream, fp)