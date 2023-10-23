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
  
  # Creates the path to the current model (i.e. Immagini/Enriched Prompts/modello/ or Immagini/ChatGPT Enriched Prompts/modello/)
  model_path = "".join([gen_image_path, model_name,'/'])

  # Creates the dictionary where to store the scores for each image
  # model_dict = {}

  # Creates the dictionary where to store every single CLIP score
  full_scores = {}

  # Extract the list of the URLs in the dataset
  url_list = dataset['Image_URL'].tolist()

  for URL in url_list:
    
    print(URL)
    
    # Extract original image ID in order to choose the correct directory (i.e. Immagini/Enriched Prompts/modello/codice/ or Immagini/ChatGPT Enriched Prompts/modello/codice)
    original_image_id = re.split(r'[_.]', URL)[-2]
    dir_path = "".join([model_path, original_image_id,'/'])

    # Dictionary where to store average score
    # scores_dict = {}

    # Dictionary where to store every score
    # every_score = {}
    
    url_scores = {}
    
    # Iterate over the images in each URL folder
    for filename in os.scandir(dir_path):
      url_scores[filename.name.split('_')[1].split('.')[0]] = get_clip_score(URL, filename.path)        
    full_scores[original_image_id] = url_scores
    
    print(full_scores)
  
  return full_scores
    
# Loading the dataset

all_captions = pd.read_csv('Scrivania/Tesi Cavallaro/Files/all_captions.csv')
# If needed, choose a subset to process
subset = all_captions.loc[:99, :]
subset

# Saving the results
# We are saving a .pkl containing the scores

imgsavepath = 'Immagini/Enriched Prompts/'
full_scores_sd = averageCLIPcalculator(imgsavepath, subset, 'StableDiffusion2.1')
with open('Scrivania/Tesi Cavallaro/Files/Average CLIP scores [ENRICHED]/enriched_scores_SD.pkl', 'wb') as fp:
    pickle.dump(full_scores_sd, fp)

##########################################################################################

imgsavepath = 'Immagini/Enriched Prompts/'
full_scores_kan = averageCLIPcalculator(imgsavepath, subset, 'Kandinsky')
with open('Scrivania/Tesi Cavallaro/Files/Average CLIP scores [ENRICHED]/enriched_scores_Kand.pkl', 'wb') as fp:
    pickle.dump(full_scores_kan, fp)

##########################################################################################

imgsavepath = 'Immagini/Enriched Prompts/'
full_scores_gli = averageCLIPcalculator(imgsavepath, subset, 'Glide')
with open('Scrivania/Tesi Cavallaro/Files/Average CLIP scores [ENRICHED]/enriched_scores_glide.pkl', 'wb') as fp:
    pickle.dump(full_scores_gli, fp)

##########################################################################################

imgsavepath = 'Immagini/Enriched Prompts/'
full_scores_dalle = averageCLIPcalculator(imgsavepath, subset, 'Dalle')
with open('Scrivania/Tesi Cavallaro/Files/Average CLIP scores [ENRICHED]/enriched_scores_dalle.pkl', 'wb') as fp:
    pickle.dump(full_scores_dalle, fp)

##########################################################################################

imgsavepath = 'Immagini/Enriched Prompts/'
full_scores_dream = averageCLIPcalculator(imgsavepath, subset, 'DreamlikeArt')
with open('Scrivania/Tesi Cavallaro/Files/Average CLIP scores [ENRICHED]/enriched_scores_dream.pkl', 'wb') as fp:
    pickle.dump(full_scores_dream, fp)