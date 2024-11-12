#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 2024

Feature extraction from the finetuned CNNs (HPC)

@author: cmila
"""

# Setup
import os
import pandas as pd
import numpy as np
import copy

# Torch and CV imports
import torch
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image



#%% Prepare data 

# Read tabular data and add a mock outcome to load the data
datadf = pd.read_csv('data/clean/quest_clean.csv')
datadf['out1'] = 1.
datadf['out2'] = 1.
datadf['out3'] = 1.


#%% Utils
exec(open('python/analysis/finetuning_utils.py').read())


#%% Model 
model = vgg16(weights='IMAGENET1K_V1')

# Freeze all model parameters
for param in model.parameters():
    param.requires_grad = False

# Edit output layer, 1 output +  weights to be trained
model.classifier[6] = torch.nn.Linear(30, 3)

# Unfreeze layer 3
model.classifier[3] = torch.nn.Linear(4096, 30)

# Print modified model and check trainable parameters
print('Model structure:')
print(model)
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
print('# trainable parameters:')
print(sum([np.prod(p.size()) for p in model_parameters]))


#%% Function to extract features with finetuning
def extract_features_finetuning(imgtype, datadf, mod, transform):
    '''
    Uses a pre-trained model to extract features for the last fully connected
    layer (VGG16) and saves them to disk
    
    Parameters:
        imgtype (str): The string indicating the image type.
        datadf (dataframe): dataframe where photograph paths can be found
        model (torch): pretrained model to use
        transform (dict): data transform to apply to data
    '''
    
    # Freeze, delete last layers
    for param in mod.parameters():
        param.requires_grad = False
    mod.classifier = torch.nn.Sequential(*[mod.classifier[i] for i in range(4)])
    
    # Print modified model and check trainable parameters
    model_parameters = filter(lambda p: p.requires_grad, mod.parameters())
    print(mod)
    mod.eval()
    
    # Prep dataset and loaders
    data_prepf = datadf[['household', imgtype, 'out1', 'out2', 'out3']] # Take relevant columns
    data_prepf = data_prepf.dropna(subset=[imgtype]) # Delete missing
    data_set = SEPDataset(data_prepf, transform=transform)
    data_loader = DataLoader(data_set, batch_size=1, shuffle=False)
    print('Image type: ' + imgtype)
    print('Data has {} instances'.format(len(data_set)))
    
    # Extract and report every 100 images
    featuresdf = pd.DataFrame(columns=['household'] + ['feat' + str(i+1) for i in range(30)])
    featuresdf['household'] = data_prepf['household']
    for i, tdata in enumerate(data_loader):
        if i%100 == 0:
            print(i)
        inputs, labels = tdata
        outputs = mod(inputs).numpy().squeeze()
        featuresdf.iloc[i,1:] = outputs
        
    # Write to disks
    featuresdf.to_csv('output/features_tuning/' + imgtype + '_features.csv', index=False)


#%% Extraction

##%% foto1
model_trans = copy.deepcopy(model)
model_trans.load_state_dict(torch.load('output/finetuning/foto1_weights.pth'), strict=False)
extract_features_finetuning('foto1', datadf, model_trans, data_transforms['test'])
del(model_trans)

##%% foto2
model_trans = copy.deepcopy(model)
model_trans.load_state_dict(torch.load('output/finetuning/foto2_weights.pth'), strict=False)
extract_features_finetuning('foto2', datadf, model_trans, data_transforms['test'])
del(model_trans)

##%% foto3
model_trans = copy.deepcopy(model)
model_trans.load_state_dict(torch.load('output/finetuning/foto3_weights.pth'), strict=False)
extract_features_finetuning('foto3', datadf, model_trans, data_transforms['test'])
del(model_trans)

##%% foto4
model_trans = copy.deepcopy(model)
model_trans.load_state_dict(torch.load('output/finetuning/foto4_weights.pth'), strict=False)
extract_features_finetuning('foto4', datadf, model_trans, data_transforms['test'])
del(model_trans)

##%% foto5
model_trans = copy.deepcopy(model)
model_trans.load_state_dict(torch.load('output/finetuning/foto5_weights.pth'), strict=False)
extract_features_finetuning('foto5', datadf, model_trans, data_transforms['test'])
del(model_trans)

##%% foto6
model_trans = copy.deepcopy(model)
model_trans.load_state_dict(torch.load('output/finetuning/foto6_weights.pth'), strict=False)
extract_features_finetuning('foto6', datadf, model_trans, data_transforms['test'])
del(model_trans)

##%% foto7
model_trans = copy.deepcopy(model)
model_trans.load_state_dict(torch.load('output/finetuning/foto7_weights.pth'), strict=False)
extract_features_finetuning('foto7', datadf, model_trans, data_transforms['test'])
del(model_trans)

##%% foto8
model_trans = copy.deepcopy(model)
model_trans.load_state_dict(torch.load('output/finetuning/foto8_weights.pth'), strict=False)
extract_features_finetuning('foto8', datadf, model_trans, data_transforms['test'])
del(model_trans)

##%% foto9
model_trans = copy.deepcopy(model)
model_trans.load_state_dict(torch.load('output/finetuning/foto9_weights.pth'), strict=False)
extract_features_finetuning('foto9', datadf, model_trans, data_transforms['test'])
del(model_trans)

##%% foto10
model_trans = copy.deepcopy(model)
model_trans.load_state_dict(torch.load('output/finetuning/foto10_weights.pth'), strict=False)
extract_features_finetuning('foto10', datadf, model_trans, data_transforms['test'])
del(model_trans)

##%% foto11
model_trans = copy.deepcopy(model)
model_trans.load_state_dict(torch.load('output/finetuning/foto11_weights.pth'), strict=False)
extract_features_finetuning('foto11', datadf, model_trans, data_transforms['test'])
del(model_trans)

##%% satellite25
model_trans = copy.deepcopy(model)
model_trans.load_state_dict(torch.load('output/finetuning/satellite25_weights.pth'), strict=False)
extract_features_finetuning('satellite25', datadf, model_trans, data_transforms['test'])
del(model_trans)

##%% satellite100
model_trans = copy.deepcopy(model)
model_trans.load_state_dict(torch.load('output/finetuning/satellite100_weights.pth'), strict=False)
extract_features_finetuning('satellite100', datadf, model_trans, data_transforms['test'])
del(model_trans)
