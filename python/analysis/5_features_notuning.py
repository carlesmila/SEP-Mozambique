#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 6 2024

Pure feature extraction from a pre-trained CNN (HPC)

@author: cmila
"""

# Setup
import os
import pandas as pd
import numpy as np

# Torch and CV imports
import torch
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.models import vgg16, VGG16_Weights


#%% Prepare tabular data 

# Read tabular data and add a mock outcome to load the data
datadf = pd.read_csv('data/clean/quest_clean.csv')
datadf['out'] = 1.


#%% Model and transforms

from torchvision.models import vgg16, VGG16_Weights

# Data transforms
data_transforms = { 
    'all': transforms.Compose([ # Just normalization for validation        
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(), # This one scales between [0 and 1]
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# Model 
model = vgg16(weights='IMAGENET1K_V1')
print(model)

# Freeze all model parameters
for param in model.parameters():
    param.requires_grad = False

# Delete output layers
model.classifier = torch.nn.Sequential(*[model.classifier[i] for i in range(4)])
print(model.classifier)

# Print modified model and check trainable parameters
print(model)
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
sum([np.prod(p.size()) for p in model_parameters]) # Should be 0

# Eval model
model.eval()

#%% Prepare custom dataset

# Custome dataset class for SEP data
class SEPDataset(Dataset):
    """SEP dataset."""
    
    def __init__(self, datadf, transform=None):
        """
        Arguments:
            datadf (data frame): Data frame containing householdIDs (col0), 
                                imagepaths (col1), and outcomes (successive cols).
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.datadf = datadf
        self.transform = transform
    
    def __len__(self):
        return len(self.datadf)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        image = Image.open(self.datadf.iloc[idx, 1])
        label = self.datadf.iloc[idx, 2]
        label = np.array([float(label)], dtype = np.float32)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

#%% Function to extract features without finetuning
def extract_features_nofinetuning(imgtype, datadf, model, transform):
    '''
    Uses a pre-trained model to extract features for the last fully connected
    layer (VGG16) and saves them to disk
    
    Parameters:
        imgtype (str): The string indicating the image type.
        datadf (dataframe): dataframe where photograph paths can be found
        model (torch): pretrained model to use
        transform (dict): data transform to apply to data
    '''
    
    # Prep dataset and loaders
    data_prepf = datadf[['household', imgtype, 'out']] # Take relevant columns
    data_prepf = data_prepf.dropna(subset=[imgtype]) # Delete missing
    data_set = SEPDataset(data_prepf, transform=transform)
    data_loader = DataLoader(data_set, batch_size=1, shuffle=False)
    print('Image type: ' + imgtype)
    print('Data has {} instances'.format(len(data_set)))
    
    # Extract and report every 100 images
    featuresdf = pd.DataFrame(columns=['household'] + ['feat' + str(i+1) for i in range(4096)])
    featuresdf['household'] = data_prepf['household']
    for i, tdata in enumerate(data_loader):
        if i%100 == 0:
            print(i)
        inputs, labels = tdata
        outputs = model(inputs).numpy().squeeze()
        featuresdf.iloc[i,1:] = outputs
        
    # Write to disks
    featuresdf.to_csv('output/features_notuning/' + imgtype + '_features.csv', index=False)


#%% Apply
extract_features_nofinetuning('foto1', datadf, model, data_transforms['all'])
extract_features_nofinetuning('foto2', datadf, model, data_transforms['all'])
extract_features_nofinetuning('foto3', datadf, model, data_transforms['all'])
extract_features_nofinetuning('foto4', datadf, model, data_transforms['all'])
extract_features_nofinetuning('foto5', datadf, model, data_transforms['all'])
extract_features_nofinetuning('foto6', datadf, model, data_transforms['all'])
extract_features_nofinetuning('foto7', datadf, model, data_transforms['all'])
extract_features_nofinetuning('foto8', datadf, model, data_transforms['all'])
extract_features_nofinetuning('foto9', datadf, model, data_transforms['all'])
extract_features_nofinetuning('foto10', datadf, model, data_transforms['all'])
extract_features_nofinetuning('foto11', datadf, model, data_transforms['all'])
extract_features_nofinetuning('satellite25', datadf, model, data_transforms['all'])
extract_features_nofinetuning('satellite100', datadf, model, data_transforms['all'])
