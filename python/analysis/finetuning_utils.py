#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 2024

Functions to finetune CNNs 

@author: cmila
"""


#%% Dataset class for SEP data
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
        labels = self.datadf.iloc[idx, [2, 3, 4]].to_numpy(dtype = np.float32) 
        labels = np.array(labels, dtype = np.float32)
        
        if self.transform:
            image = self.transform(image)
        
        return image, labels
    

#%% Model and transforms
from torchvision.models import vgg16, VGG16_Weights
VGG16_Weights.IMAGENET1K_V1.transforms()

# Data transforms
data_transforms = { 
    'train': transforms.Compose([ 
        # Data augmentation
        transforms.RandomResizedCrop(2000),
        transforms.RandomHorizontalFlip(),
        # Common steps
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(), # This one scales between [0 and 1]
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([ # Just normalization for validation        
        # Common steps
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(), # This one scales between [0 and 1]
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}