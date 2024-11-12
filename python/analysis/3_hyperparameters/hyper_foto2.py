#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sept 3 2024

CNN hyperparameter tuning (HPC)
Photo 2: Floor

@author: cmila
"""

# Setup
import os
import pandas as pd
import numpy as np
import random
# import matplotlib.pyplot as plt

# Torch and CV imports
import torch
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image


#%% Utils
exec(open('python/analysis/finetuning_utils.py').read())
    
#%% Random hyperparameter search
hyper = pd.read_csv("output/hyperparameters/random-search.csv")
    
#%% Prepare data 

# Read tabular data and take relevant columns
outcomes = pd.read_csv("data/clean/indicators.csv")
outcomes = outcomes[['household', 'exp_cat2', 'inc_cat2', 'assets_cat2']]
photopaths = pd.read_csv('data/clean/quest_clean.csv')
photopaths = photopaths[['household', 'foto2']] 
split = pd.read_csv("data/clean/datasplit.csv")

# Merge
datadf = outcomes.set_index('household').join(photopaths.set_index('household')).join(split.set_index('household'))
del(outcomes, photopaths, split)

# Binary outcomes, delete missing photographs
bindict = {'high': 1,'low': 0}
datadf.exp_cat2 = [bindict[i] for i in datadf.exp_cat2]
datadf.inc_cat2 = [bindict[i] for i in datadf.inc_cat2]
datadf.assets_cat2 = [bindict[i] for i in datadf.assets_cat2]
datadf = datadf.reset_index()

# Train data only
datadf = datadf.loc[datadf['split']=='train']
datadf = datadf[['household', 'foto2', 'exp_cat2', 'inc_cat2', 'assets_cat2']]

# Internal split: 640 train, 160 test
random.seed(1234)
split = sum(([x]*y for x,y in zip(["train", "test"], (640, 160))),[])
split = random.sample(split, 800)
datadf['split'] = split
datadf = datadf.dropna()
traindf = datadf.loc[datadf['split']=='train']
testdf = datadf.loc[datadf['split']=='test']
del(datadf)
    

for it in range(10):
    print("Hyperpameter search: " + str(it+1) + "/10")    
    
    #%% Dataset 
    training_set = SEPDataset(traindf, transform=data_transforms['train'])
    test_set = SEPDataset(testdf, transform=data_transforms['test'])

    #%% Dataloader
    bsize = int(hyper.bs[it])
    train_loader = DataLoader(training_set, batch_size=bsize, shuffle=True)
    validation_loader = DataLoader(test_set, batch_size=bsize, shuffle=False)
    print('Training set has {} instances'.format(len(training_set)))
    print('Test set has {} instances'.format(len(test_set)))
    tsteps = len(train_loader)
    vsteps = len(validation_loader)

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
    
    #%% Prepare training

    # Loss function
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean')

    # Optimization loop
    def train_one_epoch():
        
        for i, data in enumerate(train_loader):
            # Every data instance is an input + label pair
            inputs, labels = data
            
            # Zero your gradients for every batch!
            optimizer.zero_grad()
            
            # Make predictions for this batch
            outputs = model(inputs)
            
            # Compute the loss and its gradients
            loss = loss_fn(outputs, labels)
            loss.backward()
            
            # Adjust learning weights
            optimizer.step()
        
        
    #%% Run model
    epoch_number = 0
    EPOCHS = 50 
    optimizer = torch.optim.SGD(model.parameters(), 
                                lr=float(hyper.lr[it]), 
                                momentum=float(hyper.mom[it]), 
                                weight_decay=float(hyper.l2[it]))

    epoch_record = pd.DataFrame(columns=['epoch', 
                                         'train_loss', 
                                         'train_exp_acc', 'train_inc_acc', 'train_assets_acc',
                                         'val_loss', 
                                         'val_exp_acc', 'val_inc_acc', 'val_assets_acc'])

    for epoch in range(EPOCHS+1):
        
        if epoch > 0:
            print('EPOCH {}'.format(epoch_number))       
            # Make sure gradient tracking is on, and do a pass over the data
            model = model.train()
            train_one_epoch()
        else:
            print('EPOCH 0: Pre-training') 
        
        
        # Set the model to evaluation mode, disabling dropout and using population 
        # statistics for batch normalization.
        model = model.eval()
        running_tloss = 0.0
        running_tacc_exp = 0.0
        running_tacc_inc = 0.0
        running_tacc_assets = 0.0
        running_vloss = 0.0
        running_vacc_exp = 0.0
        running_vacc_inc = 0.0
        running_vacc_assets = 0.0
        
        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, tdata in enumerate(train_loader):
                tinputs, tlabels = tdata
                toutputs = model(tinputs)
                tloss = loss_fn(toutputs, tlabels)
                running_tloss += tloss 
                # Threshold 0 because we need to apply the sigmoid function to convert to probs
                running_tacc_exp += torch.sum(((toutputs[:,0] >= 0).to(torch.double) == tlabels[:,0]).to(torch.double))
                running_tacc_inc += torch.sum(((toutputs[:,1] >= 0).to(torch.double) == tlabels[:,1]).to(torch.double))
                running_tacc_assets += torch.sum(((toutputs[:,2] >= 0).to(torch.double) == tlabels[:,2]).to(torch.double))
        
        tot_tloss = running_tloss / tsteps
        tot_tacc_exp = running_tacc_exp / len(training_set) 
        tot_tacc_inc = running_tacc_inc / len(training_set)
        tot_tacc_assets = running_tacc_assets / len(training_set)
        
        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                vinputs, vlabels = vdata
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss
                # Threshold 0 because we need to apply the sigmoid function to convert to probs
                running_vacc_exp += torch.sum(((voutputs[:,0] >= 0).to(torch.double) == vlabels[:,0]).to(torch.double))
                running_vacc_inc += torch.sum(((voutputs[:,1] >= 0).to(torch.double) == vlabels[:,1]).to(torch.double))
                running_vacc_assets += torch.sum(((voutputs[:,2] >= 0).to(torch.double) == vlabels[:,2]).to(torch.double))
        
        tot_vloss = running_vloss / vsteps
        tot_vacc_exp = running_vacc_exp / len(test_set) 
        tot_vacc_inc = running_vacc_inc / len(test_set)
        tot_vacc_assets = running_vacc_assets / len(test_set)
        
        print('LOSS train {} valid {}'.format(tot_tloss, tot_vloss))
        print('ACC expenditure train {} valid {}'.format(tot_tacc_exp, tot_vacc_exp)) 
        print('ACC income train {} valid {}'.format(tot_tacc_inc, tot_vacc_inc))    
        print('ACC assets train {} valid {}'.format(tot_tacc_assets, tot_vacc_assets))    
        
        # Running loss averaged per batch and accuracy for both training and validation
        epoch_report = {'epoch': epoch_number, 
                        'train_loss': float(tot_tloss), 
                        'train_exp_acc': float(tot_tacc_exp), 
                        'train_inc_acc': float(tot_tacc_inc), 
                        'train_assets_acc': float(tot_tacc_assets), 
                        'val_loss': float(tot_vloss),
                        'val_exp_acc': float(tot_vacc_exp), 
                        'val_inc_acc': float(tot_vacc_inc),
                        'val_assets_acc': float(tot_vacc_assets)
                        }
        epoch_record.loc[epoch] = epoch_report    
        
        epoch_number += 1
        
    epoch_record.to_csv('output/hyperparameters/foto2_hyper' + str(it) + '.csv', index = False)
    
