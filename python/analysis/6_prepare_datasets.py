#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Feb 8 2024

Prepare datasets for regression analyses, one with features from the fine-tuned
CNNs and another with features extracted from the CNN without finetuning

@author: cmila
"""

# Setup
import os
import pandas as pd


#%% Fine-tuned

# Outcomes
outcomes = pd.read_csv("data/clean/indicators.csv")
outcomes = outcomes.loc[:, ['household', 'exp_all', 'inc_all', 'assets']]
outcomes.columns = ['household', 'exp_true', 'inc_true', 'assets_true']

# Data split
split = pd.read_csv("data/clean/datasplit.csv")
alldf = outcomes.merge(split, how = 'outer', on = 'household')
traindf = alldf.loc[alldf['split'] == "train"]
testdf = alldf.loc[alldf['split'] == "test"]

# Image data
imgtype = ['foto' + str(k+1) for k in range(11)] + ['satellite25', 'satellite100']
for it in imgtype:
    
    print('Processing ' + it)
    
    # Read features
    it_df = pd.read_csv('output/features_tuning/' + it + '_features.csv')
    it_df.columns = ['household'] + [it + '_' + str(k+1) for k in range(30)]
    it_traindf = it_df.loc[it_df['household'].isin(traindf['household'])]
    it_testdf = it_df.loc[it_df['household'].isin(testdf['household'])]
    
    # Stack
    if it=="foto1":
        alltrain_df = it_traindf.copy()
        alltest_df = it_testdf.copy()
    else:
        alltrain_df = alltrain_df.merge(it_traindf, how = 'outer', on = 'household')
        alltest_df = alltest_df.merge(it_testdf, how = 'outer', on = 'household')
        
    del(it_df, it_traindf, it_testdf)


# Join tables (inner because we need outcome and predictors), split
traindf = traindf.merge(alltrain_df, how = 'inner', on = 'household')
testdf = testdf.merge(alltest_df, how = 'inner', on = 'household')

# Write to disk and clean
traindf.to_csv('output/supervised_tuning/traindata.csv', index = False)
testdf.to_csv('output/supervised_tuning/testdata.csv', index = False)
del(alldf, alltest_df, alltrain_df, imgtype, it, outcomes, split, testdf, traindf)



#%% Not fine-tuned

# Outcomes
outcomes = pd.read_csv("data/clean/indicators.csv")
outcomes = outcomes.loc[:, ['household', 'exp_all', 'inc_all', 'assets']]
outcomes.columns = ['household', 'exp_true', 'inc_true', 'assets_true']

# Data split
split = pd.read_csv("data/clean/datasplit.csv")
alldf = outcomes.merge(split, how = 'outer', on = 'household')
traindf = alldf.loc[alldf['split'] == "train"]
testdf = alldf.loc[alldf['split'] == "test"]

# Image data
imgtype = ['foto' + str(k+1) for k in range(11)] + ['satellite25', 'satellite100']
for it in imgtype:
    
    print('Processing ' + it)
    
    # Read features
    it_df = pd.read_csv('output/features_notuning/' + it + '_features.csv')
    it_df.columns = ['household'] + [it + '_' + str(k+1) for k in range(4096)]
    it_traindf = it_df.loc[it_df['household'].isin(traindf['household'])]
    it_testdf = it_df.loc[it_df['household'].isin(testdf['household'])]
    
    # Stack
    if it=="foto1":
        alltrain_df = it_traindf.copy()
        alltest_df = it_testdf.copy()
    else:
        alltrain_df = alltrain_df.merge(it_traindf, how = 'outer', on = 'household')
        alltest_df = alltest_df.merge(it_testdf, how = 'outer', on = 'household')
        
    del(it_df, it_traindf, it_testdf)


# Join tables (inner because we need outcome and predictors), split
traindf = traindf.merge(alltrain_df, how = 'inner', on = 'household')
testdf = testdf.merge(alltest_df, how = 'inner', on = 'household')

# Write to disk and clean
traindf.to_csv('output/supervised_notuning/traindata.csv', index = False)
testdf.to_csv('output/supervised_notuning/testdata.csv', index = False)
del(alldf, alltest_df, alltrain_df, imgtype, it, outcomes, split, testdf, traindf)
