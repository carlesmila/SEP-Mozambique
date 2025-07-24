#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 09:24:31 2024

SHAP analysis of complete random forest model

@author: cmila
"""

# Setup
import os
import pandas as pd
import numpy as np
import joblib
import shap


#%% Prep

# Read train data
traindf = pd.read_csv('output/data_tuning/traindata.csv')
X_train = traindf.filter(regex='foto|satellite').to_numpy()

# Read test data
testdf = pd.read_csv('output/data_tuning/testdata.csv')
X_test = testdf.filter(regex='foto|satellite').to_numpy()


#%%  Indices for variable importance estimation
imgtype = ['foto' + str(k+1) for k in range(11)] + ['satellite25', 'satellite100']
coltype = testdf.filter(regex='foto|satellite').columns
coltype = [x.split('_')[0] for x in coltype]
idxs = []
for imgt in imgtype:
    idx = [i for i, val in enumerate(coltype) if val==imgt]
    idxs.append(idx)


#%% Assets

print('Assets')

# Read model and prep explainer
imputer = joblib.load('output/regression_tuning/assets_all_rf.pkl')['imputer']
model = joblib.load('output/regression_tuning/assets_all_rf.pkl')['regressor']
explainer = shap.TreeExplainer(model)

# Shap train
Ximp_train = imputer.transform(X_train)
explanation_train = explainer(Ximp_train)
train_np = explanation_train.values
train_pd = pd.DataFrame({'household': traindf['household']})
for i in range(13):
    imgtype_it = imgtype[i]
    idxs_it = idxs[i]
    train_pd[imgtype_it] = np.sum(train_np[:,idxs_it], axis=1)
train_pd.to_csv('output/shap/assets_train_shap.csv', index = False)
    
    
# Shap test
Ximp_test = imputer.transform(X_test)
explanation_test = explainer(Ximp_test)
test_np = explanation_test.values
test_pd = pd.DataFrame({'household': testdf['household']})
for i in range(13):
    imgtype_it = imgtype[i]
    idxs_it = idxs[i]
    test_pd[imgtype_it] = np.sum(test_np[:,idxs_it], axis=1)
test_pd.to_csv('output/shap/assets_test_shap.csv', index = False)
   

del(imputer, model, explainer, imgtype_it, idxs_it, 
    Ximp_train, explanation_train, train_np, train_pd,
    Ximp_test, explanation_test, test_np, test_pd)



#%% Expenditure

print('Expenditure')

# Read model and prep explainer
imputer = joblib.load('output/regression_tuning/exp_all_rf.pkl')['imputer']
model = joblib.load('output/regression_tuning/exp_all_rf.pkl')['regressor']
explainer = shap.TreeExplainer(model)

# Shap train
Ximp_train = imputer.transform(X_train)
explanation_train = explainer(Ximp_train)
train_np = explanation_train.values
train_pd = pd.DataFrame({'household': traindf['household']})
for i in range(13):
    imgtype_it = imgtype[i]
    idxs_it = idxs[i]
    train_pd[imgtype_it] = np.sum(train_np[:,idxs_it], axis=1)
train_pd.to_csv('output/shap/exp_train_shap.csv', index = False)
    
    
# Shap test
Ximp_test = imputer.transform(X_test)
explanation_test = explainer(Ximp_test)
test_np = explanation_test.values
test_pd = pd.DataFrame({'household': testdf['household']})
for i in range(13):
    imgtype_it = imgtype[i]
    idxs_it = idxs[i]
    test_pd[imgtype_it] = np.sum(test_np[:,idxs_it], axis=1)
test_pd.to_csv('output/shap/exp_test_shap.csv', index = False)
   

del(imputer, model, explainer, imgtype_it, idxs_it, 
    Ximp_train, explanation_train, train_np, train_pd,
    Ximp_test, explanation_test, test_np, test_pd)


#%% Income

print('Income')

# Read model and prep explainer
imputer = joblib.load('output/regression_tuning/inc_all_rf.pkl')['imputer']
model = joblib.load('output/regression_tuning/inc_all_rf.pkl')['regressor']
explainer = shap.TreeExplainer(model)

# Shap train
Ximp_train = imputer.transform(X_train)
explanation_train = explainer(Ximp_train)
train_np = explanation_train.values
train_pd = pd.DataFrame({'household': traindf['household']})
for i in range(13):
    imgtype_it = imgtype[i]
    idxs_it = idxs[i]
    train_pd[imgtype_it] = np.sum(train_np[:,idxs_it], axis=1)
train_pd.to_csv('output/shap/inc_train_shap.csv', index = False)
    
    
# Shap test
Ximp_test = imputer.transform(X_test)
explanation_test = explainer(Ximp_test)
test_np = explanation_test.values
test_pd = pd.DataFrame({'household': testdf['household']})
for i in range(13):
    imgtype_it = imgtype[i]
    idxs_it = idxs[i]
    test_pd[imgtype_it] = np.sum(test_np[:,idxs_it], axis=1)
test_pd.to_csv('output/shap/inc_test_shap.csv', index = False)
   

del(imputer, model, explainer, imgtype_it, idxs_it, 
    Ximp_train, explanation_train, train_np, train_pd,
    Ximp_test, explanation_test, test_np, test_pd)
