#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Feb 8 2024

Predictive modelling of household SEP (HPC)
Off-the-shelf features, resmapling, RF

@author: cmila
"""

# Setup
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV, cross_validate, KFold, RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, accuracy_score, f1_score
from sklearn.feature_selection import SelectKBest, f_classif
import joblib


#%% Prep data

# Read data
traindf = pd.read_csv('output_post/data_notuning/traindata.csv')
testdf = pd.read_csv('output_post/data_notuning/testdata.csv')
traindf = pd.concat([traindf, testdf], axis=0)
del(testdf)

# Selector hyperparameters
kbest_params = {
    'selector__k': [50, 100, 250, 500, 1000, 2500, 5000]
}

# Model hyperparameters
mod_params = {
    'classifier__min_samples_leaf': [20, 30, 40, 50],
    'classifier__max_features': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
}

# Hyperparameter grid
param_grid = param_grid = {**kbest_params,**mod_params}

# Cross-validation
inner_cv = KFold(n_splits=5)
outer_cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=0)
    
# Scoring
def acc_scorer(y_true, y_pred):
    y_true_bin = [1 if x ==0 else 0 for x in y_true]
    y_pred_bin = [1 if x ==0 else 0 for x in y_pred]
    return accuracy_score(y_true_bin, y_pred_bin)

def f1_scorer(y_true, y_pred):
    return f1_score(y_true, y_pred, average=None)[0]

scoring = {'acc': 'accuracy',
           'acc_class1': make_scorer(acc_scorer),
           'f1_class1': make_scorer(f1_scorer)}

# Pipeline
pipe = Pipeline([
   ('imputer', SimpleImputer()),
   ('selector', SelectKBest(f_classif)),
   ('classifier',  RandomForestClassifier(n_estimators=300, n_jobs=4))
   ])

search = RandomizedSearchCV(pipe, param_grid, n_iter=10, cv=inner_cv, random_state=1234)



#%% Expenditure

print('Expenditure')

# Selecting train y 
y_train = traindf['exp_cat3'].to_numpy()
y_train = [0 if x=='bottom40' else 1 if x=='mid40' else 2 for x in y_train]


#%%% Satellite

print('Satellite')

# train X
X_train = traindf.filter(like='satellite').to_numpy()

# Cross-validate
cv_score = cross_validate(search, X_train, y_train, scoring=scoring, cv=outer_cv)
cv_df = pd.DataFrame(data=cv_score)
cv_df.to_csv('output_post/classification_resampling/exp_satellite_rf.csv', index = False)

# Clean
del(X_train, cv_score, cv_df)


#%%% Outdoor
print('Outdoor')

# train X
X_train = traindf.loc[:,traindf.columns.str.contains('|'.join(['foto10_', 'foto4_', 'foto5_', 'satellite']))].to_numpy()

# Cross-validate
cv_score = cross_validate(search, X_train, y_train,
                          scoring=scoring, cv=outer_cv)
cv_df = pd.DataFrame(data=cv_score)
cv_df.to_csv('output_post/classification_resampling/exp_outdoor_rf.csv', index = False)

# Clean
del(X_train, cv_score, cv_df)


#%%% All
print('All')

# train X
X_train = traindf.filter(regex='foto|satellite').to_numpy()

# Cross-validate
cv_score = cross_validate(search, X_train, y_train,
                          scoring=scoring, cv=outer_cv)
cv_df = pd.DataFrame(data=cv_score)
cv_df.to_csv('output_post/classification_resampling/exp_all_rf.csv', index = False)

# Clean
del(X_train, cv_score, cv_df)


#%% Income

print('Income')

# Selecting train y 
y_train = traindf['inc_cat3'].to_numpy()
y_train = [0 if x=='bottom40' else 1 if x=='mid40' else 2 for x in y_train]

#%%% Satellite

print('Satellite')

# train X
X_train = traindf.filter(like='satellite').to_numpy()

# Cross-validate
cv_score = cross_validate(search, X_train, y_train,
                          scoring=scoring, cv=outer_cv)
cv_df = pd.DataFrame(data=cv_score)
cv_df.to_csv('output_post/classification_resampling/inc_satellite_rf.csv', index = False)

# Clean
del(X_train, cv_score, cv_df)


#%%% Outdoor
print('Outdoor')

# train X
X_train = traindf.loc[:,traindf.columns.str.contains('|'.join(['foto10_', 'foto4_', 'foto5_', 'satellite']))].to_numpy()


# Cross-validate
cv_score = cross_validate(search, X_train, y_train,
                          scoring=scoring, cv=outer_cv)
cv_df = pd.DataFrame(data=cv_score)
cv_df.to_csv('output_post/classification_resampling/inc_outdoor_rf.csv', index = False)

# Clean
del(X_train, cv_score, cv_df)


#%%% All
print('All')

# train X
X_train = traindf.filter(regex='foto|satellite').to_numpy()

# Cross-validate
cv_score = cross_validate(search, X_train, y_train,
                          scoring=scoring, cv=outer_cv)
cv_df = pd.DataFrame(data=cv_score)
cv_df.to_csv('output_post/classification_resampling/inc_all_rf.csv', index = False)

# Clean
del(X_train, cv_score, cv_df)


#%% Assets

print('Assets')

# Selecting train y 
y_train = traindf['assets_cat3'].to_numpy()
y_train = [0 if x=='bottom40' else 1 if x=='mid40' else 2 for x in y_train]


#%%% Satellite

print('Satellite')

# train X
X_train = traindf.filter(like='satellite').to_numpy()

# Cross-validate
cv_score = cross_validate(search, X_train, y_train,
                          scoring=scoring, cv=outer_cv)
cv_df = pd.DataFrame(data=cv_score)
cv_df.to_csv('output_post/classification_resampling/assets_satellite_rf.csv', index = False)

# Clean
del(X_train, cv_score, cv_df)


#%%% Outdoor
print('Outdoor')

# train X
X_train = traindf.loc[:,traindf.columns.str.contains('|'.join(['foto10_', 'foto4_', 'foto5_', 'satellite']))].to_numpy()

# Cross-validate
cv_score = cross_validate(search, X_train, y_train,
                          scoring=scoring, cv=outer_cv)
cv_df = pd.DataFrame(data=cv_score)
cv_df.to_csv('output_post/classification_resampling/assets_outdoor_rf.csv', index = False)

# Clean
del(X_train, cv_score, cv_df)


#%%% All
print('All')

# train X
X_train = traindf.filter(regex='foto|satellite').to_numpy()

# Cross-validate
cv_score = cross_validate(search, X_train, y_train,
                          scoring=scoring, cv=outer_cv)
cv_df = pd.DataFrame(data=cv_score)
cv_df.to_csv('output_post/classification_resampling/assets_all_rf.csv', index = False)

# Clean
del(X_train, cv_score, cv_df)
