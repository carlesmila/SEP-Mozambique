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
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV, cross_validate, KFold, RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer
from sklearn.feature_selection import SelectKBest, f_regression
from scipy.stats import spearmanr
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
    'regressor__min_samples_leaf': [20, 30, 40, 50],
    'regressor__max_features': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
}

# Hyperparameter grid
param_grid = param_grid = {**kbest_params,**mod_params}

# Cross-validation
inner_cv = KFold(n_splits=5)
outer_cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=0)

# Scoring
def spearman_scorer(y_true, y_pred):
    return spearmanr(y_true, y_pred)[0]
    
scoring = {'r2': 'r2',
           'rmse': 'neg_root_mean_squared_error',
           'rho': make_scorer(spearman_scorer)}

# Pipeline
pipe = Pipeline([
   ('imputer', SimpleImputer()),
   ('selector', SelectKBest(f_regression)),
   ('regressor',  RandomForestRegressor(n_estimators=300, n_jobs=6))
   ])

search = RandomizedSearchCV(pipe, param_grid, n_iter=10, cv=inner_cv, random_state=1234)



#%% Expenditure

print('Expenditure')

# Selecting train y 
y_train = traindf['exp_true'].to_numpy()


#%%% Satellite

print('Satellite')

# train X
X_train = traindf.filter(like='satellite').to_numpy()

# Cross-validate
cv_score = cross_validate(search, X_train, y_train, scoring=scoring, cv=outer_cv)
cv_df = pd.DataFrame(data=cv_score)
cv_df.to_csv('output_post/regression_resampling/exp_satellite_rf.csv', index = False)

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
cv_df.to_csv('output_post/regression_resampling/exp_outdoor_rf.csv', index = False)

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
cv_df.to_csv('output_post/regression_resampling/exp_all_rf.csv', index = False)

# Clean
del(X_train, cv_score, cv_df)


#%% Income

print('Income')

# Selecting train y 
y_train = traindf['inc_true'].to_numpy()


#%%% Satellite

print('Satellite')

# train X
X_train = traindf.filter(like='satellite').to_numpy()

# Cross-validate
cv_score = cross_validate(search, X_train, y_train,
                          scoring=scoring, cv=outer_cv)
cv_df = pd.DataFrame(data=cv_score)
cv_df.to_csv('output_post/regression_resampling/inc_satellite_rf.csv', index = False)

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
cv_df.to_csv('output_post/regression_resampling/inc_outdoor_rf.csv', index = False)

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
cv_df.to_csv('output_post/regression_resampling/inc_all_rf.csv', index = False)

# Clean
del(X_train, cv_score, cv_df)


#%% Assets

print('Assets')

# Selecting train y 
y_train = traindf['assets_true'].to_numpy()


#%%% Satellite

print('Satellite')

# train X
X_train = traindf.filter(like='satellite').to_numpy()

# Cross-validate
cv_score = cross_validate(search, X_train, y_train,
                          scoring=scoring, cv=outer_cv)
cv_df = pd.DataFrame(data=cv_score)
cv_df.to_csv('output_post/regression_resampling/assets_satellite_rf.csv', index = False)

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
cv_df.to_csv('output_post/regression_resampling/assets_outdoor_rf.csv', index = False)

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
cv_df.to_csv('output_post/regression_resampling/assets_all_rf.csv', index = False)

# Clean
del(X_train, cv_score, cv_df)
