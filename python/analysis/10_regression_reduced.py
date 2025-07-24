#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Feb 8 2024

Predictive modelling of household SEP (HPC)
Reduced set of features derived from SHAP

@author: cmila
"""

# Setup
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import root_mean_squared_error
from sklearn.pipeline import Pipeline
import xgboost as xgb
import joblib


#%% Prep

# Read data
traindf = pd.read_csv('output/data_tuning/traindata.csv')
testdf = pd.read_csv('output/data_tuning/testdata.csv')

# Model hyperparameters
mod_params = {
    'regressor__min_samples_leaf': [20, 30, 40, 50],
    'regressor__max_features': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
}

# Hyperparameter grid
param_grid = [{**mod_params}]

# Pipeline
pipe = Pipeline([
    ('imputer', SimpleImputer()),
    ('regressor', RandomForestRegressor(n_estimators=400, n_jobs=3))
    ])
pipe = RandomizedSearchCV(pipe, param_grid, n_iter=10, cv=10,
                               verbose=True, random_state=1234)


#%% Expenditure reduced set with foto3 (light source)
print('Expenditure')

# Selecting train y 
y_train = traindf['exp_true'].to_numpy()

# Selecting test y 
y_test = testdf['exp_true'].to_numpy()

# train X
reduced_set = ['foto10_', 'foto4_', 'foto5_', 'satellite', 'foto3_']
X_train = traindf.loc[:,testdf.columns.str.contains('|'.join(reduced_set))].to_numpy()

# test X
X_test = testdf.loc[:,testdf.columns.str.contains('|'.join(reduced_set))].to_numpy()

# Fit pipeline
pipe_fit = pipe.fit(X_train, y_train)
print(pipe_fit.best_params_)

# Predict
pred_train = pipe_fit.predict(X_train)
pred_test = pipe_fit.predict(X_test)
traindf['exp_red'] = pred_train
testdf['exp_red'] = pred_test

# Write model
joblib.dump(pipe_fit.best_estimator_, 'output/regression_reduced/exp_reduced_rf.pkl')

# Clean
del(X_train, X_test, pred_train, pred_test, pipe_fit, reduced_set)




#%% Income reduced set with foto6 (kitchen)
print('Income')

# Selecting train y 
y_train = traindf['inc_true'].to_numpy()

# Selecting test y 
y_test = testdf['inc_true'].to_numpy()

# train X
reduced_set = ['foto10_', 'foto4_', 'foto5_', 'satellite', 'foto6_']
X_train = traindf.loc[:,testdf.columns.str.contains('|'.join(reduced_set))].to_numpy()

# test X
X_test = testdf.loc[:,testdf.columns.str.contains('|'.join(reduced_set))].to_numpy()

# Fit pipeline
pipe_fit = pipe.fit(X_train, y_train)
print(pipe_fit.best_params_)

# Predict
pred_train = pipe_fit.predict(X_train)
pred_test = pipe_fit.predict(X_test)
traindf['inc_red'] = pred_train
testdf['inc_red'] = pred_test

# Write model
joblib.dump(pipe_fit.best_estimator_, 'output/regression_reduced/inc_reduced_rf.pkl')

# Clean
del(X_train, X_test, pred_train, pred_test, pipe_fit, reduced_set)


#%% Assets reduced set with foto3 (light source)
print('Assets')

# Selecting train y 
y_train = traindf['assets_true'].to_numpy()

# Selecting test y 
y_test = testdf['assets_true'].to_numpy()

# train X
reduced_set = ['foto10_', 'foto4_', 'foto5_', 'satellite', 'foto3_']
X_train = traindf.loc[:,testdf.columns.str.contains('|'.join(reduced_set))].to_numpy()

# test X
X_test = testdf.loc[:,testdf.columns.str.contains('|'.join(reduced_set))].to_numpy()

# Fit pipeline
pipe_fit = pipe.fit(X_train, y_train)
print(pipe_fit.best_params_)
# pipe_fit.best_estimator_['classifier']

# Predict
pred_train = pipe_fit.predict(X_train)
pred_test = pipe_fit.predict(X_test)
traindf['assets_red'] = pred_train
testdf['assets_red'] = pred_test

# Write model
joblib.dump(pipe_fit.best_estimator_, 'output/regression_reduced/assets_reduced_rf.pkl')

# Clean
del(X_train, X_test, pred_train, pred_test, pipe_fit, reduced_set)


#%% Write to disk
train_res = traindf.loc[: , ['household',
                 'exp_true', 'exp_red', 
                 'inc_true', 'inc_red', 
                 'assets_true', 'assets_red']]
train_res.to_csv('output/regression_reduced/train_results_reduced.csv', index = False)

test_res = testdf.loc[: , ['household',
                 'exp_true', 'exp_red', 
                 'inc_true', 'inc_red', 
                 'assets_true', 'assets_red']]
test_res.to_csv('output/regression_reduced/test_results_reduced.csv', index = False)
