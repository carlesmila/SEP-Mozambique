#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Feb 8 2024

Predictive modelling of household SEP (HPC)
Fine-tuned features, RF

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
from sklearn.pipeline import Pipeline
import xgboost as xgb
import joblib


#%% Prep

# Read data
traindf = pd.read_csv('output/supervised_tuning/traindata.csv')
testdf = pd.read_csv('output/supervised_tuning/testdata.csv')

# Model hyperparameters
mod_params = {
    'regressor__min_samples_leaf': [20, 30, 40, 50],
    'regressor__max_features': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
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


#%% Expenditure

print('Expenditure')

# Selecting train y 
y_train = traindf['exp_true'].to_numpy()

# Selecting test y 
y_test = testdf['exp_true'].to_numpy()


#%%% Satellite

print('Satellite')

# train X
X_train = traindf.filter(like='satellite').to_numpy()

# test X
X_test = testdf.filter(like='satellite').to_numpy()

# Fit pipeline
pipe_fit = pipe.fit(X_train, y_train)
print(pipe_fit.best_params_)

# Predict
pred_train = pipe_fit.predict(X_train)
pred_test = pipe_fit.predict(X_test)
traindf['exp_sat'] = pred_train
testdf['exp_sat'] = pred_test

# Write model
joblib.dump(pipe_fit.best_estimator_, 'output/supervised_tuning/exp_satellite_rf.pkl')

# Clean
del(X_train, X_test, pred_train, pred_test, pipe_fit)


#%%% Outdoor
print('Outdoor')

# train X
X_train = traindf.loc[:,testdf.columns.str.contains('|'.join(['foto10_', 'foto4_', 'foto5_', 'satellite']))].to_numpy()

# test X
X_test = testdf.loc[:,testdf.columns.str.contains('|'.join(['foto10_', 'foto4_', 'foto5_', 'satellite']))].to_numpy()

# Fit pipeline
pipe_fit = pipe.fit(X_train, y_train)
print(pipe_fit.best_params_)

# Predict
pred_train = pipe_fit.predict(X_train)
pred_test = pipe_fit.predict(X_test)
traindf['exp_outdoor'] = pred_train
testdf['exp_outdoor'] = pred_test

# Write model
joblib.dump(pipe_fit.best_estimator_, 'output/supervised_tuning/exp_outdoor_rf.pkl')

# Clean
del(X_train, X_test, pred_train, pred_test, pipe_fit)


#%%% All
print('All')

# train X
X_train = traindf.filter(regex='foto|satellite').to_numpy()

# test X
X_test = testdf.filter(regex='foto|satellite').to_numpy()

# Fit pipeline
pipe_fit = pipe.fit(X_train, y_train)
print(pipe_fit.best_params_)

# Predict
pred_train = pipe_fit.predict(X_train)
pred_test = pipe_fit.predict(X_test)
traindf['exp_all'] = pred_train
testdf['exp_all'] = pred_test

# Write model
joblib.dump(pipe_fit.best_estimator_, 'output/supervised_tuning/exp_all_rf.pkl')

# Clean
del(y_train, y_test, X_train, X_test, pred_train, pred_test, pipe_fit)


#%% Income

print('Income')

# Selecting train y 
y_train = traindf['inc_true'].to_numpy()

# Selecting test y 
y_test = testdf['inc_true'].to_numpy()


#%%% Satellite

print('Satellite')

# train X
X_train = traindf.filter(like='satellite').to_numpy()

# test X
X_test = testdf.filter(like='satellite').to_numpy()

# Fit pipeline
pipe_fit = pipe.fit(X_train, y_train)
print(pipe_fit.best_params_)

# Predict
pred_train = pipe_fit.predict(X_train)
pred_test = pipe_fit.predict(X_test)
traindf['inc_sat'] = pred_train
testdf['inc_sat'] = pred_test

# Write model
joblib.dump(pipe_fit.best_estimator_, 'output/supervised_tuning/inc_satellite_rf.pkl')

# Clean
del(X_train, X_test, pred_train, pred_test, pipe_fit)


#%%% Outdoor
print('Outdoor')

# train X
X_train = traindf.loc[:,testdf.columns.str.contains('|'.join(['foto10_', 'foto4_', 'foto5_', 'satellite']))].to_numpy()

# test X
X_test = testdf.loc[:,testdf.columns.str.contains('|'.join(['foto10_', 'foto4_', 'foto5_', 'satellite']))].to_numpy()

# Fit pipeline
pipe_fit = pipe.fit(X_train, y_train)
print(pipe_fit.best_params_)

# Predict
pred_train = pipe_fit.predict(X_train)
pred_test = pipe_fit.predict(X_test)
traindf['inc_outdoor'] = pred_train
testdf['inc_outdoor'] = pred_test

# Write model
joblib.dump(pipe_fit.best_estimator_, 'output/supervised_tuning/inc_outdoor_rf.pkl')

# Clean
del(X_train, X_test, pred_train, pred_test, pipe_fit)


#%%% All
print('All')

# train X
X_train = traindf.filter(regex='foto|satellite').to_numpy()

# test X
X_test = testdf.filter(regex='foto|satellite').to_numpy()

# Fit pipeline
pipe_fit = pipe.fit(X_train, y_train)
print(pipe_fit.best_params_)

# Predict
pred_train = pipe_fit.predict(X_train)
pred_test = pipe_fit.predict(X_test)
traindf['inc_all'] = pred_train
testdf['inc_all'] = pred_test

# Write model
joblib.dump(pipe_fit.best_estimator_, 'output/supervised_tuning/inc_all_rf.pkl')

# Clean
del(y_train, y_test, X_train, X_test, pred_train, pred_test, pipe_fit)


#%% Assets

print('Assets')

# Selecting train y 
y_train = traindf['assets_true'].to_numpy()

# Selecting test y 
y_test = testdf['assets_true'].to_numpy()


#%%% Satellite

print('Satellite')

# train X
X_train = traindf.filter(like='satellite').to_numpy()

# test X
X_test = testdf.filter(like='satellite').to_numpy()

# Fit pipeline
pipe_fit = pipe.fit(X_train, y_train)
print(pipe_fit.best_params_)
# pipe_fit.best_estimator_['classifier']

# Predict
pred_train = pipe_fit.predict(X_train)
pred_test = pipe_fit.predict(X_test)
traindf['assets_sat'] = pred_train
testdf['assets_sat'] = pred_test

# Write model
joblib.dump(pipe_fit.best_estimator_, 'output/supervised_tuning/assets_satellite_rf.pkl')

# Clean
del(X_train, X_test, pred_train, pred_test, pipe_fit)


#%%% Outdoor
print('Outdoor')

# train X
X_train = traindf.loc[:,testdf.columns.str.contains('|'.join(['foto10_', 'foto4_', 'foto5_', 'satellite']))].to_numpy()

# test X
X_test = testdf.loc[:,testdf.columns.str.contains('|'.join(['foto10_', 'foto4_', 'foto5_', 'satellite']))].to_numpy()

# Fit pipeline
pipe_fit = pipe.fit(X_train, y_train)
print(pipe_fit.best_params_)

# Predict
pred_train = pipe_fit.predict(X_train)
pred_test = pipe_fit.predict(X_test)
traindf['assets_outdoor'] = pred_train
testdf['assets_outdoor'] = pred_test

# Write model
joblib.dump(pipe_fit.best_estimator_, 'output/supervised_tuning/assets_outdoor_rf.pkl')

# Clean
del(X_train, X_test, pred_train, pred_test, pipe_fit)


#%%% All
print('All')

# train X
X_train = traindf.filter(regex='foto|satellite').to_numpy()

# test X
X_test = testdf.filter(regex='foto|satellite').to_numpy()

# Fit pipeline
pipe_fit = pipe.fit(X_train, y_train)
print(pipe_fit.best_params_)

# Predict
pred_train = pipe_fit.predict(X_train)
pred_test = pipe_fit.predict(X_test)
traindf['assets_all'] = pred_train
testdf['assets_all'] = pred_test

# Write model
joblib.dump(pipe_fit.best_estimator_, 'output/supervised_tuning/assets_all_rf.pkl')

# Clean
del(y_train, y_test, X_train, X_test, pred_train, pred_test, pipe_fit)


#%% Write to disk
train_res = traindf.loc[: , ['household',
                 'exp_true', 'exp_sat', 'exp_outdoor', 'exp_all',
                 'inc_true', 'inc_sat', 'inc_outdoor', 'inc_all',
                 'assets_true', 'assets_sat', 'assets_outdoor', 'assets_all']]
train_res.to_csv('output/supervised_tuning/train_results_rf.csv', index = False)

test_res = testdf.loc[: , ['household',
                 'exp_true', 'exp_sat', 'exp_outdoor', 'exp_all',
                 'inc_true', 'inc_sat', 'inc_outdoor', 'inc_all',
                 'assets_true', 'assets_sat', 'assets_outdoor', 'assets_all']]
test_res.to_csv('output/supervised_tuning/test_results_rf.csv', index = False)
