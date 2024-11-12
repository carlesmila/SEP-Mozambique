#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sept 3 2024

CNN hyperparameter tuning experiments

# *1. Learning rate*
# 2. Batch size
# 3. L2
# 4. Momentum
# 5. Hidden units

@author: cmila
"""

# Setup
import os
import pandas as pd
import numpy as np
import random



#%% Random hyperparameter search
random.seed(1234)
hyper = {# ID
         'ID': list(range(10)),
          # Learning rate
         'lr': np.random.choice([1e-2, 1e-3, 1e-4], 10), 
         # Batch size
         'bs': np.random.choice([8, 16, 32, 64], 10), 
         # Weight decay (L2)
         'l2': np.random.choice([1e-1, 1e-2, 1e-4, 1e-6], 10), 
         # Momentum
         'mom': np.random.choice([0.5, 0.9, 0.99], 10)}
hyper =  pd.DataFrame.from_dict(hyper)
print(hyper)

hyper.to_csv('output/hyperparameters/random-search.csv', index = False)
