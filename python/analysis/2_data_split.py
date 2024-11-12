"""
Created on Tue Jan 23 2024

Train/test split

@author: cmila
"""

# Read data
import os
import pandas as pd
import random


#%% Create indicators

# Read clean indicator data
indicators = pd.read_csv('data/clean/indicators.csv')
    
# 975 households, 800 train, 175 dev/test
split = sum(([x]*y for x,y in zip(["train", "test"], (800, 175))),[])

# Randomize
random.seed(1234)
split = random.sample(split, 975)

# Join with table and check that it is balanced wrt the outcomes
indicators['split'] = split
print(pd.crosstab(indicators['assets_cat2'], indicators['split'], margins=False))
print(pd.crosstab(indicators['exp_cat2'], indicators['split'], margins=False))
print(pd.crosstab(indicators['inc_cat2'], indicators['split'], margins=False))

# Write splits to disk
splitdf = indicators[['household', 'split']]
splitdf.to_csv("data/clean/datasplit.csv", index=False)
del(indicators, splitdf)