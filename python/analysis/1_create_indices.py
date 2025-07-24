"""
Created on Thursday Jan 18 2024

@author: Carles Mila
Analysis step 1: Creating SEP indices
"""

import os
import pandas as pd

# os.chdir('/run/user/787601554/gvfs/smb-share:server=fs05.isglobal.lan,share=hpc_airpollution/carles/SEP')
os.chdir('/PROJECTES/AIRPOLLUTION/carles/SEP')

#%% Preprocess questionnaires

# Read clean questionnaire data
quest = pd.read_csv('data/clean/quest_clean.csv')

# Get expenses and income-related SEP variables
indicators = quest.loc[:,['household',
                          'expenses01', 'expenses02', 'expenses03', 'expenses04', 
                          'expenses05', 'expenses06', 'expenses07', 'expenses08',
                          'expenses09', 'expenses10', 'expenses11', 'expenses12', 
                          'expenses13', 'expenses14',
                          'household_income01',
                          'household_income02', 'household_income03']]
with pd.option_context('display.max_rows', None, # Missing data
                       'display.max_columns', None,
                       'display.precision', 3,
                       ):
    print(indicators.isna().sum())


# Rename
newnames = {'expenses01':'exp_allfood',
            'expenses02':'exp_meat',
            'expenses03':'exp_ricebread', 
            'expenses04':'exp_fuel', 
            'expenses05':'exp_health',
            'expenses06':'exp_electricity', 
            'expenses07':'exp_education', 
            'expenses08':'exp_agriculture',
            'expenses09':'exp_clothing',
            'expenses10':'exp_household',
            'expenses11':'exp_tools', 
            'expenses12':'exp_insurance',
            'expenses13':'exp_transport',
            'expenses14':'exp_hobbies',
            'household_income01':'inc_own',
            'household_income02':'inc_salary', 
            'household_income03':'inc_other',
            'finance':'self_finance'}
indicators = indicators.rename(columns = newnames)


#%% Create aggregates

# sum of all expenses
indicators['exp_all'] = indicators[['exp_allfood', 'exp_fuel', 
                                    'exp_health', 'exp_electricity', 
                                    'exp_education', 'exp_agriculture', 
                                    'exp_clothing','exp_household',
                                    'exp_tools','exp_insurance',
                                    'exp_transport','exp_hobbies']].sum(axis=1)
# sum of all income sources
indicators['inc_all'] = indicators[['inc_own', 'inc_salary', 'inc_other']].sum(axis=1)


#%% Add asset index (created in R)
assets = pd.read_csv('data/clean/asset_index.csv')
assets.columns = ['household', 'assets', "hhsize"]
indicators = indicators.merge(assets, on = 'household')


#%% Categorical binary indicators
indicators['exp_cat2'] = pd.qcut(indicators['exp_all'], 2, labels = ["low", "high"])
indicators['inc_cat2'] = pd.qcut(indicators['inc_all'], 2, labels = ["low", "high"])
indicators['assets_cat2'] = pd.qcut(indicators['assets'], 2, labels = ["low", "high"])

#%% Categorical 40/40/20 indicators
indicators['exp_cat3'] = pd.qcut(indicators['exp_all'], q = [0, .4, .8, 1.], labels = ["bottom40", "mid40", "top20"])
indicators['inc_cat3'] = pd.qcut(indicators['inc_all'], q = [0, .4, .8, 1.], labels = ["bottom40", "mid40", "top20"])
indicators['assets_cat3'] = pd.qcut(indicators['assets'], q = [0, .4, .8, 1.], labels = ["bottom40", "mid40", "top20"])

#%% Write to disk
indicators.to_csv('data/clean/indicators.csv', index=False)
