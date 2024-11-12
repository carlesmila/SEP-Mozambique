"""
Created on Tue Oct 10 10:04:37 2023

@author: Carles Milà
Preprocessing step 1: Parsing questionnaire data
"""

import os
import glob
import pandas as pd

#%% Get main data 

# Get primary questionnaire files and merge them
main_paths = sorted(glob.glob('data/raw/DATA.2022*/*.csv'))
main_list = []
for p in main_paths:
    main_p = pd.read_csv(p, index_col=None, header=0)
    main_p = main_p.assign(file = p.split("/")[2])
    main_list.append(main_p)

main = pd.concat(main_list, axis=0, ignore_index=True)
del(main_list, main_p, p, main_paths)

# Prepare picture paths
for i in range(1, 12):
    main['foto' + str(i)] = 'data/raw/' + main['file'] + '/media/' + main['household'] + '-' + main['foto' + str(i)].str[6:]


#%% Cleaning

# Check number households visited and the main reasons of the non-successful visits
print(main.is_visit_possible.value_counts())
print(main.reason_not_visit.value_counts())

# Delete unsuccessful visits
main = main.loc[main['is_visit_possible']  == 1]

# Check duplicates 
print(main.household[main.duplicated('household')])

# Household '2901-048', delete first visit, no photos were taken
main = main.loc[main['KEY']  != '17062AY2UZCS576SB8MWG0HD8']

# Household '0407-049', delete first visit, few photos were taken
main = main.loc[main['KEY']  != '786V2D1SU0XNLT2K6FC3SLYHN']

# Households '0503-051'is repeated, take first
main = main.loc[main['KEY']  != '0BBO50R3E1R7QUEOKZ0LG2Y93']

# Household '0707-304', one of the two is wrong
main.loc[main['instanceID'] == 'VD3V7P4SKY6Y59F99VLNF8VZ2', 'household'] = '0707-260'

# Check duplicates again
print(main.household[main.duplicated('household')])

# Delete empty columns and columns we don't need
print(main.apply(lambda x: x.isna().sum()).to_string())
coldel = ['SubmissionDate', 'info_note_survey',
          'is_visit_possible', 'reason_not_visit',
          'consent_date', 
          'expenses', 'household_income',
          'start', 'end', 'today', 'instanceID',
          'instanceName', 'KEY',
          'deviceid', 'file']
main = main.drop(coldel, axis = 1)
del(coldel)

# Correct mistakes reported by field workers
main.loc[main.household=='0605-311','expenses01'] = 5051.
main.loc[main.household=='1601-073','household_income01'] = 20000.


#%%  Pictures quality checks

# Read quality check files
qchk_paths = sorted(glob.glob('data/quality_checks/*.ods'))
qchk_list = []
for p in qchk_paths:
    qchk_p = pd.read_excel(p, engine="odf")
    qchk_list.append(qchk_p)
qchk = pd.concat(qchk_list, axis=0, ignore_index=True)
del(qchk_list, qchk_p, p, qchk_paths)

# Columns for householdID and photograph path
qchk['photopath'] = 'media/' + qchk.Filename.str.slice(start = 9) + '.jpg'
qchk['typeID'] = qchk.Category.str.slice(start = 0, stop = 2) 
qchk['typeID'] = pd.to_numeric(qchk['typeID']).astype(int).astype(str)
# print(qchk.typeID.value_counts())

# Remove paths with non-valid photographs
for index, row in qchk.iterrows():
    main.loc[main.household == row['Household ID'], 'foto' + row['typeID']] = None
        
        
#%% Picture corrections

# Read corrections
updated_paths = sorted(glob.glob('data/raw/UPDATED.2022*/updated_sep_fotos.csv'))

updated_list = []
for p in updated_paths:
    updated_p = pd.read_csv(p, index_col=None, header=0)
    updated_p = updated_p.assign(file = p.split("/")[2])
    updated_list.append(updated_p)
updated = pd.concat(updated_list, axis=0, ignore_index=True)
del(updated_list, updated_p, p, updated_paths)

# Prepare picture paths
for i in range(1, 12):
    updated['foto' + str(i)] = 'data/raw/' + updated['file'] + '/media/' + updated['household'] + '-' + updated['foto' + str(i)].str[6:]

# Delete two images not in the main dataset: households '2409-209' and '0606-304'
updated = updated.loc[~updated['household'].isin(['2409-209', '0606-304'])]
all(updated.household.isin(list(main.household))) 

# Insert corrections - there are some extra images we will not use
for index, row in updated.iterrows():
    for f in range(1, 12):
        orig_path = main.loc[main.household == row['household'], 'foto' + str(f)].tolist()[0]
        correct_path = row['foto' + str(f)]
        if (not pd.isna(correct_path)) and pd.isna(orig_path):
            main.loc[main.household == row['household'], 'foto' + str(f)] = correct_path


#%% Fix wrong IDs

# We need to do it as the last step as image paths have the wrong ID
main.loc[main['household'] == '0600-304', 'household'] = '0606-304'
main.loc[main['household'] == '0708-429', 'household'] = '0802-429'
main.loc[main['household'] == '0802-028', 'household'] = '0708-028'
main.loc[main['household'] == '1803-017', 'household'] = '1901-017'
main.loc[main['household'] == '2101-186', 'household'] = '2101-106'
main.loc[main['household'] == '2101-204', 'household'] = '2201-204'
main.loc[main['household'] == '2901-034', 'household'] = '2901-037'
main.loc[main['household'] == '2903-182', 'household'] = '2903-108'
# main.loc[main['household'] == '1791-093', 'household'] = '1701-093'

#%%  Write to disk
main.to_csv('data/clean/quest_preprocessed.csv', index=False)
