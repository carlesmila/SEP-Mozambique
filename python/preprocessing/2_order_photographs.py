"""
Created on Tue Oct 26 09:06:42 2023

@author: Carles Milà
Preprocessing step 2: Ordering photographs (HPC)
"""

import os
import pandas as pd


quest = pd.read_csv('data/clean/quest_preprocessed.csv')

#### Check that all photographs do exist ####
for index, row in quest.iterrows():
    for f in range(1, 12):
        photo_path = row['foto' + str(f)]
        if not pd.isna(photo_path) and not pd.isnull(photo_path):
            if not os.path.isfile(photo_path):
                print(photo_path)

#%% Order photos by picture type
for index, row in quest.iterrows():
    for f in range(1, 12):
        origin_path = row['foto' + str(f)]
        if not pd.isna(origin_path) and not pd.isnull(origin_path):
            dest_path = 'data/clean/foto' + str(f) + '/' + origin_path.split('/')[4]
            os.system('cp ' + origin_path + ' ' + dest_path)  
            

#%% Change paths in quest
for i in range(1,12):
    quest['foto' + str(i)] = 'data/clean/foto' + str(i) + '/' + quest['foto' + str(i)].str.split('/', expand=True)[4]
    
quest['satellite25'] = 'data/clean/satellite25/' + quest['household'] + '.tiff'
quest['satellite100'] = 'data/clean/satellite100/' + quest['household'] + '.tiff'


# Write to disk
quest.to_csv('data/clean/quest_clean.csv', index=False)
