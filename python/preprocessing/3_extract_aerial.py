"""
Created on Tue Nov 10 09:00:33 2023

@author: Carles Milà
Preprocessing step 3: Extracting aerial images (HPC)
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio as rio
from rasterio.merge import merge
import glob
import copy
import fnmatch

# Read survey data, georeference and project
quest = pd.read_csv('data/clean/quest_preprocessed.csv')
quest = quest[['household', 'geopoint-Latitude', 'geopoint-Longitude', 'geopoint-Accuracy']]
quest.columns = ['household', 'latitude', 'longitude', 'accuracy']
quest = gpd.GeoDataFrame(
    quest[['household', 'accuracy']],
    geometry=gpd.points_from_xy(quest.longitude, quest.latitude), 
    crs="EPSG:4326"
)
quest = quest.to_crs("EPSG:32736")

# Values for 8-bit conversion (max 99%)
redmin = 147
redmax = 553
greenmin = 218
greenmax = 525
bluemin = 181
bluemax = 373


#%% 25m buffer

# Create 25m square buffers
buffs = copy.deepcopy(quest)
buffs['geometry'] = buffs.buffer(25, cap_style=3)

# Assign tiles to households (single tile)
tiles = gpd.read_file('data/satellite/GIS_FILES/050117329010_01_TILE_SHAPE.shp')
tiles = tiles[['prodDesc', 'fileName', 'geometry']]
tiles = tiles.to_crs("EPSG:32736")
singletile = buffs.sjoin(tiles, how="left", predicate="within")
singletile = singletile.drop_duplicates(subset=['household'], keep='first')
singletile = singletile[['household', 'accuracy', 'prodDesc', 'fileName', 'geometry']]
singletile.fileName.isna().sum() # 34 share tile, need to mosaic
multitile = singletile.loc[~singletile.fileName.notna()]
singletile = singletile.loc[singletile.fileName.notna()]
multitile = multitile.reset_index(drop = True)

# Assign tiles to households (multi tile)
multitile2 = buffs.sjoin(tiles, how="left", predicate="intersects")
multitile2 = multitile2[~multitile2.household.isin(singletile.household)]
for i, row in multitile.iterrows():
    multitile.iat[i, 2] = str(pd.unique(multitile2.loc[multitile2['household'] == row['household'],'prodDesc'])[0])
    multitile.iat[i, 3] = list(pd.unique(multitile2.loc[multitile2['household'] == row['household'],'fileName']))

# Extract mosaics
allfiles = glob.glob("data/satellite/*/*.TIF", recursive=True) # list all paths
for i, row in multitile.iterrows():
    files_i = [fnmatch.filter(allfiles, '*' + a)[0] for a in row['fileName']]
    src_mosaic_i = []
    for f in files_i:
        src_f = rio.open(f)
        src_mosaic_i.append(src_f)
    gpd_i = gpd.GeoDataFrame(row).T
    bounds_i = gpd_i["geometry"].bounds       
    bounds_i = list(bounds_i.itertuples(index=False, name=None))[0]
    mosaic_i, trans_i = merge(src_mosaic_i, bounds = bounds_i)
    # We take R-G-B rather than the original B-G-R-NIR
    mosaic_i = mosaic_i[[2,1,0]]
    # Convert to 8-bit: red
    red = mosaic_i[0] 
    redmask1 = np.where(red> redmax,True,False)
    red[redmask1] = redmax
    redmask2 = np.where(red< redmin,True,False)
    red[redmask2] = redmin
    red = (red-redmin)/(redmax-redmin)*255
    mosaic_i[0] = red
    # Convert to 8-bit: green
    green = mosaic_i[1] 
    greenmask1 = np.where(green > greenmax,True,False)
    green[greenmask1] = greenmax
    greenmask2 = np.where(green< greenmin,True,False)
    green[greenmask2] = greenmin
    green = (green-greenmin)/(greenmax-greenmin)*255
    mosaic_i[1] = green
    # Convert to 8-bit: blue
    blue = mosaic_i[2] 
    bluemask1 = np.where(blue > bluemax,True,False)
    blue[bluemask1] = bluemax
    bluemask2 = np.where(blue< bluemin,True,False)
    blue[bluemask2] = bluemin
    blue = (blue-bluemin)/(bluemax-bluemin)*255
    mosaic_i[2] = blue
    # update metadata
    meta_i = src_f.meta.copy()
    meta_i.update({"driver": "GTiff",
                   "dtype": "uint8",
                   "height": mosaic_i.shape[1],
                   "width": mosaic_i.shape[2],
                   'count': 3,
                   "transform": trans_i,
                   "crs": "EPSG:32736"})
    # Save to local (saving to server gives problems)
    with rio.open('data/clean/satellite25/' + str(row['household']) + '.tiff', "w", **meta_i) as dest:
        dest.write(mosaic_i.astype(rio.uint8))

# Extract single tiles
allfiles = glob.glob("data/satellite/*/*.TIF", recursive=True) # list all paths
for i, row in singletile.iterrows():
    files_i = fnmatch.filter(allfiles, '*' + row['fileName'])
    src_mosaic_i = []
    for f in files_i:
        src_f = rio.open(f)
        src_mosaic_i.append(src_f)
    gpd_i = gpd.GeoDataFrame(row).T
    bounds_i = gpd_i["geometry"].bounds       
    bounds_i = list(bounds_i.itertuples(index=False, name=None))[0]
    mosaic_i, trans_i = merge(src_mosaic_i, bounds = bounds_i)
    # We take R-G-B rather than the original B-G-R-NIR
    mosaic_i = mosaic_i[[2,1,0]]
    # Convert to 8-bit: red
    red = mosaic_i[0] 
    redmask1 = np.where(red> redmax,True,False)
    red[redmask1] = redmax
    redmask2 = np.where(red< redmin,True,False)
    red[redmask2] = redmin
    red = (red-redmin)/(redmax-redmin)*255
    mosaic_i[0] = red
    # Convert to 8-bit: green
    green = mosaic_i[1] 
    greenmask1 = np.where(green > greenmax,True,False)
    green[greenmask1] = greenmax
    greenmask2 = np.where(green< greenmin,True,False)
    green[greenmask2] = greenmin
    green = (green-greenmin)/(greenmax-greenmin)*255
    mosaic_i[1] = green
    # Convert to 8-bit: blue
    blue = mosaic_i[2] 
    bluemask1 = np.where(blue > bluemax,True,False)
    blue[bluemask1] = bluemax
    bluemask2 = np.where(blue< bluemin,True,False)
    blue[bluemask2] = bluemin
    blue = (blue-bluemin)/(bluemax-bluemin)*255
    mosaic_i[2] = blue
    # update metadata
    meta_i = src_f.meta.copy()
    meta_i.update({"driver": "GTiff",
                   "dtype": "uint8",
                   "height": mosaic_i.shape[1],
                   "width": mosaic_i.shape[2],
                   'count': 3,
                   "transform": trans_i,
                   "crs": "EPSG:32736"})
    # Save to local (saving to server gives problems)  
    with rio.open('data/clean/satellite25/' + str(row['household']) + '.tiff', "w", **meta_i) as dest:
        dest.write(mosaic_i.astype(rio.uint8))
        
del(buffs, tiles, singletile, multitile, multitile2)
 
       
#%% 100m buffer

# Create 100m square buffers
buffs = copy.deepcopy(quest)
buffs['geometry'] = buffs.buffer(100, cap_style=3)

# Assign tiles to households (single tile)
tiles = gpd.read_file('data/satellite/GIS_FILES/050117329010_01_TILE_SHAPE.shp')
tiles = tiles[['prodDesc', 'fileName', 'geometry']]
tiles = tiles.to_crs("EPSG:32736")
singletile = buffs.sjoin(tiles, how="left", predicate="within")
singletile = singletile.drop_duplicates(subset=['household'], keep='first')
singletile = singletile[['household', 'accuracy', 'prodDesc', 'fileName', 'geometry']]
singletile.fileName.isna().sum() # 34 share tile, need to mosaic
multitile = singletile.loc[~singletile.fileName.notna()]
singletile = singletile.loc[singletile.fileName.notna()]
multitile = multitile.reset_index(drop = True)

# Assign tiles to households (multi tile)
multitile2 = buffs.sjoin(tiles, how="left", predicate="intersects")
multitile2 = multitile2[~multitile2.household.isin(singletile.household)]
for i, row in multitile.iterrows():
    multitile.iat[i, 2] = str(pd.unique(multitile2.loc[multitile2['household'] == row['household'],'prodDesc'])[0])
    multitile.iat[i, 3] = list(pd.unique(multitile2.loc[multitile2['household'] == row['household'],'fileName']))

# Extract mosaics
allfiles = glob.glob("data/satellite/*/*.TIF", recursive=True) # list all paths
for i, row in multitile.iterrows():
    files_i = [fnmatch.filter(allfiles, '*' + a)[0] for a in row['fileName']]
    src_mosaic_i = []
    for f in files_i:
        src_f = rio.open(f)
        src_mosaic_i.append(src_f)
    gpd_i = gpd.GeoDataFrame(row).T
    bounds_i = gpd_i["geometry"].bounds       
    bounds_i = list(bounds_i.itertuples(index=False, name=None))[0]
    mosaic_i, trans_i = merge(src_mosaic_i, bounds = bounds_i)
    # We take R-G-B rather than the original B-G-R-NIR
    mosaic_i = mosaic_i[[2,1,0]]
    # Convert to 8-bit: red
    red = mosaic_i[0] 
    redmask1 = np.where(red> redmax,True,False)
    red[redmask1] = redmax
    redmask2 = np.where(red< redmin,True,False)
    red[redmask2] = redmin
    red = (red-redmin)/(redmax-redmin)*255
    mosaic_i[0] = red
    # Convert to 8-bit: green
    green = mosaic_i[1] 
    greenmask1 = np.where(green > greenmax,True,False)
    green[greenmask1] = greenmax
    greenmask2 = np.where(green< greenmin,True,False)
    green[greenmask2] = greenmin
    green = (green-greenmin)/(greenmax-greenmin)*255
    mosaic_i[1] = green
    # Convert to 8-bit: blue
    blue = mosaic_i[2] 
    bluemask1 = np.where(blue > bluemax,True,False)
    blue[bluemask1] = bluemax
    bluemask2 = np.where(blue< bluemin,True,False)
    blue[bluemask2] = bluemin
    blue = (blue-bluemin)/(bluemax-bluemin)*255
    mosaic_i[2] = blue
    # update metadata
    meta_i = src_f.meta.copy()
    meta_i.update({"driver": "GTiff",
                   "dtype": "uint8",
                   "height": mosaic_i.shape[1],
                   "width": mosaic_i.shape[2],
                   'count': 3,
                   "transform": trans_i,
                   "crs": "EPSG:32736"})
    # Save to local (saving to server gives problems)
    with rio.open('data/clean/satellite100/' + str(row['household']) + '.tiff', "w", **meta_i) as dest:
        dest.write(mosaic_i.astype(rio.uint8))

# Extract single tiles
allfiles = glob.glob("data/satellite/*/*.TIF", recursive=True) # list all paths
for i, row in singletile.iterrows():
    files_i = fnmatch.filter(allfiles, '*' + row['fileName'])
    src_mosaic_i = []
    for f in files_i:
        src_f = rio.open(f)
        src_mosaic_i.append(src_f)
    gpd_i = gpd.GeoDataFrame(row).T
    bounds_i = gpd_i["geometry"].bounds       
    bounds_i = list(bounds_i.itertuples(index=False, name=None))[0]
    mosaic_i, trans_i = merge(src_mosaic_i, bounds = bounds_i)
    # We take R-G-B rather than the original B-G-R-NIR
    mosaic_i = mosaic_i[[2,1,0]]
    # Convert to 8-bit: red
    red = mosaic_i[0] 
    redmask1 = np.where(red> redmax,True,False)
    red[redmask1] = redmax
    redmask2 = np.where(red< redmin,True,False)
    red[redmask2] = redmin
    red = (red-redmin)/(redmax-redmin)*255
    mosaic_i[0] = red
    # Convert to 8-bit: green
    green = mosaic_i[1] 
    greenmask1 = np.where(green > greenmax,True,False)
    green[greenmask1] = greenmax
    greenmask2 = np.where(green< greenmin,True,False)
    green[greenmask2] = greenmin
    green = (green-greenmin)/(greenmax-greenmin)*255
    mosaic_i[1] = green
    # Convert to 8-bit: blue
    blue = mosaic_i[2] 
    bluemask1 = np.where(blue > bluemax,True,False)
    blue[bluemask1] = bluemax
    bluemask2 = np.where(blue< bluemin,True,False)
    blue[bluemask2] = bluemin
    blue = (blue-bluemin)/(bluemax-bluemin)*255
    mosaic_i[2] = blue
    # update metadata
    meta_i = src_f.meta.copy()
    meta_i.update({"driver": "GTiff",
                   "dtype": "uint8",
                   "height": mosaic_i.shape[1],
                   "width": mosaic_i.shape[2],
                   'count': 3,
                   "transform": trans_i,
                   "crs": "EPSG:32736"})
    # Save to local (saving to server gives problems)  
    with rio.open('data/clean/satellite100/' + str(row['household']) + '.tiff', "w", **meta_i) as dest:
        dest.write(mosaic_i.astype(rio.uint8))
        
del(buffs, tiles, singletile, multitile, multitile2)