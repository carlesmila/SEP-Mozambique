# Predicting household socioeconomic position in Mozambique using satellite and household imagery

This repository contains the code for the article "Predicting household socioeconomic position in Mozambique using satellite and household imagery" by C. Milà, T. Matsena, E. Jamisse, J. Nunes, Q. Bassat, P. Petrone, E. Sicuri, C. Sacoor, and C. Tonne. The manuscript is currently considered for publication; a preprint of the article is available on [arXiv](http://arxiv.org/abs/2411.08934). To see the code corresponding to the preprint version of the article, please check the [preprint branch](https://github.com/carlesmila/SEP-Mozambique/tree/preprint) of the repository.

We used `R` version 4.2 and the package `ca` for the MCA to produce asset-based SEP. The rest of analyses were performed in `python` version 3.11.7 using the following modules: `pandas` for tabular data management; `geopandas` and `Rasterio` for spatial data management, `Pillow` for image data management, `PyTorch` and `torchvision` for computer vision deep learning models, `scikit-learn` and `xgboost` for machine learning models and workflows, `shap` for SHAP analyses, and `matplotlib` and `seaborn` for graphics. Other packages were used for additional minor tasks.


## R scripts

The R scripts included in the [R](R/) folder are the following:

* [assetindex.R](R/assetindex.R): Script that creates the asset-based SEP measure using MCA.
* [study_area_map.R](R/study_area_map.R): Script that creates the study map figure included in the manuscript.

## Python scripts

Python scripts are organised in two folders: [preprocessing](python/preprocessing), where the code to clean the data can be found; and [analysis](python/analysis) for the analysis workflow.

The [preprocessing](python/preprocessing) scripts are the following:

* [1_parse_questionnaires.py](python/preprocessing/1_parse_questionnaires.py): Script that cleans the raw questionnaire data, fixing data entry errors.
* [2_order_photographs.py](python/preprocessing/2_order_photographs.py): Script that orders the image files according to their type.
* [3_extract_aerial.py](python/preprocessing/3_extract_aerial.py): Script that preprocesses the satellite data and extracts the 25m and 100m buffers around all household geocodes.

The [analysis](python/analysis) scripts are the following:

* [1_create_indices.py](python/analysis/1_create_indices.py): Script that creates the SEP measures from questionnaire data used as ground truth.
* [2_data_split.py](python/analysis/2_data_split.py): Script that does the train/test data partition.
* [3_hyperparameters](python/analysis/3_hyperparameters): This folder contains the scripts used for hyperparameter tuning of the CNN. The script [random_search.py](python/analysis/3_hyperparameters/random_search.py) contains the code to generate the hyperparameter configurations that are tried for four different image types in the rest of the scripts in the folder.
* [4_finetuning](python/analysis/4_finetuning): This folder contains the scripts used to finetune the CNN with the hyperparameters found in the previous step. There are 13 scripts, one for each household image type + the two satellite images with the different buffer sizes. They all source the script [finetuning_utils.py](python/analysis/finetuning_utils.py), which contains utils to load datasets and apply transforms.
* [5_features_tuning.py](python/analysis/5_features_tuning.py) and [5_features_notuning.py](python/analysis/5_features_notuning.py): These two scripts contain the code to extract feature vectors from the images using the CNNs with and without finetuning, respectively.
* [6_prepare_datasets.py](python/analysis/6_prepare_datasets.py): Script that merges the ground truth SEP measures and the feature vectors and creates the datasets used in supervised analyses.
* [7_regression_tuning_rf.py](python/analysis/7_regression_tuning_rf.py) and the rest of scripts starting with 7_: They run the regression analyses used in the paper. Note that the script name and title refers to the algorithm used in each of them (random forest, XGBoost), and the feature vectors and validation used (tuning: features extracted from the CNN with finetuning and validated in test data; notuning: features extracted from the CNN without finetuning and validated in test data; resampling: features extracted from the CNN without finetuning validated using 5-times 5-fold CV).
* [8_classification_tuning_rf.py](python/analysis/8_classification_tuning_rf.py) and the rest of scripts starting with 8_: They run the classification analyses used in the paper. The script names follow the same organization as regression scripts.
* [9_shap.py](python/analysis/9_shap.py): Script that computes SHAP values for complete random forest regression models.
* [10_regression_reduced.py](python/analysis/10_regression_reduced.py): Script that runs the random forest regression models reduced with the images identified in SHAP analyses.

## Jupyter notebooks

The jupyter notbooks included in the [ipynb](ipynb/) folder are the following:

* [0_missing.ipynb](ipynb/0_missing.ipynb): Notebook used to generate the missing data table.
* [1_exploratory_outcomes.ipynb](ipynb/1_exploratory_outcomes.ipynb): Notebook used to generate the ground truth exploratory analysis figures.
* [2_exploratory_photographs.ipynb](ipynb/2_exploratory_photographs.ipynb): Notebook used to generate the photograph exploratory analysis, where example images for each SEP quartile are shown.
* [3_Hyperparameters.ipynb](ipynb/3_Hyperparameters.ipynb): Notebook that includes the graphs used to choose the best hyperparameter configuration for the CNNs.
* [4_Finetuning.ipynb](ipynb/4_Finetuning.ipynb): Notebook that generates the table with the CNN binary accuracy statistics for each SEP measure and image type.
* [5_supervised.ipynb](ipynb/5_supervised.ipynb): Notebook used to generate the results tables of the supervised analyses.
* [6_shap.ipynb](ipynb/6_shap.ipynb): Notebook that generates the figures related to SHAP analyses, namely the SHAP feature importance boxplots and the images with the largest contributions.
* [7_reduced.ipynb](ipynb/7_reduced.ipynb): Notebook used to generate the results tables of the reduced regression models with the features identified in SHAP analyses.


## Further notes

Many of the included python scripts were performed in a HPC cluster due to long computational runtime and/or data size constraints. Those scripts are appropriately labeled (HPC) in the script title. 
