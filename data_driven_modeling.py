#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   production_regress.py
@Contact :   qinxl@aircas.ac.cn
@License :   (C)Copyright 2021-2022, CropWatch Group
@Modify Time : 2023/06/28
@Author : Qin Xingli
@Version : 1.0
@Description : Code simplified with the help of ChatGPT
"""
from osgeo import gdal
import numpy as np
from sklearn.model_selection import train_test_split
import datetime
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
import pandas as pd
import glob
import re
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os
from joblib import dump, load

import read_AEZs_data
import process_country_name

os.environ['PROJ_LIB'] = r'C:\ProgramData\Anaconda3\envs\MyProductionIndex\Lib\site-packages\osgeo\data\proj'

def read_image_as_array(img_path):
    """
    Read input image data
    """
    gdal.AllRegister()
    img_data = gdal.Open(img_path, gdal.GA_ReadOnly)
    if img_data is None:
        print('{} failed to open!'.format(img_path))
        exit(1)
    im_width = img_data.RasterXSize  # Read number of image rows
    im_height = img_data.RasterYSize  # Read number of image columns
    im_bands = img_data.RasterCount
    transform = img_data.GetGeoTransform()
    proj = img_data.GetProjection()

    ith_band = img_data.GetRasterBand(1)
    im_data = ith_band.ReadAsArray(0, 0, im_width, im_height)

    return im_data, im_width, im_height, transform, proj


def get_country_phenology(country_name, crop_type):
    # 4 quarters
    crop_types = ["Maize", "Wheat", "Rice", "Soybean"]
    xls_file = r"D:\Dataset\产量分解\Support_Data_for_CPI\Country_Phenology.xlsx"
    xls_data = pd.read_excel(xls_file, sheet_name=None)
    sheet_names = list(xls_data.keys())

    # Default value is a 4*4 array filled with 1
    country_phenology = np.ones((4, 4))

    if country_name == 'United Kingdom of Great Britain and Northern Ireland':
        country_name = 'United Kingdom'

    # If the country is in sheet_names, update the corresponding country_phenology
    if country_name in sheet_names:
        ith_country_phenology = xls_data[country_name]
        ith_country_phenology = ith_country_phenology.values
        ith_country_phenology = ith_country_phenology[:, 1:]
        country_phenology = ith_country_phenology

    # Extract the corresponding row based on crop_type
    if crop_type in crop_types:
        crop_type_index = crop_types.index(crop_type)
        country_phenology_single_crop = country_phenology[crop_type_index, :]
    else:
        country_phenology_single_crop = np.ones((1, 4))

    return country_phenology_single_crop

def get_country_phenology_36(country_name, crop_type):
    # 36 means 36 weeks a year
    crop_types = ["Maize", "Wheat", "Rice", "Soybean"]
    xls_file = r"D:\Dataset\产量分解\Support_Data_for_CPI\Country_Phenology_36.xlsx"
    xls_data = pd.read_excel(xls_file, sheet_name=None)
    sheet_names = list(xls_data.keys())

    # Default value is a 4*36 array filled with 1
    country_phenology = np.ones((4, 36))

    if country_name == 'United Kingdom of Great Britain and Northern Ireland':
        country_name = 'United Kingdom'

    # If the country is in sheet_names, update the corresponding country_phenology
    if country_name in sheet_names:
        ith_country_phenology = xls_data[country_name]
        ith_country_phenology = ith_country_phenology.values
        ith_country_phenology = ith_country_phenology[:, 1:]
        country_phenology = ith_country_phenology

    # Extract the corresponding row based on crop_type
    if crop_type in crop_types:
        crop_type_index = crop_types.index(crop_type)
        country_phenology_single_crop = country_phenology[crop_type_index, :]
    else:
        country_phenology_single_crop = np.ones((1, 36))

    return country_phenology_single_crop

def read_indicators(indicator_name, data_year, require_index, country_phenology_single_crop):
    # Path of indicators
    indicator_root_folder = 'D:/Dataset/产量分解/Indicators_resample/'
    indicator_folder = '{}/{}/{}/'.format(indicator_root_folder, indicator_name, data_year)
    require_data_list = []
    require_data_image_name_list = []

    # The following features do not need to be filtered according to the country's growth period
    constant_indicator = ['GLASS_NPP', 'Irrigation', 'Soil', 'Variation', 'Dem']

    # Step 1: Traverse the folder and save the tif file paths to a temporary variable
    tif_file_paths = glob.glob(indicator_folder + '*.tif')

    # Sort by numbers in the file name
    def sort_key(s):
        # Extract numbers from file name
        numbers = re.findall(r'\d+', s)
        return [int(num) for num in numbers] if numbers else [float('inf')]

    tif_file_paths.sort(key=sort_key)

    # For sorting VCI and CALF indicators
    sort_mark = None
    if indicator_name == 'VCI':
        sort_mark = ['017NScenes8', '113NScenes8', '193NScenes8', '289NScenes8']
    if indicator_name == 'CALF':
        sort_mark = ['_JFMA', '_AMJJ', '_JASO', '_ONDJ']
    # Sort indicators according to sort_mark
    if sort_mark is not None:
        sort_require_tif_file_paths = []
        for marker in sort_mark:
            for k, image_path in enumerate(tif_file_paths):
                image_name = os.path.basename(image_path)
                if marker in image_name:
                    sort_require_tif_file_paths.append(tif_file_paths[k])
                    continue
        tif_file_paths = sort_require_tif_file_paths

    # Only do grouping and filtering if the indicator is not in constant_indicator and the number of tif files is greater than or equal to 4
    if indicator_name not in constant_indicator and len(tif_file_paths) >= 4 and country_phenology_single_crop is not None:

        # Number of files
        len_tif_file_paths = len(tif_file_paths)
        if 4 < len_tif_file_paths < 36:
            # Remove the excess part at the end of country_phenology_single_crop
            country_phenology_single_crop = country_phenology_single_crop[:len_tif_file_paths]

        # 2023-06-29 Determine the number of groups based on phenology length
        len_phenology = len(country_phenology_single_crop)

        split_tif_file_paths = np.array_split(tif_file_paths, len_phenology)
        tif_file_paths = []
        for file_paths, flag in zip(split_tif_file_paths, country_phenology_single_crop):
            if flag:
                tif_file_paths.extend(file_paths.tolist())  # Add all elements in each sublist

    # Step 2: Traverse all tif file paths, read image data one by one and save
    for input_img_path in tif_file_paths:
        img_data, _, _, _, _ = read_image_as_array(input_img_path)
        require_data = img_data[require_index]
        require_data_list.append(require_data)

        (filepath, filename) = os.path.split(input_img_path)
        require_data_image_name_list.append(filename)

    return require_data_list


def get_indicator_features(indicator_name='RAIN', data_year=2015, require_index=None,
                           country_phenology_single_crop=None):
    ith_indicator_data_list = read_indicators(indicator_name=indicator_name, data_year=data_year,
                                              require_index=require_index,
                                              country_phenology_single_crop=country_phenology_single_crop)
    if not ith_indicator_data_list:
        return []
    ith_indicator_data = np.asarray(ith_indicator_data_list).T
    ith_indicator_data_max = np.max(ith_indicator_data, axis=1).reshape([-1, 1])
    ith_indicator_data_min = np.min(ith_indicator_data, axis=1).reshape([-1, 1])
    ith_indicator_data_std = np.std(ith_indicator_data, axis=1).reshape([-1, 1])
    ith_indicator_data_sum = np.sum(ith_indicator_data, axis=1).reshape([-1, 1])
    ith_indicator_features = np.concatenate(
        [ith_indicator_data_max, ith_indicator_data_min, ith_indicator_data_std, ith_indicator_data_sum], axis=1)
    return ith_indicator_features


def get_indicator_features_temp(indicator_name='Temp', data_year=2015, require_index=None,
                                country_phenology_single_crop=None):
    ith_indicator_data_list = read_indicators(indicator_name=indicator_name, data_year=data_year,
                                              require_index=require_index,
                                              country_phenology_single_crop=country_phenology_single_crop)
    if not ith_indicator_data_list:
        return []
    ith_indicator_data = np.asarray(ith_indicator_data_list).T
    ith_indicator_data_max = np.max(ith_indicator_data, axis=1).reshape([-1, 1])
    ith_indicator_data_min = np.min(ith_indicator_data, axis=1).reshape([-1, 1])
    ith_indicator_data_std = np.std(ith_indicator_data, axis=1).reshape([-1, 1])
    ith_indicator_data_mean = np.mean(ith_indicator_data, axis=1).reshape([-1, 1])
    ith_indicator_features = np.concatenate(
        [ith_indicator_data_max, ith_indicator_data_min, ith_indicator_data_std, ith_indicator_data_mean], axis=1)

    return ith_indicator_features


def get_latlon_features(transform, require_index):
    """
    Get latitude and longitude coordinate features
    :param transform: Image coordinate transformation parameters
    :param require_index: Pixel index for which to get latitude and longitude coordinates
    :return: Latitude and longitude coordinate features
    """
    # Get pixel coordinates
    pixel_x = require_index[1]  # Note: In numpy arrays, the first index is row (y), the second index is column (x)
    pixel_y = require_index[0]

    # Use GDAL's Affine GeoTransform to convert pixel coordinates to latitude and longitude coordinates
    lon = transform[0] + pixel_x * transform[1] + pixel_y * transform[2]
    lat = transform[3] + pixel_x * transform[4] + pixel_y * transform[5]

    # Convert latitude and longitude coordinates to polar coordinates
    gx = np.sin(lon)*np.cos(lat)
    gy = np.sin(lat)
    gz = np.cos(lon)*np.cos(lat)

    # Concatenate latitude and longitude coordinates, one dimension is latitude, one dimension is longitude
    latlon_features = np.column_stack((gx, gy, gz))

    return latlon_features


def best_model_selection(X, y):
    print('best_model_selection')

    # 1-Read data, standardize, split training and test sets
    standard_scalar = StandardScaler()
    standard_scalar.fit(X)
    X = standard_scalar.transform(X)

    if len(y) > 5:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    else:
        X_train = X
        y_train = y
        X_test = X
        y_test = y

    # 2-Based on the training set, use grid search method to find the optimal parameters
    # 2.1-Define the parameter grid
    rf_params = {'n_estimators': [10, 50, 100, 200]}
    xgb_params = {'n_estimators': [10, 100, 200], 'learning_rate': [0.01, 0.03, 0.1], 'max_depth': [3, 6, 9]}
    cab_params = {'depth': [6, 10], 'learning_rate': [0.01, 0.03, 0.1], 'l2_leaf_reg': [1, 3, 5],
                  'iterations': [200, 400]}
    n_cv = 3  # Number of cross-validations

    # 2.2-Initialize the models
    rf_model = RandomForestRegressor(random_state=42)
    xgb_model = XGBRegressor()
    cab_model = CatBoostRegressor(verbose=False)

    # 2.3-Perform grid search
    # Only perform grid search when the number of samples is greater than 5
    if len(y) > 5:
        # 2.3-Perform grid search
        rf_gs = GridSearchCV(rf_model, rf_params, cv=n_cv, scoring="neg_mean_squared_error")
        xgb_gs = GridSearchCV(xgb_model, xgb_params, cv=n_cv, scoring="neg_mean_squared_error")
        cab_gs = GridSearchCV(cab_model, cab_params, cv=n_cv, scoring="neg_mean_squared_error")

        # Fit the data
        rf_gs.fit(X_train, y_train)
        xgb_gs.fit(X_train, y_train)
        cab_gs.fit(X_train, y_train)

        # Get the best model
        models = [rf_gs, xgb_gs, cab_gs]
        best_model = max(models, key=lambda model: model.best_score_)

        # Make predictions on the test set using the best model
        y_pred = best_model.predict(X_test)

        # Calculate MSE and R2 Score
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        print(f"Best model with RMSE of {rmse} and R2 Score of {r2}")
        if isinstance(best_model.best_estimator_, RandomForestRegressor):
            final_model = RandomForestRegressor(**best_model.best_params_)
            print(f"Best model is RandomForestRegressor")
        elif isinstance(best_model.best_estimator_, XGBRegressor):
            final_model = XGBRegressor(**best_model.best_params_)
            print(f"Best model is XGBRegressor")
        elif isinstance(best_model.best_estimator_, CatBoostRegressor):
            final_model = CatBoostRegressor(**best_model.best_params_, verbose=False)
            print(f"Best model is CatBoostRegressor")
        else:
            final_model = RandomForestRegressor(**best_model.best_params_)
            print(f"Best model is RandomForestRegressor")

    else:
        best_model = rf_model
        best_model.fit(X_train, y_train)
        # Make predictions on the test set using the best model
        y_pred = best_model.predict(X_test)

        # Calculate MSE and R2 Score
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        final_model = RandomForestRegressor(random_state=42)
        print(f"Best model is RandomForestRegressor with RMSE of {rmse} and R2 Score of {r2}")

    # 4-Based on the selected optimal model and optimal parameters, train the model using X and y to obtain the final model and feature importance
    final_model.fit(X, y)
    importance = final_model.feature_importances_

    return final_model, importance, r2, rmse, standard_scalar


def trained_with_trained_params(X, y, model_with_params):
    # Train the model based on previously trained model parameters
    print('train_model_with_trained_params')

    # 1-Read data, standardize, split training and test sets
    standard_scalar = StandardScaler()
    standard_scalar.fit(X)
    X = standard_scalar.transform(X)

    # Remove data where y is less than or equal to 0
    X = X[y > 0]
    y = y[y > 0]

    if len(y) > 5:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    else:
        X_train = X
        y_train = y
        X_test = X
        y_test = y

    model_with_params.fit(X_train, y_train)
    y_pred = model_with_params.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"Best model with RMSE of {rmse} and R2 Score of {r2}")

    # 4-Based on the selected optimal model and optimal parameters, train the model using X and y to obtain the final model and feature importance
    model_with_params.fit(X, y)
    importance = model_with_params.feature_importances_

    return model_with_params, importance, r2, rmse, standard_scalar


def save_model(directory, best_model, country_code, region_code, importances, r2, rmse, total_sample_num, crop_type, standard_scalar, feature_names, ith_country_name):
    # Construct the model filename
    model_filename = "trained_model_{}---{}-{}.pkl".format(country_code, region_code, crop_type)
    # Construct the full path for saving the model
    model_path = os.path.join(directory, model_filename)
    # Save the model to the specified path
    dump(best_model, model_path)
    print("Model saved to {}".format(model_path))

    # Save feature importances, R2, RMSE, total sample number, and model type
    model_type = type(best_model).__name__  # Get the type of the best model
    importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    importance_df['r2'] = [r2] + [''] * (len(importances) - 1)
    importance_df['rmse'] = [rmse] + [''] * (len(importances) - 1)
    importance_df['total_sample_num'] = [total_sample_num] + [''] * (len(importances) - 1)
    importance_df['model_type'] = [model_type] + [''] * (len(importances) - 1)
    importance_df['country_name'] = [ith_country_name] + [''] * (len(importances) - 1)  # Add country_name

    csv_filename = "trained_model_{}---{}-{}.csv".format(country_code, region_code, crop_type)
    csv_path = os.path.join(directory, csv_filename)
    importance_df.to_csv(csv_path, index=False)
    print("Feature importances, R2, RMSE, total sample number, and model type saved to {}".format(csv_path))

    # Save the StandardScaler
    standard_scalar_filename = "standard_scalar_{}---{}-{}.pkl".format(country_code, region_code, crop_type)
    standard_scalar_path = os.path.join(directory, standard_scalar_filename)
    dump(standard_scalar, standard_scalar_path)
    print("StandardScaler saved to {}".format(standard_scalar_path))
    print('-----------------------------------')


def load_model(directory, country_code, region_code, crop_type):
    # Construct the model filename
    model_filename = "trained_model_{}---{}-{}.pkl".format(country_code, region_code, crop_type)
    # Construct the full path for loading the model
    model_path = os.path.join(directory, model_filename)
    if not os.path.exists(model_path):
        print("Model not found at {}".format(model_path))
        return None, None
    # Load the model from the specified path
    loaded_model = load(model_path)
    print("Model loaded from {}".format(model_path))

    # Load the StandardScaler
    standard_scalar_filename = "standard_scalar_{}---{}-{}.pkl".format(country_code, region_code, crop_type)
    standard_scalar_path = os.path.join(directory, standard_scalar_filename)
    loaded_standard_scalar = load(standard_scalar_path)
    print("StandardScaler loaded from {}".format(standard_scalar_path))

    return loaded_model, loaded_standard_scalar


def load_trained_model_for_train(directory, country_code, region_code, crop_type):
    # Read the parameters of the original model and retrain
    # Load the model and the StandardScaler
    print('load_trained_model_for_train')
    model, standard_scaler = load_model(directory, country_code, region_code, crop_type)
    if model is None or standard_scaler is None:
        return None

    # Get the model's type and parameters
    model_type = type(model).__name__
    model_params = model.get_params()

    # Create a new model of the same type
    new_model = eval(model_type)(**model_params)

    return new_model


def check_model_exist(directory, country_code, region_code, crop_type):
    # Construct the model filename
    model_filename = "trained_model_{}---{}-{}.pkl".format(country_code, region_code, crop_type)
    # Construct the full path for loading the model
    model_path = os.path.join(directory, model_filename)
    if not os.path.exists(model_path):
        return False
    else:
        return True


def generate_feature_name(indicator_name, feature_data):
    indicator_feature_name = []
    if np.array(feature_data).size:
        for ith_dim in range(feature_data.shape[1]):
            indicator_feature_name.append(indicator_name + '_' + str(ith_dim + 1))
        return indicator_feature_name
    else:
        return []


def do_production_model_train_test(crop_type='Rice', data_year=2020, model='RF', special_mark=None, process_type='train'):
    """
    Use 2015 data to establish a regression model between yield and indicators, and regress the yield for the specified year
    :param crop_type: Crop type
    :param data_year: The year to be regressed
    :param model: Selected regression model
    :param special_mark: Give the generated result a special mark for easy identification
    :return:
    """
    print('\n Production regression')
    print('\n Crop type:', crop_type)
    starttime = datetime.datetime.now()
    # Read the matching of country labels and FAO table data to obtain valid country codes
    if process_type == 'train':
        valid_country_code_list, valid_country_name_list, _ = process_country_name.match_country_name(
            fao_datatype='harvest', crop_type=crop_type, data_year=2015)
    else:
        valid_country_code_list, valid_country_name_list, _ = process_country_name.match_country_name(
            fao_datatype='harvest', crop_type=crop_type, data_year=data_year)

    img_path_C_Code = "D:/Dataset/产量分解/CountryCodeRaster_2015-resample.tif"
    img_data_C_Code, im_width, im_height, _, _ = read_image_as_array(img_path_C_Code)

    # Read 2015 GAEZ harvested area data
    # Note: The harvested area data here is processed for consistency using FAO data.
    img_path_Area_2015 = r"G:\Dataset\产量分解\Crop_Data\Harvest_Area\HarvArea_{}_2015.tif".format(crop_type)
    img_data_Area_2015, _, _, _, _ = read_image_as_array(img_path_Area_2015)

    # Read 2015 GAEZ production data
    img_path_Prod_2015 = "G:\Dataset\产量分解\Crop_Data\GAEZ_Production\GAEZAct2015_Production_{}_Total.tif".format(
        crop_type)

    img_data_Prod_2015, _, _, transform, proj = read_image_as_array(img_path_Prod_2015)

    nodata_value_true = -3.39999995214e+038
    nodata_value = -3.30000e+038

    # Read 2020 harvested area data
    area_file_folder = 'D:/Dataset/产量分解/Crop_Data/Harvest_Area/'
    img_path_Area_2020 = area_file_folder + 'HarvArea_{}_{}.tif'.format(crop_type, data_year)
    img_data_Area_2020, im_width, im_height, transform, proj = read_image_as_array(img_path_Area_2020)

    img_data_Prod_2020 = img_data_Prod_2015.copy()
    img_data_Prod_2020[np.where(img_data_Prod_2015 > nodata_value)] = 0


    ################################################################
    # For production regression based on agro-ecological zones
    country_names_and_AEZs_dict = read_AEZs_data.get_AEZs_info()
    AEZs_image_data = read_AEZs_data.read_AEZs_image()
    #################################################################
    if data_year in [2000, 2001, 2010, 2011, 2012]:
        use_vci = False
    else:
        use_vci = True

    if use_vci:
        model_save_path = r'G:\Dataset\产量分解\Crop_Data\Trained_models'
    else:
        model_save_path = r'G:\Dataset\产量分解\Crop_Data\Trained_models_no_vci'

    result_production_file_folder = 'D:/Dataset/产量分解/Crop_Data/Predict_Production'

    counter = 0
    # Process each country according to the country code

    for i in range(len(valid_country_code_list)):

        ith_country_code = valid_country_code_list[i]
        ith_country_name = valid_country_name_list[i]
        marker_for_retrain = False

        if ith_country_name == 'Russian Federation':
            ith_country_name = 'Russia'
        if ith_country_name == 'Viet Nam':
            ith_country_name = 'Vietnam'
        if ith_country_name == 'United States of America':
            ith_country_name = 'United States'
        if ith_country_name == 'Syrian Arab Republic':
            ith_country_name = 'Syria'
        if ith_country_name == 'Iran (Islamic Republic of)':
            ith_country_name = 'Iran'

        print('\nCountry name:', valid_country_name_list[i])

        # Read phenology table
        ith_country_phenology = get_country_phenology(valid_country_name_list[i],
                                                      crop_type)  # Here must use valid_country_name_list[i], because the phenology table needs to be consistent with FAO and countrycode_2015, while ith_country_name is modified to be consistent with AEZs
        ith_country_phenology_36 = get_country_phenology_36(valid_country_name_list[i], crop_type)
        if np.sum(ith_country_phenology) == 0:
            print('This country has no phenology data for this crop, skipping this country')
            continue


        if ith_country_code == 9999 and data_year <= 2011:  # Special treatment for Sudan in 2010-2011
            ith_code_index = np.where(np.logical_or(img_data_C_Code == 6, img_data_C_Code == 74))
        else:
            ith_code_index = np.where(img_data_C_Code == ith_country_code)
        # If ith_code_index=[], it means the country has no data in the current image, so skip this country
        if len(ith_code_index[0]) == 0:
            print('This country has no data in the current image')
            continue

        if ith_country_code == 9999 and data_year <= 2011:  # Special treatment for Sudan in 2010-2011
            index_of_region_to_process = [6, 74]
        else:
            index_of_region_to_process = [ith_country_code]

        check_if_has_aez = False
        if ith_country_name in country_names_and_AEZs_dict.keys():
            check_if_has_aez = True
            print('Regression by ecological zone')
            ith_AEZs_codes = country_names_and_AEZs_dict[ith_country_name]
            print('Number of ecological zones is {}'.format(len(ith_AEZs_codes)))
            index_of_region_to_process = ith_AEZs_codes

        # Regression for each region
        for jth_region_code in index_of_region_to_process:
            # counter += 1
            # if counter >= 3:
            #     break
            print('region code is {}'.format(jth_region_code))
            if check_if_has_aez:
                ith_code_index = np.where(AEZs_image_data == jth_region_code)
            else:
                ith_code_index = np.where(img_data_C_Code == jth_region_code)

            # Sometimes model training fails and needs to be retrained. Here, check if the model exists. If it exists, skip it, i.e., only train models that don't exist
            if process_type == 'train':
                if not marker_for_retrain:
                    if check_model_exist(model_save_path, ith_country_code, jth_region_code, crop_type):
                        print('Model already exists, skipping')
                        continue
            else:
                if ith_country_code != 9999 and data_year <= 2011:
                    if not check_model_exist(model_save_path, ith_country_code, jth_region_code, crop_type):
                        print('Model does not exist, skipping')
                        continue

            feature_names = []

            ###############################################################################################
            # This part is for common features, including harvested area, soil texture, irrigation type, latitude and longitude, elevation, elevation variation coefficient

            # Get the production of specified pixels in 2015
            ith_Prod_2015 = img_data_Prod_2015[ith_code_index]

            feature_names.append('Area')

            # Read soil texture data, feature dimension = 3
            ith_soil_features = read_indicators(indicator_name='Soil', data_year=2015,
                                                require_index=ith_code_index, country_phenology_single_crop=None)
            ith_soil_features = np.asarray(ith_soil_features).T
            feature_names.extend(['Soil_Clay', 'Soil_Sand', 'Soil_Silt'])

            # Read irrigation type data
            ith_irrigation_features = read_indicators(indicator_name='Irrigation', data_year=2015,
                                                      require_index=ith_code_index,
                                                      country_phenology_single_crop=None)
            ith_irrigation_features = np.asarray(ith_irrigation_features)
            # One-hot encode the irrigation type data, feature dimension = 3
            ith_irrigation_data_flat = ith_irrigation_features.flatten().reshape(-1, 1)
            encoder = OneHotEncoder(sparse=False)
            ith_irrigation_data_onehot = encoder.fit_transform(ith_irrigation_data_flat)
            ith_irrigation_data_onehot = ith_irrigation_data_onehot.reshape(ith_irrigation_features.shape[0],
                                                                            ith_irrigation_features.shape[1], -1)
            feature_names.extend(generate_feature_name('Water', ith_irrigation_data_onehot[0]))

            # Read latitude and longitude coordinate features, feature dimension = 2
            # Based on im_width, im_height, transform parameters, convert the pixel coordinates represented by ith_code_index to latitude and longitude coordinates
            ith_latlon_features = get_latlon_features(transform, ith_code_index)
            feature_names.extend(['Locate_gx', 'Locate_gy', 'Locate_gz'])

            # Read DEM data, feature dimension = 1
            ith_dem_features = read_indicators(indicator_name='DEM', data_year=2015,
                                               require_index=ith_code_index, country_phenology_single_crop=None)
            ith_dem_features = np.asarray(ith_dem_features).T
            feature_names.append('DEM_Elevation')

            # Read elevation variation coefficient data, feature dimension = 1
            ith_variation_features = read_indicators(indicator_name='Variation', data_year=2015,
                                                     require_index=ith_code_index,
                                                     country_phenology_single_crop=None)
            ith_variation_features = np.asarray(ith_variation_features).T
            feature_names.append('DEM_Variation')

            ###############################################################################################

            def fetch_target_year_features(year, area_data, code_index, country_phenology, country_phenology_36,
                                           use_vci=True):
                # Get harvested area for the year
                area = area_data[code_index]
                area = np.asarray(area).reshape([-1, 1])

                # Get NPP for all pixels of the year
                npp_data_list = read_indicators(indicator_name='NPP', data_year=year, require_index=code_index,
                                                country_phenology_single_crop=country_phenology)
                npp_features = np.asarray(
                    npp_data_list).T  # Convert to format where rows are pixels and columns are features
                feature_names.extend(generate_feature_name('NPP', npp_features))

                # Get RAIN features for specified pixels of the year
                rain_features = get_indicator_features(indicator_name='RAIN', data_year=year, require_index=code_index,
                                                       country_phenology_single_crop=country_phenology_36)
                feature_names.extend(generate_feature_name('RAIN', rain_features))

                # Get PAR features for specified pixels of the year
                par_features = get_indicator_features(indicator_name='PAR', data_year=year, require_index=code_index,
                                                      country_phenology_single_crop=country_phenology_36)
                feature_names.extend(generate_feature_name('PAR', par_features))

                # Get TMP features for specified pixels of the year
                tmp_features = get_indicator_features_temp(indicator_name='TMP', data_year=year,
                                                           require_index=code_index,
                                                           country_phenology_single_crop=country_phenology_36)
                feature_names.extend(generate_feature_name('TMP', tmp_features))

                # Read GLASS NPP features for specified pixels of the year
                # Read GLASS NPP features for specified pixels of the year
                GLASS_NPP = read_indicators(indicator_name='GLASS_NPP', data_year=year, require_index=code_index,
                                            country_phenology_single_crop=None)
                GLASS_NPP = np.asarray(GLASS_NPP).T
                feature_names.append('GLASS_NPP')

                # Get GLASS_LAI features for specified pixels of the year
                GLASS_LAI = get_indicator_features(indicator_name='GLASS_LAI', data_year=year, require_index=code_index,
                                                   country_phenology_single_crop=country_phenology)
                feature_names.extend(generate_feature_name('GLASS_LAI', GLASS_LAI))

                # Get VCI features for specified pixels of the year
                if use_vci:
                    vci_features = read_indicators(indicator_name='VCI', data_year=year, require_index=code_index,
                                                   country_phenology_single_crop=country_phenology)
                    vci_features = np.asarray(vci_features).T
                    feature_names.extend(generate_feature_name('VCI', vci_features))

                # Get CALF features for specified pixels of the year
                calf_features = read_indicators(indicator_name='CALF', data_year=year, require_index=code_index,
                                                country_phenology_single_crop=country_phenology)
                calf_features = np.asarray(calf_features).T
                feature_names.extend(generate_feature_name('CALF', calf_features))

                # Merge all features
                if use_vci:
                    all_features = [area, ith_soil_features, ith_irrigation_data_onehot[0], ith_latlon_features,
                                    ith_dem_features, ith_variation_features, npp_features,
                                    rain_features, par_features,
                                    tmp_features, GLASS_NPP, GLASS_LAI, vci_features,
                                    calf_features]
                else:  # Data for 2010-2012 does not have VCI features
                    all_features = [area, ith_soil_features, ith_irrigation_data_onehot[0], ith_latlon_features,
                                    ith_dem_features, ith_variation_features, npp_features,
                                    rain_features, par_features,
                                    tmp_features, GLASS_NPP, GLASS_LAI,
                                    calf_features]

                features = np.concatenate(all_features, axis=1)

                # Get indices where there are nodata or infinite or NaN values
                bad_indexes = np.where((features <= nodata_value) | np.isinf(features) | np.isnan(features))
                # Set values where there are nodata or infinite or NaN to 0
                features[bad_indexes] = 0

                return features

                ###############################################################################################

            if process_type == 'train':

                # Use 2015 data to train the model
                ith_features_2015 = fetch_target_year_features(2015, img_data_Area_2015, ith_code_index,
                                                               ith_country_phenology, ith_country_phenology_36,
                                                               use_vci=use_vci)

                # Remove data where the first dimension feature is 0
                ith_Prod_2015 = ith_Prod_2015[ith_features_2015[:, 0] != 0]
                ith_features_2015 = ith_features_2015[ith_features_2015[:, 0] != 0]

                X_2015 = ith_features_2015
                y_2015 = ith_Prod_2015

                if len(y_2015) <= 10:
                    print('Too few samples, not training')
                    continue
                else:
                    total_samples = len(y_2015)
                    print('Number of training samples: {}'.format(total_samples))
                ########################################## End of training data reading ##########################################

                ########################################## Start model training ##########################################

                trained_old_model_path = r'G:\Dataset\产量分解\Crop_Data\Trained_models_Best/'
                model_with_params = load_trained_model_for_train(trained_old_model_path, ith_country_code,
                                                                 jth_region_code, crop_type)
                if model_with_params is None:
                    # if marker_for_retrain is False:
                    #     continue
                    print('Optimal model parameters do not exist, retraining')
                    # Train and select the optimal model
                    best_model, feature_importance, r2, rmse, standard_scalar = best_model_selection(X_2015, y_2015)
                else:
                    best_model, feature_importance, r2, rmse, standard_scalar = trained_with_trained_params(X_2015,
                                                                                                            y_2015,
                                                                                                            model_with_params)

                # Save the model locally
                save_model(model_save_path, best_model, ith_country_code, jth_region_code, feature_importance, r2, rmse,
                           total_samples, crop_type, standard_scalar, feature_names, ith_country_name)
                #############################################################################

            else:
                # Load local model
                if ith_country_code == 9999 and data_year <= 2011:  # Special treatment for Sudan
                    loaded_model, loaded_standard_scalar = load_model(model_save_path, jth_region_code,
                                                                      jth_region_code, crop_type)
                else:
                    loaded_model, loaded_standard_scalar = load_model(model_save_path, ith_country_code,
                                                                      jth_region_code, crop_type)
                if loaded_model is None:
                    print('Model does not exist, skipping')
                    continue

                # Use 2020 data for prediction
                # Read test data
                ith_features_2020 = fetch_target_year_features(data_year, img_data_Area_2020, ith_code_index,
                                                               ith_country_phenology, ith_country_phenology_36,
                                                               use_vci=use_vci)

                ith_features_2020 = loaded_standard_scalar.transform(ith_features_2020)
                # Predict 2020 production
                ith_Prod_2020_predict = loaded_model.predict(ith_features_2020)

                # Get indices where ith_Prod_2015 is less than or equal to 0
                index_less_than_zero = np.where(ith_Prod_2015 <= 0)

                # Set corresponding indices in ith_Prod_2020_predict to the values at the corresponding positions in ith_Prod_2015
                ith_Prod_2020_predict[index_less_than_zero] = ith_Prod_2015[index_less_than_zero]

                # Assign values
                img_data_Prod_2020[ith_code_index] = ith_Prod_2020_predict

                ################################## End of prediction ##################################

            if process_type == 'test':
                # Set values less than 0 to Nodata
                index_below_zero = np.where(img_data_Prod_2020 < 0)
                img_data_Prod_2020[index_below_zero] = nodata_value_true

                # Create 2020 production data

                if special_mark is None:
                    img_path_Prod_2020 = "{}/Production_{}_{}_before_revised.tif".format(result_production_file_folder,
                                                                                         crop_type,
                                                                                         data_year)
                else:
                    img_path_Prod_2020 = "{}/Production_{}_{}_before_revised_{}.tif".format(
                        result_production_file_folder, crop_type,
                        data_year, special_mark)
                driver = gdal.GetDriverByName("GTiff")
                dataset = driver.Create(img_path_Prod_2020, int(im_width), int(im_height), 1, gdal.GDT_Float32)
                if (dataset != None):
                    dataset.SetGeoTransform(transform)  # Write affine transformation parameters
                    dataset.SetProjection(proj)  # Write projection

                dataset.GetRasterBand(1).SetNoDataValue(nodata_value_true)

                # Write data
                dataset.GetRasterBand(1).WriteArray(img_data_Prod_2020)
                del dataset

            endtime = datetime.datetime.now()
            print('\nRunning time: {} seconds'.format((endtime - starttime).seconds))