#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Calibrate the calculated production using FAOSTAT data to ensure the sum of national production is accurate
"""
from osgeo import gdal
import numpy as np
import process_country_name
import os
import datetime

os.environ['PROJ_LIB'] = r'C:\ProgramData\Anaconda3\envs\MyProductionIndex\Lib\site-packages\osgeo\data\proj'

NODATA_VALUE = -3.39999995214e+038


def read_image_as_array(img_path):
    """Read input image data"""
    gdal.AllRegister()
    img_data = gdal.Open(img_path, gdal.GA_ReadOnly)
    if img_data is None:
        raise ValueError(f'{img_path} failed to open!')

    im_width = img_data.RasterXSize
    im_height = img_data.RasterYSize
    transform = img_data.GetGeoTransform()
    proj = img_data.GetProjection()

    ith_band = img_data.GetRasterBand(1)
    im_data = ith_band.ReadAsArray(0, 0, im_width, im_height)

    return im_data, im_width, im_height, transform, proj


def mask_nodata(img_data, base_image_path):
    """Mask nodata values based on a reference image"""
    base_image = gdal.Open(base_image_path)
    nodata_value_base = base_image.GetRasterBand(1).GetNoDataValue()
    base_image_data = base_image.GetRasterBand(1).ReadAsArray()
    index_nodata = np.where(base_image_data == nodata_value_base)
    img_data[index_nodata] = nodata_value_base
    return img_data


def process_country_production(img_data_result, img_data_C_Code, img_data_Prod_2020_init,
                               country_code, country_value_2020, data_year):
    """Process production data for a single country"""
    if country_code == 9999 and data_year in [2010, 2011]:  # Special handling for Sudan in 2010-2011
        ith_code_index = np.where(np.logical_or(img_data_C_Code == 6, img_data_C_Code == 74))
    else:
        ith_code_index = np.where(img_data_C_Code == country_code)

    ith_C_Prod_2020_init = img_data_Prod_2020_init[ith_code_index]
    ith_C_value_Sum_2020_init = np.sum(ith_C_Prod_2020_init[np.where(ith_C_Prod_2020_init != NODATA_VALUE)])
    ith_C_value_Sum_2020_FAO = float(country_value_2020) / 1000  # Convert to 1000 tonnes

    if ith_C_value_Sum_2020_init <= 0:
        img_data_result[ith_code_index] = 0
    else:
        img_data_result[ith_code_index] = (img_data_result[ith_code_index] /
                                           ith_C_value_Sum_2020_init *
                                           ith_C_value_Sum_2020_FAO)

    return img_data_result


def do_data_consistency(crop_type='Rice', data_year=2020, if_revised=True):
    """Main function to revise production data"""
    print(f'\nProduction optimization')
    print(f'\nCrop type: {crop_type}')
    starttime = datetime.datetime.now()

    valid_country_code_list, valid_country_name_list, valid_country_value_list_2020 = \
        process_country_name.match_country_name(fao_datatype='Production', crop_type=crop_type,
                                                if_revised=if_revised, data_year=data_year)

    img_path_C_Code = "D:/Dataset/产量分解/CountryCodeRaster_2015-resample.tif"
    img_data_C_Code, im_width, im_height, _, _ = read_image_as_array(img_path_C_Code)

    result_file_folder = 'D:/Dataset/产量分解/Crop_Data/Predict_Production/'
    img_path_Prod_2020_init = f"{result_file_folder}/Production_{crop_type}_{data_year}_before_revised.tif"
    img_data_Prod_2020_init, _, _, transform, proj = read_image_as_array(img_path_Prod_2020_init)

    area_file_folder = 'D:/Dataset/产量分解/Crop_Data/Harvest_Area/'
    img_path_area_2020 = f'{area_file_folder}/HarvArea_{crop_type}_{data_year}.tif'
    img_data_area_2020, _, _, _, _ = read_image_as_array(img_path_area_2020)

    img_data_result = np.asarray(img_data_Prod_2020_init, dtype=np.float32)

    # Process based on harvest area data
    img_data_result[img_data_area_2020 == NODATA_VALUE] = NODATA_VALUE
    img_data_result[img_data_area_2020 == 0] = 0

    # Process each country
    for i, country_code in enumerate(valid_country_code_list):
        img_data_result = process_country_production(img_data_result, img_data_C_Code, img_data_Prod_2020_init,
                                                     country_code, valid_country_value_list_2020[i], data_year)

    # Set values below zero to NODATA_VALUE
    img_data_result[img_data_result < 0] = NODATA_VALUE

    # Create output file
    result_file_folder = 'G:/Dataset/产量分解/Crop_Data/Revised_Production'
    img_path_Prod_2020_revised = f"{result_file_folder}/Production_{crop_type}_{data_year}.tif"

    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(img_path_Prod_2020_revised, int(im_width), int(im_height), 1, gdal.GDT_Float32)
    if dataset is not None:
        dataset.SetGeoTransform(transform)
        dataset.SetProjection(proj)
    dataset.GetRasterBand(1).SetNoDataValue(NODATA_VALUE)
    dataset.GetRasterBand(1).WriteArray(img_data_result)
    del dataset

    endtime = datetime.datetime.now()
    print(f'\nRunning time: {(endtime - starttime).seconds} seconds')


if __name__ == "__main__":
    do_data_consistency()