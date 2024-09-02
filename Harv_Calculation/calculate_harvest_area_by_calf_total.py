# -*- coding: utf-8 -*-
"""
Calculate Harvest Area by CALF Total

This script generates harvest area images for specified crop types and years.
It uses CALF_Total as a weight and is based on cal_country_area.py.

@File    : calculate_harvest_area_by_calf_total.py
@Contact : qinxl@aircas.ac.cn
@License : (C)Copyright 2021-2022, CropWatch Group
@Modify Time : 2023
@Author : Qin Xingli
@Version : 1.0
"""

import os
import datetime
from typing import List, Tuple

import numpy as np
import pandas as pd
from osgeo import gdal

import process_country_name
import read_AEZs_data
from production_index_calculation import read_indicators
from DataProcessTools import interpolate_arrays_for_Calf_Total, find_outliers_for_Calf_Total

# Set GDAL environment variable
os.environ['PROJ_LIB'] = r'C:\ProgramData\Anaconda3\envs\MyProductionIndex\Lib\site-packages\osgeo\data\proj'


def read_image_as_array(img_path: str) -> Tuple[np.ndarray, int, int, Tuple, str, float]:
    """
    Read input image data and return as numpy array along with metadata.

    Args:
        img_path (str): Path to the image file.

    Returns:
        tuple: Image data as numpy array, width, height, transform, projection, and nodata value.

    Raises:
        IOError: If the image file cannot be opened.
    """
    gdal.AllRegister()
    img_data = gdal.Open(img_path, gdal.GA_ReadOnly)
    if img_data is None:
        raise IOError(f'Failed to open {img_path}')

    im_width = img_data.RasterXSize
    im_height = img_data.RasterYSize
    transform = img_data.GetGeoTransform()
    proj = img_data.GetProjection()

    band = img_data.GetRasterBand(1)
    im_data = band.ReadAsArray(0, 0, im_width, im_height)
    nodata_value = band.GetNoDataValue()

    return im_data, im_width, im_height, transform, proj, nodata_value


# List of countries to be updated (as of 2023-07-18)
COUNTRIES_TO_UPDATE = [
    "Zambia", "Kenya", "Ethiopia", "Morocco", "Uzbekistan", "Kyrgyzstan",
    "Kazakhstan", "Turkey", "Romania", "Ukraine", "Italy", "France",
    "Australia", "Viet Nam", "Bangladesh", "Syrian Arab Republic",
    "Hungary", "Belarus", "Malawi", "Cameroon", "Nigeria", "Myanmar",
    "Thailand", "Indonesia", "Russian Federation", "Iran (Islamic Republic of)"
]

# Countries to be updated as of 2023-07-21
COUNTRIES_TO_UPDATE_0721 = ['Sudan (former)']
COUNTRIES_TO_UPDATE_0721_2 = ['Sudan', 'South Sudan']


def calculate_harvest_area(crop_type: str = 'Rice', data_year: int = 2020) -> None:
    """
    Calculate and generate harvest area image for a specific crop type and year.

    Args:
        crop_type (str): Type of crop. Default is 'Rice'.
        data_year (int): Year for which to calculate harvest area. Default is 2020.
    """
    print(f'\nGenerating harvest area image for {crop_type} in {data_year}')
    start_time = datetime.datetime.now()

    # Set file paths
    harv_file_folder = r"D:\Dataset\产量分解\Crop_Data\Harvest_Area"
    calf_total_folder = r"D:\Dataset\产量分解\Crop_Data\Calf_Total"
    result_file_folder = r'D:\Dataset\产量分解\Crop_Data\Harvest_Area_part_updated_0721'

    # Get valid country codes, names, and values
    valid_country_codes, valid_country_names, valid_country_values = process_country_name.match_country_name(
        fao_datatype='harvest', crop_type=crop_type, data_year=data_year)

    # Read country label image
    country_label_path = r"D:\Dataset\产量分解\Support_Data_for_CPI\CountryCodeRaster_2015-resample.tif"
    country_label_data, im_width, im_height, _, _, _ = read_image_as_array(country_label_path)

    # Read harvest area image for the base year (2015)
    harvest_file_path = os.path.join(harv_file_folder, f'HarvArea_{crop_type}_2015.tif')
    harvest_data, _, _, transform, proj, nodata_value = read_image_as_array(harvest_file_path)
    nodata_value = nodata_value or 0

    # Read CALF Total for base and target years
    calf_total_base_path = os.path.join(calf_total_folder, f'Crop_Total_Redistribute_2015_{crop_type}.tif')
    calf_total_base_data, _, _, _, _, _ = read_image_as_array(calf_total_base_path)

    calf_total_target_path = os.path.join(calf_total_folder, f'Crop_Total_Redistribute_{data_year}_{crop_type}.tif')
    calf_total_target_data, _, _, _, _, _ = read_image_as_array(calf_total_target_path)

    # Create intermediate data array
    data_middle = np.zeros([im_height, im_width], dtype=np.float32)

    # Get AEZs information
    country_names_and_AEZs_dict = read_AEZs_data.get_AEZs_info()
    AEZs_image_data = read_AEZs_data.read_AEZs_image()
    AEZs_names = read_AEZs_data.get_AEZs_names_dict()

    for country_code, country_name, country_value in zip(valid_country_codes, valid_country_names,
                                                         valid_country_values):
        print(f'\nProcessing country: {country_name}')

        # Normalize country names
        country_name = normalize_country_name(country_name)

        # Get country pixels
        country_pixels = get_country_pixels(country_code, country_label_data, data_year)

        # Process country data
        process_country_data(country_name, country_code, country_pixels, country_value,
                             harvest_data, calf_total_base_data, calf_total_target_data,
                             country_names_and_AEZs_dict, AEZs_image_data, AEZs_names,
                             data_middle, crop_type, nodata_value, im_width, im_height, transform, proj)

    # Apply final processing to data_middle
    data_middle[data_middle < 0] = nodata_value
    data_middle[harvest_data == 0] = 0
    data_middle[harvest_data == nodata_value] = 0

    # Save the processed data as a new GeoTIFF file
    save_geotiff(data_middle, im_width, im_height, transform, proj, nodata_value,
                 crop_type, data_year, harv_file_folder)

    end_time = datetime.datetime.now()
    print(f'\nExecution time: {(end_time - start_time).seconds} seconds')

def normalize_country_name(country_name: str) -> str:
    """Normalize country names for consistency."""
    name_mapping = {
        'Russian Federation': 'Russia',
        'Viet Nam': 'Vietnam',
        'United States of America': 'United States',
        'Syrian Arab Republic': 'Syria',
        'Iran (Islamic Republic of)': 'Iran'
    }
    return name_mapping.get(country_name, country_name)

def get_country_pixels(country_code: int, country_label_data: np.ndarray, data_year: int) -> np.ndarray:
    """Get pixels corresponding to a country."""
    if country_code == 9999 and data_year <= 2011:  # Special handling for Sudan 2010-2011
        return np.logical_or(country_label_data == 6, country_label_data == 74)
    return country_label_data == country_code

def process_country_data(country_name: str, country_code: int, country_pixels: np.ndarray,
                         country_value: float, harvest_data: np.ndarray,
                         calf_total_base_data: np.ndarray, calf_total_target_data: np.ndarray,
                         country_names_and_AEZs_dict: dict, AEZs_image_data: np.ndarray,
                         AEZs_names: dict, data_middle: np.ndarray, crop_type: str,
                         nodata_value: float, im_width: int, im_height: int,
                         transform: Tuple, proj: str) -> None:
    """Process data for a single country."""
    if country_name in country_names_and_AEZs_dict:
        print('Performing regression by ecological zone')
        region_codes = country_names_and_AEZs_dict[country_name]
        region_data = AEZs_image_data
        region_type = 'AEZs'
    else:
        region_codes = [country_code]
        region_data = country_pixels
        region_type = 'Country'

    area_sum = 0
    region_areas = []
    region_crop_totals = []
    region_crop_totals_2015 = []
    region_area_ratios_init = []
    region_area_ratios = []
    pixel_area_ratios = []

    for region_code in region_codes:
        region_pixels = region_data == region_code

        process_region_data(region_code, region_pixels, harvest_data, calf_total_base_data,
                            calf_total_target_data, area_sum, region_areas, region_crop_totals,
                            region_crop_totals_2015, region_area_ratios_init, pixel_area_ratios,
                            crop_type, country_name, region_type, AEZs_names)

    calculate_region_area_ratios(region_areas, area_sum, region_crop_totals,
                                 region_crop_totals_2015, region_area_ratios_init,
                                 region_area_ratios)

    apply_area_ratios(region_codes, region_data, pixel_area_ratios, region_area_ratios,
                      country_value, harvest_data, data_middle)

def process_region_data(region_code: int, region_pixels: np.ndarray, harvest_data: np.ndarray,
                        calf_total_base_data: np.ndarray, calf_total_target_data: np.ndarray,
                        area_sum: float, region_areas: List[float], region_crop_totals: List[float],
                        region_crop_totals_2015: List[float], region_area_ratios_init: List[float],
                        pixel_area_ratios: List[np.ndarray], crop_type: str, country_name: str,
                        region_type: str, AEZs_names: dict) -> None:
    """Process data for a single region within a country."""
    if region_type == 'AEZs':
        print(f"Ecological zone name: {AEZs_names[region_code]}")

    region_harvest_2015 = harvest_data[region_pixels]
    region_harvest_2015_sum = np.sum(region_harvest_2015[region_harvest_2015 != 0])

    pixel_area_ratio_init = calculate_pixel_area_ratio(region_harvest_2015, region_harvest_2015_sum)

    region_calf_total_2015 = calf_total_base_data[region_pixels]
    region_calf_total_target = calf_total_target_data[region_pixels]

    region_calf_total_2015 = handle_outliers_and_interpolate(region_calf_total_2015,
                                                             region_harvest_2015,
                                                             region_calf_total_target)

    pixel_area_ratio = calculate_updated_pixel_area_ratio(pixel_area_ratio_init,
                                                          region_calf_total_2015,
                                                          region_calf_total_target,
                                                          region_harvest_2015)

    if crop_type == 'Soybean' and country_name in ['Germany', 'Angola'] and region_code in [58, 7]:
        region_harvest_2015_sum = 0

    area_sum += region_harvest_2015_sum
    region_areas.append(region_harvest_2015_sum)
    region_crop_totals.append(np.sum(region_calf_total_target))
    region_crop_totals_2015.append(np.sum(region_calf_total_2015))
    pixel_area_ratios.append(pixel_area_ratio)

def calculate_pixel_area_ratio(region_harvest: np.ndarray, region_harvest_sum: float) -> np.ndarray:
    """Calculate the initial pixel area ratio for a region."""
    region_harvest[region_harvest == 0] = 0
    return region_harvest / region_harvest_sum if region_harvest_sum != 0 else region_harvest

def handle_outliers_and_interpolate(calf_total_2015: np.ndarray, harvest_2015: np.ndarray,
                                    calf_total_target: np.ndarray) -> np.ndarray:
    """Handle outliers and perform interpolation on CALF total data."""
    outliers_index = find_outliers_for_Calf_Total.find_outliers(calf_total_2015, harvest_2015, calf_total_target)
    if len(outliers_index) > 0:
        calf_total_2015 = interpolate_arrays_for_Calf_Total.inverse_distance_weighted_interpolation_by_index(calf_total_2015, outliers_index)

    if interpolate_arrays_for_Calf_Total.need_interpolation(calf_total_2015, harvest_2015, calf_total_target):
        calf_total_2015 = interpolate_arrays_for_Calf_Total.interpolate_arrays(calf_total_2015, harvest_2015, calf_total_target)

    return calf_total_2015

def calculate_updated_pixel_area_ratio(pixel_area_ratio_init: np.ndarray, calf_total_2015: np.ndarray,
                                       calf_total_target: np.ndarray, harvest_2015: np.ndarray) -> np.ndarray:
    """Calculate the updated pixel area ratio based on CALF total data."""
    if np.sum(calf_total_target) == 0:
        return pixel_area_ratio_init

    calf_total_sum_base = np.sum(calf_total_2015)
    calf_total_sum_target = np.sum(calf_total_target)
    pixel_area_ratio = np.zeros_like(pixel_area_ratio_init)

    non_zero_indices = calf_total_2015 != 0
    pixel_area_ratio[non_zero_indices] = (calf_total_sum_base / calf_total_sum_target * calf_total_target[non_zero_indices]) / calf_total_2015[non_zero_indices]

    # Apply maximum constraint
    max_constraint = np.maximum(20 / harvest_2015, pixel_area_ratio_init)
    pixel_area_ratio = np.minimum(pixel_area_ratio, max_constraint)

    return pixel_area_ratio_init * pixel_area_ratio

def calculate_region_area_ratios(region_areas: List[float], area_sum: float,
                                 region_crop_totals: List[float], region_crop_totals_2015: List[float],
                                 region_area_ratios_init: List[float], region_area_ratios: List[float]) -> None:
    """Calculate area ratios for each region."""
    for area, crop_total, crop_total_2015 in zip(region_areas, region_crop_totals, region_crop_totals_2015):
        area_ratio_init = area / area_sum if area_sum != 0 else 0
        region_area_ratios_init.append(area_ratio_init)

        if sum(region_crop_totals) == 0 or crop_total_2015 == 0:
            region_area_ratios.append(area_ratio_init)
        else:
            area_ratio = area_ratio_init * sum(region_crop_totals_2015) * crop_total / sum(region_crop_totals) / crop_total_2015
            region_area_ratios.append(area_ratio)

    if sum(region_area_ratios) != 0:
        region_area_ratios[:] = [ratio / sum(region_area_ratios) for ratio in region_area_ratios]

def apply_area_ratios(region_codes: List[int], region_data: np.ndarray,
                      pixel_area_ratios: List[np.ndarray], region_area_ratios: List[float],
                      country_value: float, harvest_data: np.ndarray, data_middle: np.ndarray) -> None:
    """Apply calculated area ratios to update the data_middle array."""
    for region_code, pixel_area_ratio, region_area_ratio in zip(region_codes, pixel_area_ratios, region_area_ratios):
        region_pixels = region_data == region_code
        region_area = region_area_ratio * country_value
        pixel_area = region_area * pixel_area_ratio

        # Apply constraint
        harvest_2015_pixel = harvest_data[region_pixels]
        if harvest_2015_pixel.size != 0:
            harvest_2015_pixel_max = np.max(harvest_2015_pixel)
            harvest_2015_pixel_sum = np.sum(harvest_2015_pixel[harvest_2015_pixel > 0])
            if harvest_2015_pixel_sum != 0:
                max_constraint = max(harvest_2015_pixel_max * region_area / harvest_2015_pixel_sum, harvest_2015_pixel_max)
                pixel_area = np.minimum(pixel_area, max_constraint)

        data_middle[region_pixels] = pixel_area

def save_geotiff(data: np.ndarray, width: int, height: int, transform: Tuple,
                 proj: str, nodata_value: float, crop_type: str, year: int, output_folder: str) -> None:
    """Save the processed data as a GeoTIFF file."""
    output_path = os.path.join(output_folder, f'HarvArea_{crop_type}_{year}.tif')
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(output_path, width, height, 1, gdal.GDT_Float32)

    if dataset is not None:
        dataset.SetGeoTransform(transform)
        dataset.SetProjection(proj)

    if nodata_value is not None and nodata_value != 0:
        dataset.GetRasterBand(1).SetNoDataValue(nodata_value)

    dataset.GetRasterBand(1).WriteArray(data)
    dataset.FlushCache()

def main():
    """Main function to process harvest areas for multiple crop types and years."""
    crop_types = ['Maize', 'Wheat', 'Rice', 'Soybean']
    years = range(2022, 2023)

    for crop_type in crop_types:
        for year in years:
            print(f'Processing crop type: {crop_type}, year: {year}')
            calculate_harvest_area(crop_type=crop_type, data_year=year)

if __name__ == '__main__':
    main()