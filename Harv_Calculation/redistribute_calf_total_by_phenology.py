# -*- coding: utf-8 -*-
"""
This script redistributes the CALF (Cropped Arable Land Fraction) Total by phenology.
It adjusts the seasonal scale "CALF Total" (total planted area for each pixel) based on crop phenology.
"""

import os
import numpy as np
import pandas as pd
from osgeo import gdal
import process_country_name


def read_image_as_array(img_path):
    """
    Read input image data and return as numpy array along with metadata.

    Args:
        img_path (str): Path to the image file.

    Returns:
        tuple: Image data as numpy array, width, height, transform, projection, and nodata value.
    """
    gdal.AllRegister()
    img_data = gdal.Open(img_path, gdal.GA_ReadOnly)
    if img_data is None:
        raise IOError(f"Failed to open {img_path}")

    im_width = img_data.RasterXSize
    im_height = img_data.RasterYSize
    transform = img_data.GetGeoTransform()
    proj = img_data.GetProjection()

    band = img_data.GetRasterBand(1)
    im_data = band.ReadAsArray(0, 0, im_width, im_height)
    nodata_value = band.GetNoDataValue()

    return im_data, im_width, im_height, transform, proj, nodata_value


def read_tif_files(input_files):
    """
    Read multiple TIF files and return as a list of GDAL datasets.

    Args:
        input_files (list): List of file paths to TIF files.

    Returns:
        list: List of GDAL datasets.
    """
    return [gdal.Open(file) for file in input_files]


def split_and_distribute(base_image, input_images):
    """
    Split and distribute pixel values from base image to input images based on their ratios.

    Args:
        base_image (gdal.Dataset): Base image dataset.
        input_images (list): List of input image datasets.

    Returns:
        list: List of output numpy arrays.
    """
    base_array = base_image.ReadAsArray()
    input_arrays = [image.ReadAsArray() for image in input_images]

    height, width = base_array.shape

    for input_array in input_arrays:
        if input_array.shape != base_array.shape:
            raise ValueError("All input images must have the same dimensions as the base image.")

    ratios = [input_array / (np.sum(input_arrays, axis=0) + 1e-10) for input_array in input_arrays]
    output_arrays = [base_array * ratio for ratio in ratios]

    return output_arrays


def save_tif_files(output_arrays, base_image, output_files):
    """
    Save output arrays as TIF files using metadata from the base image.

    Args:
        output_arrays (list): List of numpy arrays to save.
        base_image (gdal.Dataset): Base image dataset to get metadata from.
        output_files (list): List of output file paths.
    """
    driver = gdal.GetDriverByName("GTiff")
    base_proj = base_image.GetProjection()
    base_geotransform = base_image.GetGeoTransform()

    for output_array, output_file in zip(output_arrays, output_files):
        rows, cols = output_array.shape
        output_dataset = driver.Create(output_file, cols, rows, 1, gdal.GDT_Float32)
        output_dataset.SetGeoTransform(base_geotransform)
        output_dataset.SetProjection(base_proj)
        output_dataset.GetRasterBand(1).WriteArray(output_array)
        output_dataset.FlushCache()


def process_crop_data(crop_harvest_files, crop_total_files, crop_type, data_year):
    """
    Process crop data for a specific crop type and year.

    Args:
        crop_harvest_files (list): List of crop harvest file paths.
        crop_total_files (list): List of crop total file paths.
        crop_type (str): Type of crop being processed.
        data_year (int): Year of the data being processed.

    Returns:
        numpy.ndarray: Processed crop data as a 2D numpy array.
    """
    crop_types = ["Maize", "Wheat", "Rice", "Soybean"]
    valid_country_codes, valid_country_names, _ = process_country_name.match_country_name(
        fao_datatype='harvest', crop_type=crop_type, data_year=data_year)

    country_label_path = r"D:\Dataset\产量分解\Support_Data_for_CPI\CountryCodeRaster_2015-resample.tif"
    country_label_data, im_width, im_height, transform, proj, _ = read_image_as_array(country_label_path)

    phenology_file = r"D:\Dataset\产量分解\Support_Data_for_CPI\Country_Phenology.xlsx"
    phenology_data = pd.read_excel(phenology_file, sheet_name=None)
    sheet_names = list(phenology_data.keys())

    crop_harvest_images = [read_image_as_array(file)[0] for file in crop_harvest_files]
    crop_total_images = [np.round(read_image_as_array(file)[0]).astype(np.int16) for file in crop_total_files]

    crop_index = crop_types.index(crop_type)
    new_calf_total_image = np.zeros((im_height, im_width))
    output_count = 0

    for country_code, country_name in zip(valid_country_codes, valid_country_names):
        country_name = country_name.replace('United Kingdom of Great Britain and Northern Ireland', 'United Kingdom')
        country_name = country_name.replace('T?rkiye', 'Turkey')

        print(f'\nProcessing country: {country_name}')

        if country_code == 9999 and data_year <= 2011:
            country_pixels = np.logical_or(country_label_data == 6, country_label_data == 74)
        else:
            country_pixels = country_label_data == country_code

        if country_name in sheet_names:
            country_phenology = phenology_data[country_name].values[:, 1:]
            output_count += 1
        else:
            country_phenology = np.ones((4, 4))

        for season in range(4):
            season_phenology = country_phenology[:, season]
            if season_phenology[crop_index] == 0:
                continue

            season_total = crop_total_images[season][country_pixels]
            crop_harvest = crop_harvest_images[crop_index][country_pixels]

            tmp_value_image = np.sum(
                [crop_harvest_images[i][country_pixels] for i, val in enumerate(season_phenology) if val != 0], axis=0)
            if tmp_value_image is not None and tmp_value_image.size > 0:
                season_total_redistributed = season_total * np.divide(
                    crop_harvest,
                    tmp_value_image,
                    out=np.zeros_like(crop_harvest),
                    where=tmp_value_image != 0
                )
            else:
                season_total_redistributed = np.zeros_like(season_total)

            new_calf_total_image[country_pixels] += season_total_redistributed

            print(f'Max value in pixels: {np.max(new_calf_total_image[country_pixels])}')
            print(f'Min value in pixels: {np.min(new_calf_total_image[country_pixels])}')

        new_calf_total_image[new_calf_total_image < 1e-8] = 0
        write_array_as_image(new_calf_total_image, im_width, im_height, transform, proj, 0.0, data_year, crop_type)
        print(f'Number of output subtables: {output_count}')

        return new_calf_total_image

def write_array_as_image(img_data, im_width, im_height, transform, proj, nodata_value, year, crop_type):
    """
    Write array data as a GeoTIFF image.

    Args:
        img_data (numpy.ndarray): Image data as a 2D numpy array.
        im_width (int): Image width in pixels.
        im_height (int): Image height in pixels.
        transform (tuple): Geotransform tuple.
        proj (str): Projection string.
        nodata_value (float): No data value.
        year (int): Year of the data.
        crop_type (str): Type of crop.
    """
    output_path = r"D:\Dataset\产量分解\Crop_Data\Calf_Total\Crop_Total_Redistribute_{}_{}.tif".format(year,
                                                                                                       crop_type)
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(output_path, im_width, im_height, 1, gdal.GDT_Float32)
    dataset.SetGeoTransform(transform)
    dataset.SetProjection(proj)
    dataset.GetRasterBand(1).WriteArray(img_data)
    # Uncomment the following line if you want to set a no data value
    # dataset.GetRasterBand(1).SetNoDataValue(nodata_value)
    dataset.FlushCache()

def process_crop_data_for_year(year):
    """
    Process crop data for a specific year.

    Args:
        year (int): Year to process data for.
    """
    crop_types = ["Maize", "Wheat", "Rice", "Soybean"]
    harvest_area_folder = r"D:\Dataset\产量分解\Crop_Data\Harvest_Area"
    calf_total_folder = r"D:\Dataset\产量分解\Indicators_resample\CALF_Crop_Total"

    # Use 2015 data for harvest area as it's the only year with real values
    crop_harvest_files = [os.path.join(harvest_area_folder, f"HarvArea_{crop}_2015.tif") for crop in crop_types]

    # Prepare CALF total files for the given year
    crop_total_file_names = [f'{year - 1}_ONDJ', f'{year}_JFMA', f'{year}_AMJJ', f'{year}_JASO']
    crop_total_files = [os.path.join(calf_total_folder, str(year), f"cropped_or_not_{name}.tif") for name in
                        crop_total_file_names]

    for crop_type in crop_types:
        process_crop_data(crop_harvest_files, crop_total_files, crop_type, year)

    print(f"Processing completed successfully for year {year}.")

if __name__ == "__main__":
    years_to_process = range(2022, 2023)
    for year in years_to_process:
        process_crop_data_for_year(year)