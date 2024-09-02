"""
Reset harvest area data based on FAO statistics.
"""

from osgeo import gdal
import numpy as np
import process_country_name
import os

os.environ['PROJ_LIB'] = r'C:\ProgramData\Anaconda3\envs\MyProductionIndex\Lib\site-packages\osgeo\data\proj'


def read_image_as_array(img_path):
    """
    Read input image data as numpy array.

    Args:
        img_path (str): Path to the image file.

    Returns:
        tuple: A tuple containing:
            - im_data (numpy.ndarray): Image data as a 2D numpy array.
            - im_width (int): Image width.
            - im_height (int): Image height.
            - transform (tuple): Geotransform parameters.  
            - proj (str): Projection information.
            - nodata_value (float): NoData value of the image.
    """
    gdal.AllRegister()
    img_data = gdal.Open(img_path, gdal.GA_ReadOnly)
    if img_data is None:
        print(f'Failed to open {img_path}!')
        exit(1)

    im_width = img_data.RasterXSize
    im_height = img_data.RasterYSize
    transform = img_data.GetGeoTransform()
    proj = img_data.GetProjection()

    band = img_data.GetRasterBand(1)
    im_data = band.ReadAsArray(0, 0, im_width, im_height)
    nodata_value = band.GetNoDataValue()

    return im_data, im_width, im_height, transform, proj, nodata_value


def get_gaez_data_by_country_code(gaez_data, country_label, country_code, nodata_value):
    """
    Get GAEZ harvest area data for a specific country.

    Args:
        gaez_data (numpy.ndarray): GAEZ harvest area data.
        country_label (numpy.ndarray): Country code raster data.
        country_code (int): Country code.
        nodata_value (float): NoData value.

    Returns:
        numpy.ndarray: GAEZ harvest area data for the given country.
    """
    code_index = np.where(country_label == country_code)
    code_gaez_data = gaez_data[code_index]
    code_gaez_data = code_gaez_data[code_gaez_data != nodata_value]
    return code_gaez_data


def calculate_country_area(crop_type='Rice', data_year=2020):
    """
    Calculate harvest area for each country based on GAEZ data and FAO statistics.

    Args:
        crop_type (str): Crop type. Default is 'Rice'.
        data_year (int): Year of FAO data. Default is 2020.

    Returns:
        str: Path to the output harvest area file.
    """
    print(f'\nGenerating harvest area data for {crop_type} in {data_year}...')

    country_code_file = 'D:\\Dataset\\产量分解\\Support_Data_for_CPI\\CountryCodeRaster_2015-resample.tif'
    gaez_harvest_file = os.path.join('G:\\Dataset\\产量分解\\Crop_Data\\Harvest_Area',
                                     f'HarvArea_{crop_type}_2015_mask.tif')
    output_dir = 'D:/Dataset/产量分解/Crop_Data/Harvest_Area/'

    # Get valid country codes and corresponding FAO data
    country_codes, _, fao_data = process_country_name.match_country_name(
        fao_datatype='harvest', crop_type=crop_type, data_year=data_year)

    country_label, width, height, _, _, _ = read_image_as_array(country_code_file)
    gaez_data, _, _, transform, proj, nodata_value = read_image_as_array(gaez_harvest_file)
    if nodata_value is None:
        nodata_value = 0

    result_data = np.zeros([height, width], dtype=np.float32)

    # Calculate harvest area for each country
    for idx, code in enumerate(country_codes):
        gaez_data_2015 = get_gaez_data_by_country_code(gaez_data, country_label, code, nodata_value)
        sum_2015 = np.sum(gaez_data_2015)  # unit: ha
        sum_2020 = float(fao_data[idx]) / 1000.0  # convert unit from 1000 ha to ha

        code_index = np.where(country_label == code)
        if sum_2015 <= 0:
            result_data[code_index] = 0
        else:
            result_data[code_index] = (gaez_data[code_index] / sum_2015) * sum_2020

    # Set nodata and zero values  
    result_data[gaez_data < 0] = nodata_value
    result_data[gaez_data == 0] = 0
    result_data[gaez_data == nodata_value] = 0

    # Save result to file
    output_file = os.path.join(output_dir, f'HarvArea_{crop_type}_{data_year}.tif')
    driver = gdal.GetDriverByName("GTiff")
    output = driver.Create(output_file, width, height, 1, gdal.GDT_Float32)
    if output is None:
        print('Creating output file failed.')
        exit(1)

    output.SetGeoTransform(transform)
    output.SetProjection(proj)
    if nodata_value != 0:
        output.GetRasterBand(1).SetNoDataValue(nodata_value)
    output.GetRasterBand(1).WriteArray(result_data)
    del output

    return output_file


def main():
    area_file = calculate_country_area(crop_type='Soybean', data_year=2015)
    print(f'\nGenerated harvest area file: {area_file}')


if __name__ == '__main__':
    main()