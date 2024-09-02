#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   read_AEZs_data.py    
@Contact :   qinxl@aircas.ac.cn
@License :   (C)Copyright 2021-2022, CropWatch Group
@Modify Time : 2022/4/20 16:18
@Author : Qin Xingli
@Version : 1.0
@Description : 用于读取农业生态区的数据
"""
from osgeo import gdal
import numpy as np
import csv


def read_image_as_array(img_path):
    """
    read input image data
    """
    gdal.AllRegister()
    img_data = gdal.Open(img_path, gdal.GA_ReadOnly)
    if img_data is None:
        print('{}打开失败！'.format(img_path))
        exit(1)
    im_width = img_data.RasterXSize  # 读取图像行数
    im_height = img_data.RasterYSize  # 读取图像列数
    im_bands = img_data.RasterCount
    transform = img_data.GetGeoTransform()
    proj = img_data.GetProjection()

    ith_band = img_data.GetRasterBand(1)
    im_data = ith_band.ReadAsArray(0, 0, im_width, im_height)

    return im_data, im_width, im_height, transform, proj


def read_AEZs_image():

    AEZs_image_path = r"D:\Dataset\产量分解\Support_Data_for_CPI\AEZs_223_new_ShpToRaster-Resample.tif"
    AEZs_data, im_width, im_height, transform, proj = read_image_as_array(AEZs_image_path)

    return AEZs_data


def read_csv(file_path):
    with open(file_path, 'r', encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
    return rows


def get_AEZs_info():

    AEZs_table_path = r"D:\Dataset\产量分解\Support_Data_for_CPI\AEZs_223_new.csv"
    AEZs_table = read_csv(AEZs_table_path)

    # 首先确定有的国家的名称
    country_names = []
    for i in range(1, len(AEZs_table)):
        ith_country_name = AEZs_table[i][4]
        if ith_country_name not in country_names:
            country_names.append(ith_country_name)

    country_names_and_AEZs_dict = {}
    for i in range(len(country_names)):
        ith_country_name = country_names[i]
        country_names_and_AEZs_dict[ith_country_name] = []
        if ith_country_name == 'U.K. of Great Britain and Northern Ireland':
            print('a')
        for j in range(1, len(AEZs_table)):
            jth_country_name = AEZs_table[j][4]
            jth_AEZ_code = int(AEZs_table[j][1])
            if ith_country_name == jth_country_name:
                country_names_and_AEZs_dict[ith_country_name].append(jth_AEZ_code)

    return country_names_and_AEZs_dict


def get_AEZs_names_dict():
    AEZs_names_dict = {}
    AEZs_data = []
    AEZs_file_path = r"D:\Dataset\产量分解\Support_Data_for_CPI\AEZs_223_new.csv"
    with open(AEZs_file_path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == 0:
                continue
            AEZs_data.append(row)

    for j in AEZs_data:
        jth_AEZ_code = int(j[1])
        jth_AEZ_name = j[2]
        AEZs_names_dict[jth_AEZ_code] = jth_AEZ_name

    return AEZs_names_dict


# 2023-01-12 ADM0_Code和AEZs_code的对应关系
def get_AEZs_ADM0_Code_dict():
    AEZs_names_dict = {}
    AEZs_data = []
    AEZs_file_path = r"D:\Dataset\产量分解\Support_Data_for_CPI\AEZs_223_new.csv"
    with open(AEZs_file_path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == 0:
                continue
            AEZs_data.append(row)

    for j in AEZs_data:
        jth_AEZ_code = int(j[1])
        jth_AEZ_ADM0_Code = j[3]
        AEZs_names_dict[jth_AEZ_code] = jth_AEZ_ADM0_Code

    return AEZs_names_dict