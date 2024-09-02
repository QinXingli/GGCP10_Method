#!/usr/bin/env python

from difflib import SequenceMatcher
import csv
import numpy as np


def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


def read_csv(file_path):
    with open(file_path, 'r', encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
    return rows


def read_xlxs(file_path, fao_datatype, data_year):

    import xlrd

    book = xlrd.open_workbook(file_path)
    # print('sheet:', book.sheet_names())

    if fao_datatype == 'harvest':
        xlxs_data_type = 'Area'
    else:
        xlxs_data_type = 'Production'
    required_book_name = '{}{}'.format(xlxs_data_type, data_year)
    required_book_index = book.sheet_names().index(required_book_name)

    sheet = book.sheet_by_index(required_book_index)

    n_rows = sheet.nrows
    n_cols = sheet.ncols
    all_rows = []
    for i in range(0, n_rows):
        ith_row = sheet.row_values(i)
        # remove the first column
        ith_row__ = [ith_row[k] for k in range(1, len(ith_row))]
        all_rows.append(ith_row__)

    return all_rows


def match_country_name(fao_datatype='harvest', crop_type='rice', if_revised=False, data_year=2020):

    file_path_country_code = r"D:\Dataset\产量分解\Support_Data_for_CPI\CountryCode_2015.csv"
    if if_revised:
        file_path_fao = "D:/Dataset/产量分解/国家尺度作物信息表/FAOSTAT_global_{}_{}_{}_revised.csv".format(fao_datatype, crop_type, data_year)
    else:
        file_path_fao = "D:/Dataset/产量分解/国家尺度作物信息表/FAOSTAT_global_{}_{}_{}.csv".format(fao_datatype, crop_type, data_year)
    data_fao = read_csv(file_path_fao)
    #############################################################

    data_country_code = read_csv(file_path_country_code)

    # 读取匹配表
    matched_country_code_path = r"G:\Dataset\产量分解\Support_Data_for_CPI\GAUL_Code_and_FAO_Code.xlsx"

    import pandas as pd

    def read_excel_file(path):
        df = pd.read_excel(path)
        # convert the data to a list
        data = df.values.tolist()
        return data
    matched_data = read_excel_file(matched_country_code_path)

    valid_country_code_list = []
    valid_country_name_list = []
    valid_country_value_list = []
    for i in range(1, len(data_country_code)):
        ith_country_name = data_country_code[i][1]
        ith_country_code = int(data_country_code[i][0])

        # 获取匹配表中的FAO国家代码
        match_fao_code = None
        for j in range(0, len(matched_data)):
            GAUL_country_code = int(matched_data[j][0])
            if GAUL_country_code != ith_country_code:
                continue
            else:
                match_fao_code = int(matched_data[j][2])
                break

        # 获取FAO国家代码对应的国家名称和数值
        for j in range(1, len(data_fao)):
            jth_value = data_fao[j][4]
            jth_country_name = data_fao[j][1]
            jth_country_code = int(data_fao[j][0])
            if jth_country_code == match_fao_code:

                valid_country_code_list.append(ith_country_code)
                valid_country_name_list.append(ith_country_name)
                valid_country_value_list.append(jth_value)
                break

    return valid_country_code_list, valid_country_name_list, valid_country_value_list
