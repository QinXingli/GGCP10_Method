#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Batch production script for CropWatch Group.

This script processes crop production data for multiple years and crop types.
It uses regression models and performs data revisions.

"""

import logging
from typing import List, Optional

# Import custom modules

import data_driven_modeling
import data_consistency

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
CROP_TYPES = ['Maize', 'Wheat', 'Rice', 'Soybean']
PROCESS_TYPES = ['train', 'test']
YEAR_RANGE = [i for i in range(2010, 2021)]


def process_crop_data(crop_type: str, data_year: int,  special_mark: Optional[str] = None,
                      process_type: str = 'test'):
    """
    Process crop production data for a specific crop type and year.

    Args:
        crop_type (str): Type of crop to process.
        data_year (int): Year of data to process.
        special_mark (str, optional): Special marker for the process. Defaults to None.
        process_type (str): Type of processing ('train' or 'test'). Defaults to 'test'.

    Returns:
        None
    """
    logging.info(f"Processing {crop_type} data for year {data_year}")

    data_driven_modeling.do_production_model_train_test(
        crop_type=crop_type,
        data_year=data_year,
        special_mark=special_mark,
        process_type=process_type
    )

    data_consistency.do_data_consistency(
        crop_type=crop_type,
        data_year=data_year,
        if_revised=False
    )

    logging.info(f"Finished processing {crop_type} data for year {data_year}")


def main():
    # Configuration

    special_mark = None
    process_type = PROCESS_TYPES[1]  # 'test'

    selected_years = YEAR_RANGE
    selected_crops = CROP_TYPES

    logging.info(f"Selected crop types: {selected_crops}")
    logging.info(f"Selected years: {selected_years}")

    # Process data for each year and crop type
    for year in selected_years:
        for crop in selected_crops:
            process_crop_data(crop, year, special_mark, process_type)


if __name__ == "__main__":
    main()