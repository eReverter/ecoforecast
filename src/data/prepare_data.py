"""
Script to prepare data for modelling either for forecasting or classification.

For classification, the label of the next hour maximum surplus region is added. Then, the idea is to predict that label based on the previous hour only.

For forecasting, the data is converted to a time series format with the following features:
- series_id
- timestamp
- surplus*

*Another option would be to train different models for every region, or two models, one for generated data and another one for loaded data.
Since there is not much data, I think it is better to train a single model for all regions and both types of data.

In the classification model, only complete data will be used. That is, The 3 months of data where all values of UK are available.
In the forecasting model, all data from 2022 wil be used.

Processed data will always look like:
timestamp | etype_1_region_1 | ... | etype_2_region_m 
"""
from src.definitions import PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR
import pandas as pd
import os

### GENERAL FUNCTIONS ###

def load_data():
    train = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'train.csv'), parse_dates=['timestamp'])
    test = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'test.csv'), parse_dates=['timestamp'])
    return train, test

def load_holidays():
    for filename in os.listdir(EXTERNAL_DATA_DIR):
        if 'holiday' in filename:
            pass

def get_surplus():
    pass

def get_labels(df):
    """
    Add target labels to the DataFrame. That is, the country index of the next hour maximum surplus.

    :param df: DataFrame with the interim data.
    :return: DataFrame with labels added.
    """
    for region in REGION:
        load_col = f"Load_{region}"
        gen_col = f"green_energy_{region}"
        df[f'{region}_surplus'] = df[gen_col] - df[load_col]

    # Get the country index of the next hour maximum surplus
    surplus_cols = [col for col in df.columns if col.endswith('_surplus')]
    
    # TODO: Finish this function

def split_load_gen():
    pass



### FORECASTING ENERGY FUNCTIONS ###

def get_lags():
    pass

### CLASSIFICATION FUNCTIONS ###


