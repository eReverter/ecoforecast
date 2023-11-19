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
import argparse
from src.definitions import (
    PROCESSED_DATA_DIR, 
    EXTERNAL_DATA_DIR,
    REGION,
    REGION_MAPPING,
    PREDICTIONS_DIR)
import pandas as pd
import os

### GENERAL FUNCTIONS ###

def load_data():
    train = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'train.csv'), parse_dates=['timestamp'])
    test = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'validation.csv'), parse_dates=['timestamp'])
    return train, test

def add_is_weekend(df):
    df['is_weekend'] = df['timestamp'].dt.dayofweek.isin([5, 6])
    return df

def get_surplus(df):
    for region in REGION:
        df[f'{region}_surplus'] = df[f'{region}_gen'] - df[f'{region}_load']
    return df

def get_curr_max(df):
    """
    Get the current maximum surplus region
    """
    max_col = df.filter(regex='surplus').idxmax(axis=1)
    country_code = max_col.str.split('_', expand=True)[0]
    df['curr_max'] = country_code

    return df

def get_cls_target(df):
    """
    Get the label for the dataframe using the column with the maximum surplus
    """
    # Use the next observation country code as the target
    df['target'] = df['curr_max'].shift(-1)
    return df

def get_ohe_from_cat(df, cat='curr_max'):
    """
    Get one-hot encoding from a categorical column
    """
    df = pd.concat([df, pd.get_dummies(df[cat], prefix=cat)], axis=1)
    return df

def split_load_gen():
    pass

### FORECASTING ENERGY FUNCTIONS ###

def convert_to_timeseries(df, value_columns=[], metadata_columns=[]):
    """
    Convert the DataFrame to a time series format with the following features:

    - series_id (country code extracted from surplus columns)
    - timestamp
    - surplus
    - Additional metadata (e.g., is_weekend)

    :param df: DataFrame to be converted.
    :param surplus_columns: List of columns in df that contain surplus data.
    :param metadata_columns: List of additional metadata columns to retain (excluding 'timestamp').
    """
    # Ensure 'timestamp' is always included and not duplicated
    id_vars = ['timestamp'] + metadata_columns

    if len(value_columns) == 0:
        value_columns = [col for col in df.columns if 'surplus' in col]

    # Selecting surplus variables and metadata
    df_selected = df[id_vars + value_columns]

    # Melt the dataframe to convert to time series format
    df_melted = df_selected.melt(id_vars=id_vars, var_name='series_id', value_name='surplus')

    # Extracting the country code from the series_id
    df_melted['series_id'] = df_melted['series_id'].str.split('_').str[0]

    return df_melted

def get_lags(ts, lags, value_col='surplus', series_id_col='series_id'):
    """
    Adds lagged values to the time series data.

    :param df: DataFrame with time series data.
    :param lags: List of lag values.
    :param value_col: Name of the column containing values.
    :param series_id_col: Name of the column containing series identifiers.
    :return: DataFrame with lagged values added.
    """
    ts.sort_values(by=[series_id_col, 'timestamp'], inplace=True)
    for lag in lags:
        ts[f'{value_col}_lag_{lag}'] = ts.groupby(series_id_col)[value_col].shift(lag)
    return ts

def get_forecast_target(ts, prediction_horizon, value_col='surplus', series_id_col='series_id'):
    ts['target'] = ts.groupby(series_id_col)[value_col].shift(-prediction_horizon)
    return ts

def add_is_holiday(ts):
    # Convert timestamp to date for comparison
    ts['date'] = ts['timestamp'].dt.date

    # Initialize the is_holiday column to False
    ts['is_holiday'] = False

    # Iterate through each file in the directory
    for filename in os.listdir(EXTERNAL_DATA_DIR):
        if 'holiday' in filename:
            # Extract country code from filename
            country_code = filename.split('_')[0]
            mapping = {v: k for k, v in REGION_MAPPING.items()}
            mapped_country_code = mapping[country_code]

            # Load holidays for the country
            holidays = pd.read_csv(os.path.join(EXTERNAL_DATA_DIR, filename), parse_dates=['Date'])
            holidays['date'] = holidays['Date'].dt.date

            # Find indices in the time series where the country and date match a holiday
            holiday_indices = ts[(ts['series_id'] == mapped_country_code) & (ts['date'].isin(holidays['date']))].index

            # Mark these as holidays
            ts.loc[holiday_indices, 'is_holiday'] = True

    ts.drop('date', axis=1, inplace=True)
    return ts
            
def normalize_values(ts, value_col='value', series_id_col='series_id'):
    """
    Normalizes values in time series data using the min-max scaler.

    :param df: DataFrame with time series data.
    :param value_col: Name of the column containing values.
    :param series_id_col: Name of the column containing series identifiers.
    :return: DataFrame with values normalized.
    """
    ts[value_col] = ts.groupby(series_id_col)[value_col].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    return ts

### CLASSIFICATION FUNCTIONS ###

### MAIN ###
def prepare_reference_predictions(validation_file):
    """
    Prepare the reference predictions for the validation set.
    """
    # Load validation set
    validation = pd.read_csv(validation_file, parse_dates=['timestamp'])

    # Add surplus column
    validation = get_surplus(validation)

    # Add target column
    validation = get_curr_max(validation)
    validation = get_cls_target(validation)

    # Build predictions.json with the target column
    predictions = validation[['timestamp', 'target']]
    predictions.to_json(os.path.join(PREDICTIONS_DIR, 'predictions.json'), orient='records')

def main():
    parser = argparse.ArgumentParser(description='Prepare predictions fo reference')
    parser.add_argument('--validation', type=str, help='Path to validation file')
    args = parser.parse_args()

    prepare_reference_predictions(args.validation)

if __name__ == '__main__':
    main()
