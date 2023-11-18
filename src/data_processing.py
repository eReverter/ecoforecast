import argparse
import os
import numpy as np
import pandas as pd
from definitions import (
    RAW_DATA_DIR, 
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    GREEN_ENERGY)

def data_checks(df):
    assert len(df['UnitName'].unique()) == 1, 'Multiple units in the DataFrame'

def load_raw_data(type, region):
    """
    Load raw data from a given type and region to a single DataFrame.
    """
    # Get all files in the data path
    files = os.listdir(RAW_DATA_DIR)

    # Load all files that comply with the type and region into a single DataFrame
    df = pd.DataFrame()
    for file in files:
        if file.startswith(f'{type}_{region}'):
            df = pd.concat([df, pd.read_csv(f'{RAW_DATA_DIR}/{file}')], ignore_index=True)
    
    # One timestamp is enough
    df['timestamp'] = df['StartTime'].str.replace('Z', '')
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%dT%H:%M%z')
    df.drop(columns=['StartTime', 'EndTime'], inplace=True)
    df.set_index('timestamp', inplace=True)
    df.reset_index(inplace=True)
    return df

def _estimate_timestamp_freq(df, timestamp_col='timestamp'):
    # Convert the time column to datetime
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    # Calculate the differences between consecutive timestamps
    time_diffs = df[timestamp_col].diff().dropna()

    # Find the most common difference
    estimated_freq = time_diffs.value_counts().idxmax()

    return estimated_freq

def filter_green_energy(df):
    """
    Filter the DataFrame to only include green energy sources.
    """
    return df[df['PsrType'].isin(GREEN_ENERGY)]

def fill_time_series_gaps(df, timestamp_col, groupby_cols, target_col):
    """
    Fills gaps in time series data for each series based on the specified frequency.

    :param df: DataFrame with time series data.
    :param freq: Frequency string (e.g., '1H' for hourly data).
    :param datetime_col: Name of the column containing time stamps.
    :param groupby_cols: List of column names to group by (series identifiers).
    :param target_col: Name of the column containing values.
    :return: DataFrame with gaps filled.
    """
    # Ensure datetimes are timezone-aware
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True)

    df.sort_values(by=groupby_cols + [timestamp_col], inplace=True)

    filled_series = []

    for group_keys, group in df.groupby(groupby_cols):
        full_range = pd.date_range(start=group[timestamp_col].min(), 
                                   end=group[timestamp_col].max(), 
                                   freq=_estimate_timestamp_freq(group, timestamp_col),
                                   tz='UTC')
        group.set_index(timestamp_col, inplace=True)
        group = group.reindex(full_range, method=None)
        
        # Ensure groupby_cols are backfilled with the appropriate values
        for col, value in zip(groupby_cols, group_keys):
            group[col] = value

        group[target_col].fillna(np.nan, inplace=True)
        group.reset_index(inplace=True)
        group.rename(columns={'index': timestamp_col}, inplace=True)
        filled_series.append(group)

    return pd.concat(filled_series)

def impute_missing_values(df, timestamp_col, groupby_cols):
    """
    Impute missing values in the DataFrame by taking the mean between the previous and next values.
    If the missing values are at the start or end of the DataFrame, they will be filled with the closest non-missing value.
    """
    # Sort the DataFrame by group by columns and datetime column
    df = df.sort_values(groupby_cols + [timestamp_col])

    # Impute missing values with linear interpolation
    df_interpolated = df.interpolate(method='linear', limit_direction='both')

    return df_interpolated

def aggregate_to_hourly(df, timestamp_col, groupby_cols, aggregate_cols):
    # Set the 'timestamp_col' as the index
    df.set_index(timestamp_col, inplace=True)

    # Group by specified columns and the hour, then aggregate
    df_grouped = df.groupby([pd.Grouper(freq='H')] + groupby_cols)
    aggregated_df = df_grouped.agg({col: 'sum' for col in aggregate_cols}).reset_index()

    return aggregated_df

def load_data(file_path):
    # TODO: Load data from CSV file

    return df

def clean_data(df):
    # TODO: Handle missing values, outliers, etc.

    return df_clean

def preprocess_data(df):
    # TODO: Generate new features, transform existing features, resampling, etc.

    return df_processed

def save_data(df, output_file):
    # TODO: Save processed data to a CSV file
    pass

def parse_arguments():
    parser = argparse.ArgumentParser(description='Data processing script for Energy Forecasting Hackathon')
    parser.add_argument(
        '--input_file',
        type=str,
        default='data/raw_data.csv',
        help='Path to the raw data file to process'
    )
    parser.add_argument(
        '--output_file', 
        type=str, 
        default='data/processed_data.csv', 
        help='Path to save the processed data'
    )
    return parser.parse_args()

def main(input_file, output_file):
    df = load_data(input_file)
    df_clean = clean_data(df)
    df_processed = preprocess_data(df_clean)
    save_data(df_processed, output_file)

if __name__ == "__main__":
    args = parse_arguments()
    main(args.input_file, args.output_file)