import argparse
import os
import numpy as np
import pandas as pd
from src.definitions import (
    RAW_DATA_DIR, 
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    GREEN_ENERGY)
from src.config import setup_logger

### RAW DATA PROCESSING -> INTERIM DATA ###

def data_checks(df):
    assert len(df['UnitName'].unique()) == 1, 'Multiple units in the DataFrame'

def load_raw_data(type, region):
    """
    Load raw data from a given type and region to a single DataFrame.

    :param type: Type of data to load (e.g., 'load', 'gen').
    :param region: Region to load data from (e.g., 'HU', 'SP').
    :return: DataFrame with the loaded data and a timestamp column.
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
    """
    Estimate the frequency of the timestamps in the DataFrame.
    
    :param df: DataFrame with timestamps.
    :param timestamp_col: Name of the column containing timestamps.
    :return: Estimated frequency of the timestamps.
    """
    # Convert the time column to datetime
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    # Calculate the differences between consecutive timestamps
    time_diffs = df[timestamp_col].diff().dropna()

    # Find the most common difference
    estimated_freq = time_diffs.value_counts().idxmax()

    return estimated_freq

def filter_green_energy(df):
    """
    Filter the DataFrame to only include green energy sources defined in the GREEN_ENERGY list.

    :param df: DataFrame with the raw data.
    :return: DataFrame with only green energy sources.
    """
    return df[df['PsrType'].isin(GREEN_ENERGY)]

def fill_time_series_gaps(df, timestamp_col, groupby_cols, target_col):
    """
    Fills gaps in time series data for each series based on the specified frequency.

    :param df: DataFrame with time series data.
    :param timestamp_col: Name of the column containing time stamps.
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

    :param df: DataFrame with missing values.
    :param timestamp_col: Name of the column containing time stamps.
    :param groupby_cols: List of column names to group by (series identifiers).
    :return: DataFrame with missing values imputed.
    """
    # Sort the DataFrame by group by columns and datetime column
    df = df.sort_values(groupby_cols + [timestamp_col])

    # Impute missing values with linear interpolation
    df_interpolated = df.interpolate(method='linear', limit_direction='both')

    return df_interpolated

def aggregate_to_hourly(df, timestamp_col, groupby_cols, aggregate_cols):
    """
    Aggregate the DataFrame to hourly values.

    :param df: DataFrame with the raw data.
    :param timestamp_col: Name of the column containing time stamps.
    :param groupby_cols: List of column names to group by (series identifiers).
    :param aggregate_cols: List of column names to aggregate.
    :return: DataFrame with hourly values.
    """
    # Set the 'timestamp_col' as the index
    df.set_index(timestamp_col, inplace=True)

    # Group by specified columns and the hour, then aggregate
    df_grouped = df.groupby([pd.Grouper(freq='H')] + groupby_cols)
    aggregated_df = df_grouped.agg({col: 'sum' for col in aggregate_cols}).reset_index()

    return aggregated_df


def process_raw_data():
    """
    
    """
    return

### INTERIM DATA PROCESSING -> PROCESSED DATA ###

def aggregate_to_green_energy(df, filter_out=[]):
    """
    Aggregate each hourly timestamp to a single row, with all the different PsrTypes aggregated into one value.

    :param df: DataFrame with the interim data.
    :param filter_out: List of PsrTypes to filter out.
    :return: DataFrame with aggregated values.
    """
    # Filter out the specified PsrTypes
    df = df[~df['PsrType'].isin(filter_out)]

    # Group by timestamp and aggregate
    df_grouped = df.groupby(['timestamp'])
    df_aggregated = df_grouped.agg({'ActualGenerationOutput': 'sum'}).reset_index()

    return df_aggregated
    
def _pivot_df(df, index_col, columns_col, values_col):
    """
    Pivot the DataFrame to have the values in the 'values_col' as columns, with the 'columns_col' as the column names.

    :param df: DataFrame to pivot.
    :param index_col: Name of the column to use as index.
    :param columns_col: Name of the column to use as columns.
    """
    return df.pivot(index=index_col, columns=columns_col, values=values_col)

def merge_interim_data():
    """
    Load and merge datasets from the interim data folder.

    :return: DataFrame with the merged data.
    """
    # Get all files in the data path
    files = os.listdir(INTERIM_DATA_DIR)

    # Load all files that comply with the type and region into a single DataFrame
    df = pd.DataFrame()
    for file in files:

        # Get region and type to be added as prefix to the columns
        region, type = file.split('_')[1:3]

        # Load the file
        df_file = pd.read_csv(f'{INTERIM_DATA_DIR}/{file}')

        # Pivot the DataFrame
        df_file = _pivot_df(df_file, 'timestamp', 'PsrType', 'ActualGenerationOutput')

        # Add prefix to the columns
        df_file.columns = [f'{region}_{type}_{col}' for col in df_file.columns]

        # Horizontally merge the DataFrames
        df = pd.concat([df, df_file], axis=1)    

    return df

def process_interim_data():
    """
    
    """
    return

### PIPELINE ###

def parser_add_arguments(parser):
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

def main():
    """
    Process raw and interim data and save it to the processed data folder.
    """
    parser = argparse.ArgumentParser(description='Data processing script for Energy Forecasting Hackathon')
    parser_add_arguments(parser)
    args = parser.parse_args()

if __name__ == "__main__":
    main()