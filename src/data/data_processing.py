"""
Script to process raw data into interim and processed data.
"""
import argparse
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.definitions import (
    RAW_DATA_DIR, 
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    GREEN_ENERGY,
    TYPE,
    REGION)

from src.config import setup_logger
from src.metrics import StatisticsETL

logger = setup_logger()
statistics = StatisticsETL()

### RAW DATA PROCESSING -> INTERIM DATA ###

def load_raw_data(etype, region, mode='train'):
    """
    Load raw data from a given etype and region to a single DataFrame.

    :param etype: Type of data to load (e.g., 'load', 'gen').
    :param region: Region to load data from (e.g., 'HU', 'SP').
    :return: DataFrame with the loaded data and a timestamp column.
    """
    # Get all files in the data path
    files = os.listdir(f'{RAW_DATA_DIR}/{mode}/')

    # Load all files that comply with the etype and region into a single DataFrame
    df = pd.DataFrame()
    for file in files:
        if file.startswith(f'{etype}_{region}'):
            data = pd.read_csv(f'{RAW_DATA_DIR}/{mode}/{file}')
            logger.info(f'Loading {file}, shape: {data.shape}')
            df = pd.concat([df, data], ignore_index=True)
    
    # One timestamp is enough
    df['timestamp'] = df['StartTime'].str.replace('Z', '')
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%dT%H:%M%z')
    df.drop(columns=['StartTime', 'EndTime', 'AreaID'], inplace=True)
    df.set_index('timestamp', inplace=True)
    df.reset_index(inplace=True)
    return df

def _update_columns(df, etype):
    """
    Simplify the column names of the DataFrame.

    :param df: DataFrame with the raw data.
    :param etype: Type of data to load (e.g., 'load', 'gen').
    :return: DataFrame with simplified column names.
    """
    df.drop(columns=['UnitName'], inplace=True)
    if etype == 'gen':
        columns = ['timestamp', 'energy_type', 'value']
    elif etype == 'load':
        columns = ['timestamp', 'value']
    else:
        raise ValueError(f'Unknown etype: {etype}')
    df.columns = columns
    return df

def _estimate_timestamp_freq(df, timestamp_col='timestamp'):
    """
    Estimate the frequency of the timestamps in the DataFrame.
    
    :param df: DataFrame with timestamps.
    :param timestamp_col: Name of the column containing timestamps.
    :return: Estimated frequency of the timestamps in timedelta format.
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
    return df[df['energy_type'].isin(GREEN_ENERGY)]

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
    aggregated_df = df_grouped.agg({col: 'sum' for col in aggregate_cols}, skipna=True).reset_index()

    return aggregated_df

def resample_hourly_accounting_for_missing_intervals(df, groupby_cols=None):
    """
    Resamples data hourly such that if the frequency is lower than 1 hour, 
    the sum is divided by the number of intervals per hour. Handles optional grouping.

    :param df: DataFrame with time series data.
    :param groupby_cols: Optional list of column names to group by before resampling.
    :return: DataFrame with hourly values, resampled within each group if specified.
    """
    # Handle the case with no grouping
    if not groupby_cols:
        return _resample_group(df)
    
    # Container for the resampled data
    resampled_list = []

    # Iterate over each group if groupby columns are provided
    for name, group in df.groupby(groupby_cols):
        resampled_group = _resample_group(group.drop(groupby_cols, axis=1))
        
        # Add the group identifier
        if isinstance(name, tuple):
            for col, value in zip(groupby_cols, name):
                resampled_group[col] = value
        else:
            resampled_group[groupby_cols[0]] = name

        # Append the result to the list
        resampled_list.append(resampled_group)

    # Concatenate all the resampled data
    return pd.concat(resampled_list).reset_index().dropna()

def _resample_group(group):
    """ Helper function to resample a given group """
    # Estimate the frequency (15T or 30T)
    estimated_freq = _estimate_timestamp_freq(group)

    # Set the timestamp as index
    group['timestamp'] = pd.to_datetime(group['timestamp'])
    group.set_index('timestamp', inplace=True)
    
    # Initial resample to account for possible duplicated timestamps
    group = group.resample(estimated_freq).sum()

    # Resample to hourly, getting the sum and the count
    hourly_sum = group.resample('H').sum()
    hourly_count = group.resample('H').count()
    
    # Determine the number of intervals per hour (4 for 15T, 2 for 30T)
    intervals_per_hour = pd.Timedelta('1H') // estimated_freq

    # Adjust the hourly sum based on the count
    hourly_sum['value'] = hourly_sum['value'] * intervals_per_hour / hourly_count['value']
    
    return hourly_sum

def interpolate_zeros(df, column_name):
    """
    Interpolate zeros in the DataFrame.

    :param df: DataFrame with the raw data.
    :param column_name: Name of the column containing the values.
    :return: DataFrame with zeros interpolated.
    """
    # Step 1: Create a mask of original NaN values
    original_na_mask = df[column_name].isna()

    # Step 2: Replace 0s with NaNs temporarily
    df[column_name].replace(0, np.nan, inplace=True)

    # Step 3: Perform interpolation on NaNs (which includes the original 0s)
    df[column_name].interpolate(method='linear', direction='both', inplace=True)

    # Step 4: Restore original NaN values using the mask
    df.loc[original_na_mask, column_name] = np.nan

    return df

def process_raw_data(args):
    """
    Load, process, and save raw data from the raw data folder to the interim data folder.

    :param args: Arguments from the command line.
    """
    tqdm_bar = tqdm(total=len(REGION) * len(TYPE), desc='Processing raw data')
    for region in REGION:
        for etype in TYPE:
            # Update progress bar and its description to convey region and etype of energy
            tqdm_bar.update(1)
            tqdm_bar.set_description(f'Processing {etype} data from {region}...')

            # Constants
            groupby_cols = ['energy_type'] if etype == 'gen' else []
            name = f'{etype}_{region}'
            
            # Load raw data
            df = load_raw_data(etype, region, args.mode)

            # Assert units are the same
            assert len(df['UnitName'].unique()) == 1, 'Multiple units in the DataFrame'

            df = _update_columns(df, etype)
            statistics.update_original_count(name, df)

            # Filter out non-green energy sources
            if etype == 'gen':
                df = filter_green_energy(df)
            statistics.update_green_energy_count(name, df)

            # Fill gaps in time series
            statistics.update_estimated_frequency(name, _estimate_timestamp_freq(df, 'timestamp'))
            # statistics.update_estimated_frequency(name, 0)
            if args.fill_time_series_gaps:
                df = fill_time_series_gaps(df, 'timestamp', groupby_cols, 'value')

            # Impute missing values or interpolate zeros and keep NaNs
            na_count = df['value'].isna().sum()
            statistics.update_na_count(name, na_count)
            imp_count = 0
            if args.impute_missing_values:
                df = impute_missing_values(df, 'timestamp', groupby_cols)
                imp_count = df['value'].isna().sum()
            else:
                df = interpolate_zeros(df, 'value')
                imp_count = df['value'].isna().sum()
            statistics.update_imputed_count(name, imp_count)

            # Aggregate to hourly values
            if args.account_for_missing_intervals:
                df = resample_hourly_accounting_for_missing_intervals(df, groupby_cols)
            else:
                df = aggregate_to_hourly(df, 'timestamp', groupby_cols, ['value'])
            statistics.update_aggregated_count(name, df)

            # Save the DataFrame
            df.to_csv(f'{INTERIM_DATA_DIR}/{args.mode}/{region}_{etype}.csv', index=False)
            
    statistics.display_statistics()
    return statistics

### INTERIM DATA PROCESSING -> PROCESSED DATA ###

def aggregate_to_green_energy(df):
    """
    Aggregate each hourly timestamp to a single row, with all the different energy_type aggregated into one value.

    :param df: DataFrame with the interim data.
    :param filter_out: List of energy_type to filter out.
    :return: DataFrame with aggregated values.
    """

    # Group by timestamp and aggregate
    df_grouped = df.groupby([pd.Grouper(freq='H')])
    df_aggregated = df_grouped.agg({'value': 'sum'}, skipna=True).reset_index()

    return df_aggregated

def process_interim_data(args):
    """
    Load, merge, and save datasets from the interim data folder.
    """
    # Get all files in the data path
    files = os.listdir(f'{INTERIM_DATA_DIR}/{args.mode}')

    # Load all files that comply with the etype and region into a single DataFrame
    df = pd.DataFrame()

    tqdm_bar = tqdm(total=len(files), desc='Processing interim data')
    for file in files:
        # Ignore non-csv files (.gitkeep)
        if not file.endswith('.csv'):
            continue

        # Update bar
        tqdm_bar.update(1)
        tqdm_bar.set_description(f'Processing {file}...')

        # Get region and etype to be added as prefix to the columns
        region, etype = file.split('.')[0].split('_')

        # Load the file
        df_file = pd.read_csv(f'{INTERIM_DATA_DIR}/{args.mode}/{file}', parse_dates=['timestamp'])

        # Aggregate to green energy
        if etype == 'gen':
            df_file = aggregate_to_green_energy(df_file.set_index('timestamp'))
        
        # Log
        logger.info(f'Loaded {file}, shape: {df_file.shape}')

        # Add prefix to the columns
        df_file.columns = [f'{region}_{etype}' if col != 'timestamp' else col for col in df_file.columns]

        # Merge DataFrames by timestamp column (outer join), firsst iteration has no data to merge
        if df.empty:
            df = df_file
        else:
            df = pd.merge(df, df_file, how='outer', on='timestamp')

    # Save the DataFrame
    logger.info(f'Saving {args.mode} data, shape: {df.shape}')
    df.to_csv(f'{PROCESSED_DATA_DIR}/{args.mode}.csv', index=False)
    return

### MAIN ###

def parser_add_arguments(parser):
    parser.add_argument('--process_raw_data', action='store_true', help='Process raw data')
    parser.add_argument('--process_interim_data', action='store_true', help='Process interim data')
    parser.add_argument('--mode', default='train', choices=['train', 'validation'], help='Either train or val data')
    parser.add_argument('--fill_time_series_gaps', action='store_true', help='Fill time series gaps')
    parser.add_argument('--impute_missing_values', action='store_true', help='Impute missing values')
    parser.add_argument('--account_for_missing_intervals', action='store_true', help='Account for missing intervals')

def main():
    """
    Process raw and interim data and store it as processed data.
    """
    parser = argparse.ArgumentParser(description='Data processing script for Energy Forecasting Hackathon')
    parser_add_arguments(parser)
    args = parser.parse_args()

    # Process raw data
    if args.process_raw_data:
        process_raw_data(args)
    
    # Process interim data
    if args.process_interim_data:
        process_interim_data(args)

if __name__ == "__main__":
    main()