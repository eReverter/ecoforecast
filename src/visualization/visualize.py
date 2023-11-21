"""
Script to visualize the processed data.
"""
# General imports
import argparse
import warnings
warnings.filterwarnings("ignore")

# Data related imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
import seaborn as sns

# Local imports
from src.definitions import (
    PROCESSED_DATA_DIR,
    FIGURES_DIR,
    TYPE,
)
from src.data.prepare_data import (
    get_surplus,
    get_curr_max,
)

### MAIN ###

def add_parser_arguments(parser):
    parser.add_argument("--mode", type=str, help="Either train or validation")

def main():
    parser = argparse.ArgumentParser(description="Visualize Processed Energy Data")
    add_parser_arguments(parser)
    args = parser.parse_args()

    if args.mode == "train":
        file_path = f"{PROCESSED_DATA_DIR}/train.csv"
    elif args.mode == "validation":
        file_path = f"{PROCESSED_DATA_DIR}/validation.csv"
    else:
        raise ValueError("Invalid mode. Please choose either train or validation.")
    
    # Set global plot properties
    rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

    ### Load the processed data ###
    df = pd.read_csv(file_path, parse_dates=['timestamp'])

    # Binary Heatmap of Missing Values
    print("Plotting Binary Heatmap of Missing Values")
    sorted_columns = sorted(df.columns, key=lambda x: (x.split('_')[0], x.split('_')[1]) if '_' in x else (x, ''))

    # Create month labels
    month_labels = df['timestamp'].dt.to_period('M').astype(str).unique()
    row_indices = np.linspace(0, len(df) - 1, len(month_labels), dtype=int)

    plt.figure(figsize=(15, 8))
    sns.heatmap(df[sorted_columns].isnull(), cbar=False, yticklabels=False, cmap='cividis')
    plt.title("Binary Heatmap of Missing Values")
    plt.xlabel("Columns")

    # Set the y-ticks to display month labels
    plt.yticks(row_indices, month_labels, rotation=0)

    # Save the figure
    plt.savefig(f"{FIGURES_DIR}/{args.mode}_missing_values_heatmap.png")
    plt.close()

    ### Line Plots for Energy Evolution Over Time by Region ###
    print("Plotting Line Plots for Energy Evolution Over Time by Region")
    regions = sorted(set(col.split('_')[0] for col in df.columns if '_' in col))
    for region in regions:
        fig, ax = plt.subplots(figsize=(10, 5))
        for etype in TYPE:
            column_name = f'{region}_{etype}'
            if column_name in df.columns:
                rolling_mean = df[column_name].rolling(window=24).mean() # Get smoothed visualization
                ax.plot(df['timestamp'], rolling_mean, label=f'{region} {etype} (Smoothed)')
                ax.plot(df['timestamp'], df[column_name], alpha=0.1)
        ax.set_title(f"Energy Evolution Over Time - {region}")
        ax.legend(loc='upper right')
        plt.tight_layout()

        # Save the figure
        plt.savefig(f"{FIGURES_DIR}/{args.mode}_{region}_energy_evolution.png")
        plt.close()

    # Calculate surplus and max surplus region
    print("Calculating Surplus and Max Surplus Region")
    df = get_surplus(df)
    df = get_curr_max(df)

    ### Plot for the maximum region at every timestamp ###
    plt.figure(figsize=(15, 8))

    # Sort regions alphabetically
    unique_regions = sorted(df['curr_max'].unique())

    # Assign y-positions based on sorted regions
    y_positions = range(len(unique_regions))
    region_positions = {region: pos for region, pos in zip(unique_regions, y_positions)}
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_regions)))

    for region, color in zip(unique_regions, colors):
        region_df = df[df['curr_max'] == region]
        plt.scatter(region_df['timestamp'], region_df['curr_max'].map(region_positions), color=color, alpha=0.6, label=region)

    plt.yticks(y_positions, unique_regions)
    plt.xlabel('Timestamp')
    plt.ylabel('Region with Max Surplus')
    plt.title('Region with Maximum Surplus at Each Timestamp')
    plt.legend()

    # Save the figure
    plt.savefig(f"{FIGURES_DIR}/{args.mode}_max_surplus_region.png")
    plt.close()

    
if __name__ == "__main__":
    main()