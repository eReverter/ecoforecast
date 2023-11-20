"""
Metrics for the end-to-end pipeline.
"""
import os
import datetime
import json
from termcolor import colored
import pandas as pd
import argparse
from src.definitions import (
    PREDICTIONS_DIR,
)

### ETL METRICS ###

class StatisticsETL:
    def __init__(self, start_date=None, end_date=None):
        """
        Initializes the StatisticsETL object with start and end dates for the ETL process.
        """
        self.start_date = start_date
        self.end_date = end_date
        self.original_count = {}
        self.green_energy_count = {}
        self.imputed_count = {}
        self.aggregated_count = {}
        self.outlier_count = {}
        self.na_count = {}
        self.estimated_frequency = {}

    def update_original_count(self, type_region, df):
        """
        Update the original row count for a given type and region.
        """
        self.original_count[type_region] = df.shape[0]

    def update_green_energy_count(self, type_region, df):
        """
        Update the count of green energy rows for a given type and region.
        """
        self.green_energy_count[type_region] = df.shape[0]

    def update_imputed_count(self, type_region, na_count):
        """
        Update the count of imputed rows for a given type and region.
        """
        self.imputed_count[type_region] = na_count

    def update_aggregated_count(self, type_region, df):
        """
        Update the count of aggregated rows for a given type and region.
        """
        self.aggregated_count[type_region] = df.shape[0]

    def update_na_count(self, type_region, na_count):
        """
        Update the count of NA values for a given type and region.
        """
        self.na_count[type_region] = na_count

    def update_outlier_count(self, type_region, outlier_count):
        """
        Update the count of identified outliers for a given type and region.
        """
        self.outlier_count[type_region] = outlier_count

    def update_estimated_frequency(self, type_region, freq):
        """
        Update the estimated frequency for a given type and region.
        """
        self.estimated_frequency[type_region] = freq

    def generate_statistics(self):
        """
        Generate a comprehensive summary of the collected statistics.
        """
        summary = {
            'Original Data Count': self.original_count,
            'Green Energy Data Count': self.green_energy_count,
            'Imputed Data Count': self.imputed_count,
            'Aggregated Data Count': self.aggregated_count,
            'NA Count': self.na_count,
            'Outlier Count': self.outlier_count,
            'Estimated Frequency': self.estimated_frequency
        }
        return summary
    
    def display_statistics(self):
        """
        Display the statistics with color coding for each dataset.
        """
        for dataset, stats in self.generate_statistics().items():
            print(colored(f'{dataset}:', 'blue'))
            for type_region, count in stats.items():
                print(colored(f'  {type_region}: {count}', 'green'))

    def display_statistics_as_table(self):
        """
        Display the collected statistics as a formatted table.
        """
        # Create a dictionary to hold reformatted statistics
        reformatted_stats = {}

        # Loop through each category of statistics and reformat
        for stat_category, stats in self.generate_statistics().items():
            for type_region, count in stats.items():
                if type_region not in reformatted_stats:
                    reformatted_stats[type_region] = {}
                reformatted_stats[type_region][stat_category] = count

        # Convert to DataFrame for better visualization
        stats_df = pd.DataFrame.from_dict(reformatted_stats, orient='index')
        
        # Print the DataFrame
        print(stats_df)

    def save_statistics_to_json(self, file_path):
        """
        Save the collected statistics to a JSON file.
        """
        with open(file_path, 'w') as file:
            json.dump(self.generate_statistics(), file, indent=4)

    def __repr__(self):
        """
        Represent the StatisticsETL object as a string.
        """
        return f"<StatisticsETL from {self.start_date} to {self.end_date}>"

class DataProcessingStatistics:
    def __init__(self):
        # Using nested dictionaries to track counts for each combination of energy type, and region
        self.data_counts = {}
        self.loss_reasons = {}
        self.estimated_frequency = {}

    def _initialize_key(self, etype, region):
        key = (etype, region)
        if key not in self.data_counts:
            self.data_counts[key] = {
                'original': 0,
                'processed': 0,
                'missing_values': 0,
                'imputed_values': 0,
                'zero_values': 0,
            }
        return key

    def update_counts(self, etype, region, stage, count):
        key = self._initialize_key(etype, region)
        self.data_counts[key][stage] += count

    def log_loss_reason(self, etype, region, reason, count):
        key = (etype, region)
        if key not in self.loss_reasons:
            self.loss_reasons[key] = []
        self.loss_reasons[key].append((reason, count))

    def update_estimated_frequency(self, etype, region, freq):
        key = self._initialize_key(etype, region)
        self.estimated_frequency[key] = freq

    def display_statistics(self):
        # Displaying the statistics in a structured format
        print(colored("\n--- Data Processing Statistics ---\n", 'white', attrs=['bold']))
        for key, stats in self.data_counts.items():
            etype, region = key
            header = f"Energy Type: {etype}, Region: {region}"
            print(colored(header, 'blue', attrs=['bold']))

            freq = self.estimated_frequency[key]
            freq_info = f" Estimated Frequency: {freq}"
            print(colored(freq_info, 'cyan'))

            for stage, count in stats.items():
                stage_info = f"  {stage} Count: {count}"
                print(colored(stage_info, 'white'))

            if key in self.loss_reasons:
                print(colored("  Loss Reasons:", 'red', attrs=['bold']))
                for reason, count in self.loss_reasons[key]:
                    reason_info = f"\t{reason}: {count}"
                    print(colored(reason_info, 'light_red'))

            print("")  # Blank line for better separation

    def generate_report(self, file_path):
        with open(file_path, 'w') as file:
            file.write(f"Data Processing Report\n")
            file.write(f"Generated on: {datetime.datetime.now()}\n\n")

            for key, stats in self.data_counts.items():
                etype, region = key
                file.write(f"Energy Type: {etype}, Region: {region}\n")
                file.write(f" Estimated Frequency: {self.estimated_frequency[key]}\n")
                for stage, count in stats.items():
                    file.write(f"  {stage} Count: {count}\n")
                if key in self.loss_reasons:
                    file.write("  Loss Reasons:\n")
                    for reason, count in self.loss_reasons[key]:
                        file.write(f"\t{reason}: {count}\n")
                file.write("\n")
            
            file.write("End of Report\n")

class InterimDataProcessingStatistics:
    def __init__(self):
        self.file_stats = {}
        self.merged_stats = {'rows': 0, 'columns': 0}

    def update_file_stats(self, filename, pre_shape, post_shape):
        self.file_stats[filename] = {'pre_shape': pre_shape, 'post_shape': post_shape}

    def update_merged_stats(self, df):
        self.merged_stats['rows'] = len(df)
        self.merged_stats['columns'] = len(df.columns)

    def display_statistics(self):
        # Displaying the statistics in a structured format
        print(colored("\n--- Interim Data Processing Statistics ---\n", 'white', attrs=['bold']))
        for filename, stats in self.file_stats.items():
            print(colored(f"File: {filename}", 'blue', attrs=['bold']))
            print(colored(f"\tPre-processing shape: {stats['pre_shape']}", 'green'))
            print(colored(f"\tPost-processing shape: {stats['post_shape']}", 'yellow'))
            print("")

        print(colored("Merged DataFrame Statistics:", 'magenta', attrs=['bold']))
        print(colored(f"\tTotal rows: {self.merged_stats['rows']}", 'green'))
        print(colored(f"\tTotal columns: {self.merged_stats['columns']}", 'yellow'))

    def generate_report(self, file_path):
        with open(file_path, 'w') as file:
            file.write("Interim Data Processing Report\n")
            file.write(f"Generated on: {datetime.datetime.now()}\n\n")

            for filename, stats in self.file_stats.items():
                file.write(f"File: {filename}\n")
                file.write(f"\tPre-processing shape: {stats['pre_shape']}\n")
                file.write(f"\tPost-processing shape: {stats['post_shape']}\n")
                file.write("\n")

            file.write("Merged DataFrame Statistics:\n")
            file.write(f"\tTotal rows: {self.merged_stats['rows']}\n")
            file.write(f"\tTotal columns: {self.merged_stats['columns']}\n")
            file.write("\nEnd of Report\n")

### TRAINING METRICS ###

### EVALUATION METRICS ###

def f1_score(predictions, actual):
    """
    Compute f1 score between prediction and actual values in a multi-class classification problem.
    """
    from sklearn.metrics import f1_score, precision_score, recall_score
    f1 = f1_score(actual, predictions, average='weighted')
    precision = precision_score(actual, predictions, average='weighted')
    recall = recall_score(actual, predictions, average='weighted')
    return f1, precision, recall

### MAIN ###

def main():
    parser = argparse.ArgumentParser(description='Compute evaluation metrics')
    parser.add_argument('--predictions', type=str, help='Path to predictions file')
    args = parser.parse_args()

    # Load predictions and actual values from the .json files
    actual = pd.read_json(os.path.join(PREDICTIONS_DIR, 'predictions.json'))
    predictions = pd.read_json(args.predictions)

    # Ensure the values are paired by timestamp if possible
    if 'timestamp' in actual and 'timestamp' in predictions: # forecasting
        merged = actual.merge(predictions, how='inner', on='timestamp')
        merged.dropna(inplace=True)
        actual = merged.target_x.tolist()
        predictions = merged.target_y.tolist()
    else: # cls
        actual = actual.iloc[:-1].target.tolist()
        predictions = predictions.iloc[:-1].target.tolist()

    # Compute the f1 score
    f1, precision, recall = f1_score(predictions, actual)

    # termcolor
    print(colored(f'F1 Score: {f1}', 'blue'))
    print(colored(f'Precision: {precision}', 'green'))
    print(colored(f'Recall: {recall}', 'red'))


if __name__ == '__main__':
    main()