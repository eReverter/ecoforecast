"""
Script containing the metrics used to evaluate the ETL and the performance of the models.
"""
# General imports
import argparse
import os
from termcolor import colored
import warnings
warnings.filterwarnings("ignore")

# Data processing imports
import datetime
import pandas as pd
from sklearn.metrics import (
    f1_score, 
    precision_score, 
    recall_score,
)

# Local imports
from src.definitions import (
    PREDICTIONS_DIR,
)

### ETL STATISTICS ###

class DataProcessingStatistics:
    """
    Class to track statistics of the data processing steps related to the raw data.
    """
    def __init__(self):
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

            print("")

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
    """
    Class to track statistics of the data processing steps related to the interim data.
    """
    def __init__(self):
        self.file_stats = {}
        self.merged_stats = {'rows': 0, 'columns': 0}

    def update_file_stats(self, filename, pre_shape, post_shape):
        self.file_stats[filename] = {'pre_shape': pre_shape, 'post_shape': post_shape}

    def update_merged_stats(self, df):
        self.merged_stats['rows'] = len(df)
        self.merged_stats['columns'] = len(df.columns)

    def display_statistics(self):
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

### PERFORMANCE METRICS ###

def get_model_performance(actual, predictions):
    """
    Compute f1 score between prediction and actual values in a multi-class classification problem.

    :param predictions: list of predicted values.
    :param actual: list of actual values.
    """
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
    actual = pd.read_json(os.path.join(PREDICTIONS_DIR, 'val_predictions.json'))
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
    f1, precision, recall = get_model_performance(actual, predictions)

    # termcolor
    print(colored(f'F1 Score: {f1}', 'blue'))
    print(colored(f'Precision: {precision}', 'green'))
    print(colored(f'Recall: {recall}', 'red'))

if __name__ == '__main__':
    main()