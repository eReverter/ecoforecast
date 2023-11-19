"""
Enhanced script to train an LSTM model.
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import argparse
import json
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

from src.data.prepare_data import (
    load_data,
    add_is_weekend,
    get_surplus,
    get_forecast_target,
    convert_to_timeseries,
    get_lags,
    add_is_holiday,
    get_ohe_from_cat,
)
from src.definitions import PROCESSED_DATA_DIR, MODELS_DIR
from src.config import setup_logger

logger = setup_logger()

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size=50, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size

        # Adding an additional LSTM layer and introducing dropout
        self.lstm1 = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(hidden_layer_size, hidden_layer_size, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out1, _ = self.lstm1(input_seq)
        dropout_out1 = self.dropout1(lstm_out1)
        lstm_out2, _ = self.lstm2(dropout_out1)
        dropout_out2 = self.dropout2(lstm_out2)

        # Getting the output from the last LSTM layer for each sequence
        predictions = self.linear(dropout_out2[:, -1, :])
        return predictions

def prepare_data(df):
    df = add_is_weekend(df)
    df = get_surplus(df)
    df = convert_to_timeseries(df, metadata_columns=['is_weekend'])
    df = add_is_holiday(df)
    df = get_ohe_from_cat(df, cat='series_id')

    # No need to add lags as separate columns
    df = df.replace(0, np.nan)  # Replace 0s with NaNs if necessary
    df.dropna(inplace=True)  # Drop rows with NaNs
    return df

def create_sequences(data, lags):
    xs, ys = [], []
    for i in range(len(data) - lags):
        x = data.iloc[i:(i + lags)].to_numpy()
        y = data.iloc[i + lags]['surplus']  # Assuming 'surplus' is your target variable
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def train_model(model, train_loader, learning_rate=0.001, epochs=10):
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for seq, labels in tqdm(train_loader):
            optimizer.zero_grad()
            y_pred = model(seq)
            loss = loss_function(y_pred.squeeze(), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        logger.info(f'Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}')

def parser_add_arguments(parser):
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    return parser

def main():
    parser = argparse.ArgumentParser(description='Train LSTM model')
    parser_add_arguments(parser)
    args = parser.parse_args()

    # Load data
    train, _ = load_data()

    # Data Preparation
    logger.info("Preparing data...")
    train = prepare_data(train)

    # Prepare data for training in LSTM
    x_train, y_train = create_sequences(train.drop(['timestamp', 'series_id'], axis=1), lags=3)

    # Normalizing the data
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train.reshape(-1, x_train.shape[2]))
    x_train_scaled = x_train_scaled.reshape(-1, 3, x_train.shape[2])  # 3 is the number of lags

    y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1)).flatten()

    x_train_tensor = torch.tensor(x_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)

    train_data = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

    # Model configuration
    CONFIG_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model_config.json')
    with open(CONFIG_PATH, 'r') as f:
        model_config = json.load(f)  # LSTM configuration

    model = LSTMModel(input_size=x_train_tensor.shape[2], **model_config)
    
    # Training
    train_model(model, train_loader, learning_rate=0.001, epochs=7)

    # Save Model
    model_path = os.path.join(f'{MODELS_DIR}/forecasting/lstm', 'lstm_model.pth')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model trained and saved at {model_path}")

if __name__ == "__main__":
    main()
