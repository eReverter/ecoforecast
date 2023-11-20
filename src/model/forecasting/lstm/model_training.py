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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import warnings
from itertools import product
from sklearn.model_selection import train_test_split
import joblib

warnings.filterwarnings('ignore', category=FutureWarning)

from src.data.prepare_data import (
    load_data,
    add_is_weekend,
    get_surplus,
    get_forecast_target,
    convert_to_timeseries,
    add_is_holiday,
    get_ohe_from_cat,
)
from src.definitions import MODELS_DIR, VAL_SIZE, SEED
from src.config import setup_logger

logger = setup_logger()

# Set seeds for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

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

def evaluate_model(model, val_loader, loss_function):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for seq, labels in val_loader:
            y_pred = model(seq)
            loss = loss_function(y_pred.squeeze(), labels)
            total_loss += loss.item()
    return total_loss / len(val_loader)

def parser_add_arguments(parser):
    parser.add_argument('--use-grid', action='store_true', help='Use grid search for model tuning')
    parser.add_argument('--scaler', type=str, default='standard', choices=['standard', 'minmax'], help='Scaler to use')
    parser.add_argument('--cnn', action='store_true', help='Use CNNLSTM model')
    return parser

def main():
    parser = argparse.ArgumentParser(description='Train LSTM model')
    parser_add_arguments(parser)
    args = parser.parse_args()

    train, _ = load_data()
    train = prepare_data(train)

    # Splitting data for validation
    train_data, val_data = train_test_split(train, test_size=VAL_SIZE, random_state=SEED)
    x_train, y_train = create_sequences(train_data.drop(['timestamp', 'series_id'], axis=1), lags=3)
    x_val, y_val = create_sequences(val_data.drop(['timestamp', 'series_id'], axis=1), lags=3)

    # Normalizing the data
    if args.scaler == 'standard':
        x_scaler = StandardScaler()
        y_scaler = StandardScaler()
    elif args.scaler == 'minmax':
        x_scaler = MinMaxScaler()
        y_scaler = MinMaxScaler()
    else:
        raise ValueError("Invalid scaler")

    # Scale training data (features)
    x_train_scaled = x_scaler.fit_transform(x_train.reshape(-1, x_train.shape[2]))
    x_train_scaled = x_train_scaled.reshape(-1, 3, x_train.shape[2])  # 3 is the number of lags

    # Scale validation data using the same scaler (features)
    x_val_scaled = x_scaler.transform(x_val.reshape(-1, x_val.shape[2]))
    x_val_scaled = x_val_scaled.reshape(-1, 3, x_val.shape[2])

    # If you decide to use the same scaler for y
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1)).flatten()

    # Convert to PyTorch tensors
    x_train_tensor = torch.tensor(x_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
    x_val_tensor = torch.tensor(x_val_scaled, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32)

    # Create TensorDataset and DataLoader for data
    train_data = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_data = TensorDataset(x_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False)

    CONFIG_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model_config.json')
    if args.use_grid:
        hidden_layer_sizes = [50, 100, 150]
        learning_rates = [0.001, 0.01]
        num_epochs = [5, 10]

        best_val_loss = float('inf')
        best_model_params = {}

        for hidden_size, lr, epochs in product(hidden_layer_sizes, learning_rates, num_epochs):
            architecture = LSTMModel if not args.cnn else CNNLSTMModel
            model = architecture(input_size=x_train_tensor.shape[2], hidden_layer_size=hidden_size)
            # model = LSTMModel(input_size=x_train_tensor.shape[2], hidden_layer_size=hidden_size)
            train_model(model, train_loader, learning_rate=lr, epochs=epochs)
            
            val_loss = evaluate_model(model, val_loader, nn.MSELoss())
            logger.info(f"Val Loss: {val_loss} for params {hidden_size}, {lr}, {epochs}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_params = {'hidden_layer_size': hidden_size, 'learning_rate': lr, 'epochs': epochs}
                best_model = model
                # best_model_state = model.state_dict()

        logger.info(f"Best Model Params: {best_model_params}, Loss: {best_val_loss}")

        # Save the best model parameters to config
        with open(CONFIG_PATH, 'w') as f:
            json.dump(best_model_params, f)

        model = best_model
    else:
        # Load model configuration
        with open(CONFIG_PATH, 'r') as f:
            model_config = json.load(f)
        model = LSTMModel(input_size=x_train_tensor.shape[2], hidden_layer_size=model_config['hidden_layer_size'])
        train_model(model, train_loader, learning_rate=model_config['learning_rate'], epochs=model_config['epochs'])

    # Save Model
    model_path = os.path.join(MODELS_DIR, 'forecasting', 'lstm', 'model.pth')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model trained and saved at {model_path}")

    # Save scalers
    x_scaler_path = os.path.join(MODELS_DIR, 'forecasting/lstm', 'x_scaler.pkl')
    y_scaler_path = os.path.join(MODELS_DIR, 'forecasting/lstm', 'y_scaler.pkl')
    joblib.dump(x_scaler, x_scaler_path)
    joblib.dump(y_scaler, y_scaler_path)
    logger.info(f"Scalers saved at {x_scaler_path} and {y_scaler_path}")

if __name__ == "__main__":
    main()

