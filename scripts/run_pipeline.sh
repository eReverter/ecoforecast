#!/bin/bash
export PYTHONPATH=$(pwd):$PYTHONPATH

# echo "Running data ingestion for training data..."
# python src/data/data_ingestion.py \
#     --start_time "2022-01-01" \
#     --end_time "2023-01-31" \
#     --output_path data/raw/train

# echo "Running data ingestion for validation data..."
# python src/data/data_ingestion.py \
#     --start_time "2022-01-01" \
#     --end_time "2023-01-31" \
#     --output_path data/raw/validation

# echo "Running data ingestion for holidays data..."
# python src/data/holiday_ingestion.py \
#     --start_year 2022 \
#     --end_year 2023

echo "Processing training data..."
python src/data/data_processing.py \
    --process_raw_data \
    --interpolate_zeros \
    --process_interim_data \
    --mode train

echo "Processing validation data..."
python src/data/data_processing.py \
    --process_raw_data \
    --interpolate_zeros \
    --process_interim_data \
    --mode validation

echo "Visualizing training data..."
python src/visualization/visualize.py \
    --mode train

echo "Visualizing validation data..."
python src/visualization/visualize.py \
    --mode validation

echo "Preparing data for evaluation..."
python src/data/prepare_data.py \
    --validation data/processed/validation.csv

echo "Training XGBoost CLS model..."
python src/model/classification/xgboost/model_training.py 

echo "Training XGBoost REG model..."
python src/model/forecasting/xgboost/model_training.py

echo "Training LightGBM REG model..."
python src/model/forecasting/lightgbm/model_training.py

echo "Training LSTM model..."
python src/model/forecasting/lstm/model_training.py

echo "Predicting with XGBoost CLS model..."
python src/model/classification/xgboost/model_prediction.py \
    --model models/classification/xgboost/model.json \
    --encoder models/classification/xgboost/label_encoder.npy

echo "Predicting with XGBoost REG model..."
python src/model/forecasting/xgboost/model_prediction.py \
    --model models/forecasting/xgboost/model.json

echo "Predicting with LightGBM REG model..."
python src/model/forecasting/lightgbm/model_prediction.py \
    --model models/forecasting/lightgbm/model.txt

echo "Predicting with LSTM model..."
python src/model/forecasting/lstm/model_prediction.py \
    --model models/forecasting/lstm/model.pth

echo "Evaluating naive baseline predictions..."
python src/metrics.py \
    --predictions predictions/baseline.json

echo "Evaluating XGBoost CLS predictions..."
python src/metrics.py \
    --predictions predictions/xgboost_cls_predictions.json

echo "Evaluating XGBoost REG predictions..."
python src/metrics.py \
    --predictions predictions/xgboost_reg_predictions.json

echo "Evaluating LightGBM predictions..."
python src/metrics.py \
    --predictions predictions/lightgbm_reg_predictions.json

echo "Evaluating LSTM model predictions..."
python src/metrics.py \
    --predictions predictions/lstm_predictions.json

echo "Pipeline completed."
