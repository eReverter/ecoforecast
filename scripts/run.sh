#!/bin/bash
export PYTHONPATH=$(pwd):$PYTHONPATH


# python src/data/data_ingestion.py

# python src/data/data_processing.py \
#     --process_raw_data \
#     --process_interim_data \
#     --mode validation
    
# python src/data/prepare_data.py \
#     --validation data/processed/validation.csv

# python src/model/baseline.py \
#     --data data/processed/validation.csv \

python src/model/classification/xgboost/model_training.py

python src/model/classification/xgboost/model_prediction.py \
    --model models/xgboost/model.json \
    --encoder models/xgboost/label_encoder.npy \
    --data data/processed/validation.csv

python src/metrics.py \
    --predictions predictions/xgboost_cls_predictions.json