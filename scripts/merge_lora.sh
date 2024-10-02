#!/bin/bash

MODEL_NAME="allenai/Molmo-7B-D-0924"
# MODEL_NAME="allenai/Molmo-7B-O-0924"

export PYTHONPATH=src:$PYTHONPATH

python src/merge_lora_weights.py \
    --model-path /path/to/your/model \
    --model-base $MODEL_NAME  \
    --save-model-path /output/directory \
    --safe-serialization