#!/bin/bash

# You can use 2B instead of 7B
MODEL_NAME="allenai/Molmo-7B-D-0924"
# MODEL_NAME="allenai/Molmo-7B-O-0924"

export PYTHONPATH=src:$PYTHONPATH

# Currently, molmo does not support gradient_checkpointing
# Also it only supports fp32 training

deepspeed src/training/train.py \
    --deepspeed scripts/zero3_offload.json \
    --model_id $MODEL_NAME \
    --data_path /path/to/your/training/data.json \
    --image_folder /path/to/your/image/folder \
    --freeze_vision_tower False \
    --freeze_llm False \
    --tune_projector True \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir output/fft_test \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --projector_lr 1e-5 \
    --vision_lr 2e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing False \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 10 \
    --dataloader_num_workers 4