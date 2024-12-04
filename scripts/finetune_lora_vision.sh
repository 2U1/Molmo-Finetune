#!/bin/bash

# You can use 2B instead of 7B
MODEL_NAME="allenai/Molmo-7B-D-0924"
# MODEL_NAME="allenai/Molmo-7B-O-0924"

export PYTHONPATH=src:$PYTHONPATH

# If you want to tune the `wte` with LoRA, You need to tune `ff_out` together
# ff_out is the lm_head layer in other models.
# I can't find the exact embed_token so, its better to just tune the ff_out too.
# --lora_namespan_exclude "['ff_out']"

deepspeed src/training/train.py \
    --lora_enable True \
    --vision_lora True \
    --lora_rank 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --num_lora_modules -1 \
    --deepspeed scripts/zero3.json \
    --model_id $MODEL_NAME \
    --data_path /path/to/your/training/data.json \
    --image_folder /path/to/your/image/folder \
    --freeze_vision_tower False \
    --freeze_llm False \
    --tune_projector True \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir output/lora_vision_test \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 2e-4 \
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