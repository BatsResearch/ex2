#!/usr/bin/env bash

export NCCL_P2P_LEVEL=NVL
ulimit -Sn $(ulimit -Hn)
RANDOMPORT=$(shuf -i8000-9999 -n1)

accelerate launch \
    --main_process_port "${RANDOMPORT}" \
    train.py \
    --ppo_config.model_name="mistralai/Mistral-7B-v0.1" \
    --ppo_config.log_with="wandb" \
    --vlm_name="openai/clip-vit-base-patch32" \
    --output_dir="model_output/trained_adapters" \
    --data_root="path/to/data/root" \
    --dataset_name="pets" \
    --split="trainval" \
    --save_freq="20"
