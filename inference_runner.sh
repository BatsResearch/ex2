#!/usr/bin/env bash

python inference.py \
    --model_name='/path/to/trained/adapters' \
    --base_model_name='base_model_name_or_path' \
    --output_filepath='/where/to/save/generated/descs' \
    --eval_dataset='pets'
