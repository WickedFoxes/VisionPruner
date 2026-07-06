#!/bin/bash

# 학습 실행 (Overfitting을 위해 Epoch를 100으로 설정)
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/triton_autotune}
export PYTHONNOUSERSITE=${PYTHONNOUSERSITE:-1}
DATA_PATH=${DATA_PATH:-./playground/data/mme_dummy_test.json}
IMAGE_FOLDER=${IMAGE_FOLDER:-./playground/data/mme/images}
OUTPUT_DIR=${OUTPUT_DIR:-./checkpoints/llava-v1.5-7b-vision-pruner-overfit-test}
ADAM_EPSILON=${ADAM_EPSILON:-1e-6}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-0.1}

if [[ -z "${LEARNING_RATE+x}" ]]; then
    LEARNING_RATE=1e-5
fi

python llava/train/train_vision_pruner.py \
    --model_name_or_path liuhaotian/llava-v1.5-7b \
    --version v1 \
    --data_path "$DATA_PATH" \
    --image_folder "$IMAGE_FOLDER" \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --vision_pruner_value_layer_idx 0 \
    --vision_pruner_context_layer_idx 9 \
    --bf16 True \
    --output_dir "$OUTPUT_DIR" \
    --overwrite_output_dir True \
    --num_train_epochs 40 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --learning_rate $LEARNING_RATE \
    --adam_epsilon "$ADAM_EPSILON" \
    --max_grad_norm "$MAX_GRAD_NORM" \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "constant" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none
