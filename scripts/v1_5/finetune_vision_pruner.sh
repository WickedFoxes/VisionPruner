#!/bin/bash

export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/triton_autotune}
export PYTHONNOUSERSITE=${PYTHONNOUSERSITE:-1}
DATA_PATH=${DATA_PATH:-./playground/data/llava_instruct_150k.json}
DATA_FRACTION=${DATA_FRACTION:-1.0}
DATA_SEED=${DATA_SEED:-42}
IMAGE_FOLDER=${IMAGE_FOLDER:-./playground/data/coco/train2017}
VISION_PRUNER_TRAIN_MODE=${VISION_PRUNER_TRAIN_MODE:-}
VISION_PRUNER_INIT_MODE=${VISION_PRUNER_INIT_MODE:-llm}
OUTPUT_DIR=${OUTPUT_DIR:-./checkpoints/llava-v1.5-7b-vision-pruner}
ADAM_EPSILON=${ADAM_EPSILON:-1e-6}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-0.1}

if [[ -z "${LEARNING_RATE+x}" ]]; then
    case "$VISION_PRUNER_TRAIN_MODE" in
        llm_freeze_q|llm_freeze_k|llm_freeze_kv|llm_freeze_kv_random_rest|llm_freeze_v_ffn|llm_freeze_v_ffn_random_rest)
            LEARNING_RATE=1e-6
            ;;
        *)
            LEARNING_RATE=2e-5
            ;;
    esac
fi

deepspeed --num_gpus 2 llava_lp/train/train_vision_pruner.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path liuhaotian/llava-v1.5-7b \
    --version v1 \
    --data_path "$DATA_PATH" \
    --data_fraction "$DATA_FRACTION" \
    --data_subset_seed "$DATA_SEED" \
    --image_folder "$IMAGE_FOLDER" \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --vision_pruner_train_mode "$VISION_PRUNER_TRAIN_MODE" \
    --vision_pruner_init_mode "$VISION_PRUNER_INIT_MODE" \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --vision_pruner_decoder_layer_idx 0 \
    --bf16 True \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 1 \
    --learning_rate $LEARNING_RATE \
    --adam_epsilon "$ADAM_EPSILON" \
    --max_grad_norm "$MAX_GRAD_NORM" \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
