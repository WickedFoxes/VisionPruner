#!/bin/bash
set -euo pipefail

DATASETS="${1:-pope}"
MODEL_PATH="${MODEL_PATH:-}"
MODEL_BASE="${MODEL_BASE:-}"
ATTENTION_SCORE_PRUNING_TOKEN_BUDGET="${ATTENTION_SCORE_PRUNING_TOKEN_BUDGET:-32}"
ATTENTION_SCORE_PRUNING_NUM_IMAGE_TOKENS="${ATTENTION_SCORE_PRUNING_NUM_IMAGE_TOKENS:-576}"
ATTENTION_SCORE_PRUNING_TOP_P="${ATTENTION_SCORE_PRUNING_TOP_P:-}"
ATTENTION_SCORE_PRUNING_LAYER="${ATTENTION_SCORE_PRUNING_LAYER:-}"
ATTENTION_SCORE_PRUNING_HEAD_REDUCTION="${ATTENTION_SCORE_PRUNING_HEAD_REDUCTION:-mean}"
ATTENTION_SCORE_PRUNING_DEBUG_TOPK="${ATTENTION_SCORE_PRUNING_DEBUG_TOPK:-0}"
ATTENTION_SCORE_PRUNING_DEBUG_TOPK_LIMIT="${ATTENTION_SCORE_PRUNING_DEBUG_TOPK_LIMIT:-5}"
CONV_MODE="${CONV_MODE:-vicuna_v1}"
TEMPERATURE="${TEMPERATURE:-0}"
RUN_EVAL="${RUN_EVAL:-1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-./.cache/triton-${USER:-user}}"

if [ -z "$MODEL_PATH" ]; then
    echo "Set MODEL_PATH to a LLaVA checkpoint path." >&2
    exit 1
fi

mkdir -p "$TRITON_CACHE_DIR"
export TRITON_CACHE_DIR
export CUDA_VISIBLE_DEVICES

if [ -n "$ATTENTION_SCORE_PRUNING_TOP_P" ]; then
    CKPT="${CKPT:-$(basename "$MODEL_PATH")-attnscore-top${ATTENTION_SCORE_PRUNING_TOP_P}}"
else
    CKPT="${CKPT:-$(basename "$MODEL_PATH")-attnscore${ATTENTION_SCORE_PRUNING_TOKEN_BUDGET}}"
fi

COMMON_ARGS=(
    --model-path "$MODEL_PATH"
    --temperature "$TEMPERATURE"
    --conv-mode "$CONV_MODE"
    --use-attention-score-pruning
    --attention-score-pruning-head-reduction "$ATTENTION_SCORE_PRUNING_HEAD_REDUCTION"
)

if [ -n "$MODEL_BASE" ]; then
    COMMON_ARGS+=(--model-base "$MODEL_BASE")
fi

if [ -n "$ATTENTION_SCORE_PRUNING_LAYER" ]; then
    COMMON_ARGS+=(--attention-score-pruning-layer "$ATTENTION_SCORE_PRUNING_LAYER")
fi

if [ -n "$ATTENTION_SCORE_PRUNING_TOP_P" ]; then
    COMMON_ARGS+=(--attention-score-pruning-top-p "$ATTENTION_SCORE_PRUNING_TOP_P")
else
    COMMON_ARGS+=(
        --attention-score-pruning-token-budget "$ATTENTION_SCORE_PRUNING_TOKEN_BUDGET"
        --attention-score-pruning-num-image-tokens "$ATTENTION_SCORE_PRUNING_NUM_IMAGE_TOKENS"
    )
fi

if [ "$ATTENTION_SCORE_PRUNING_DEBUG_TOPK" = "1" ]; then
    COMMON_ARGS+=(--attention-score-pruning-debug-topk)
    COMMON_ARGS+=(--attention-score-pruning-debug-topk-limit "$ATTENTION_SCORE_PRUNING_DEBUG_TOPK_LIMIT")
fi

run_pope() {
    local answer_file="./playground/data/eval/pope/answers/$CKPT.jsonl"
    python -m llava.eval.model_vqa_loader \
        "${COMMON_ARGS[@]}" \
        --question-file "./playground/data/eval/pope/llava_pope_test.jsonl" \
        --image-folder "./playground/data/eval/pope/val2014" \
        --answers-file "$answer_file"

    if [ "$RUN_EVAL" = "1" ]; then
        python -m llava.eval.eval_pope \
            --annotation-dir "./playground/data/eval/pope/coco" \
            --question-file "./playground/data/eval/pope/llava_pope_test.jsonl" \
            --result-file "$answer_file"
    fi
}

if [ "$DATASETS" = "all" ]; then
    DATASETS="pope"
fi

for dataset in ${DATASETS//,/ }; do
    case "$dataset" in
        pope) run_pope ;;
        *)
            echo "Unknown dataset: $dataset" >&2
            exit 1
            ;;
    esac
done
