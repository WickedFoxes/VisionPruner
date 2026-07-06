#!/bin/bash
set -euo pipefail

DATASETS="${1:-all}"
MODEL_PATH="${MODEL_PATH:-}"
MODEL_BASE="${MODEL_BASE:-}"
VISION_PRUNER_TOKEN_BUDGET="${VISION_PRUNER_TOKEN_BUDGET:-32}"
VISION_PRUNER_NUM_IMAGE_TOKENS="${VISION_PRUNER_NUM_IMAGE_TOKENS:-576}"
VISION_PRUNER_TOP_P="${VISION_PRUNER_TOP_P:-}"
CONV_MODE="${CONV_MODE:-vicuna_v1}"
TEMPERATURE="${TEMPERATURE:-0}"
RUN_EVAL="${RUN_EVAL:-1}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-./.cache/triton-${USER:-user}}"
VISION_PRUNER_DEBUG_WEIGHTS="${VISION_PRUNER_DEBUG_WEIGHTS:-0}"
VISION_PRUNER_DEBUG_WEIGHTS_ONLY="${VISION_PRUNER_DEBUG_WEIGHTS_ONLY:-0}"
VISION_PRUNER_DEBUG_TOPK="${VISION_PRUNER_DEBUG_TOPK:-0}"
VISION_PRUNER_DEBUG_TOPK_LIMIT="${VISION_PRUNER_DEBUG_TOPK_LIMIT:-5}"
VISION_PRUNER_VERBOSE="${VISION_PRUNER_VERBOSE:-0}"

if [ -z "$MODEL_PATH" ]; then
    echo "Set MODEL_PATH to a LLaVA VisionPruner checkpoint path." >&2
    exit 1
fi

mkdir -p "$TRITON_CACHE_DIR"
export TRITON_CACHE_DIR

CKPT="${CKPT:-$(basename "$MODEL_PATH")-vp${VISION_PRUNER_TOKEN_BUDGET}}"

COMMON_ARGS=(
    --model-path "$MODEL_PATH"
    --temperature "$TEMPERATURE"
    --conv-mode "$CONV_MODE"
    --use-vision-pruner
)

if [ "$VISION_PRUNER_DEBUG_WEIGHTS" = "1" ]; then
    COMMON_ARGS+=(--vision-pruner-debug-weights)
fi

if [ "$VISION_PRUNER_DEBUG_WEIGHTS_ONLY" = "1" ]; then
    COMMON_ARGS+=(--vision-pruner-debug-weights-only)
fi

if [ "$VISION_PRUNER_DEBUG_TOPK" = "1" ]; then
    COMMON_ARGS+=(--vision-pruner-debug-topk)
    COMMON_ARGS+=(--vision-pruner-debug-topk-limit "$VISION_PRUNER_DEBUG_TOPK_LIMIT")
fi

if [ "$VISION_PRUNER_VERBOSE" = "1" ]; then
    COMMON_ARGS+=(--vision-pruner-verbose)
fi

if [ -n "$MODEL_BASE" ]; then
    COMMON_ARGS+=(--model-base "$MODEL_BASE")
fi

if [ -n "$VISION_PRUNER_TOP_P" ]; then
    COMMON_ARGS+=(--vision-pruner-top-p "$VISION_PRUNER_TOP_P")
else
    COMMON_ARGS+=(
        --vision-pruner-token-budget "$VISION_PRUNER_TOKEN_BUDGET"
        --vision-pruner-num-image-tokens "$VISION_PRUNER_NUM_IMAGE_TOKENS"
    )
fi

if [ "$VISION_PRUNER_DEBUG_WEIGHTS_ONLY" = "1" ]; then
    python -m llava.eval.model_vqa_loader \
        "${COMMON_ARGS[@]}" \
        --question-file "./playground/data/eval/pope/llava_pope_test.jsonl" \
        --image-folder "./playground/data/eval/pope/val2014" \
        --answers-file "./playground/data/eval/pope/answers/${CKPT}.debug.jsonl"
    exit 0
fi

run_gqa() {
    gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
    IFS=',' read -ra GPULIST <<< "$gpu_list"
    local chunks="${#GPULIST[@]}"
    local split="${GQA_SPLIT:-llava_gqa_testdev_balanced}"
    local gqa_dir="${GQA_DIR:-./playground/data/eval/gqa/data}"
    local answer_dir="./playground/data/eval/gqa/answers/$split/$CKPT"

    mkdir -p "$answer_dir"
    for idx in $(seq 0 $((chunks - 1))); do
        CUDA_VISIBLE_DEVICES="${GPULIST[$idx]}" python -m llava.eval.model_vqa_loader \
            "${COMMON_ARGS[@]}" \
            --question-file "./playground/data/eval/gqa/$split.jsonl" \
            --image-folder "./playground/data/eval/gqa/data/images" \
            --answers-file "$answer_dir/${chunks}_${idx}.jsonl" \
            --num-chunks "$chunks" \
            --chunk-idx "$idx" &
    done
    wait

    local output_file="$answer_dir/merge.jsonl"
    : > "$output_file"
    for idx in $(seq 0 $((chunks - 1))); do
        cat "$answer_dir/${chunks}_${idx}.jsonl" >> "$output_file"
    done

    python scripts/convert_gqa_for_eval.py \
        --src "$output_file" \
        --dst "$gqa_dir/testdev_balanced_predictions.json"

    if [ "$RUN_EVAL" = "1" ]; then
        (cd "$gqa_dir" && python eval/eval.py --tier testdev_balanced)
    fi
}

run_vqav2() {
    gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
    IFS=',' read -ra GPULIST <<< "$gpu_list"
    local chunks="${#GPULIST[@]}"
    local split="${VQAV2_SPLIT:-llava_vqav2_mscoco_test-dev2015}"
    local answer_dir="./playground/data/eval/vqav2/answers/$split/$CKPT"

    mkdir -p "$answer_dir"
    for idx in $(seq 0 $((chunks - 1))); do
        CUDA_VISIBLE_DEVICES="${GPULIST[$idx]}" python -m llava.eval.model_vqa_loader \
            "${COMMON_ARGS[@]}" \
            --question-file "./playground/data/eval/vqav2/$split.jsonl" \
            --image-folder "./playground/data/eval/vqav2/test2015" \
            --answers-file "$answer_dir/${chunks}_${idx}.jsonl" \
            --num-chunks "$chunks" \
            --chunk-idx "$idx" &
    done
    wait

    local output_file="$answer_dir/merge.jsonl"
    : > "$output_file"
    for idx in $(seq 0 $((chunks - 1))); do
        cat "$answer_dir/${chunks}_${idx}.jsonl" >> "$output_file"
    done

    python scripts/convert_vqav2_for_submission.py --split "$split" --ckpt "$CKPT"
}

run_textvqa() {
    local answer_file="./playground/data/eval/textvqa/answers/$CKPT.jsonl"
    python -m llava.eval.model_vqa_loader \
        "${COMMON_ARGS[@]}" \
        --question-file "./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl" \
        --image-folder "./playground/data/eval/textvqa/train_images" \
        --answers-file "$answer_file"

    if [ "$RUN_EVAL" = "1" ]; then
        python -m llava.eval.eval_textvqa \
            --annotation-file "./playground/data/eval/textvqa/TextVQA_0.5.1_val.json" \
            --result-file "$answer_file"
    fi
}

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

run_mme() {
    local answer_file="./playground/data/eval/MME/answers/$CKPT.jsonl"
    python -m llava.eval.model_vqa_loader \
        "${COMMON_ARGS[@]}" \
        --question-file "./playground/data/eval/MME/llava_mme.jsonl" \
        --image-folder "./playground/data/eval/MME/MME_Benchmark_release_version" \
        --answers-file "$answer_file"

    if [ "$RUN_EVAL" = "1" ]; then
        (
            cd "./playground/data/eval/MME"
            python convert_answer_to_mme.py --experiment "$CKPT"
            cd eval_tool
            python calculation.py --results_dir "answers/$CKPT"
        )
    fi
}

run_mmbench() {
    local split="${MMBENCH_SPLIT:-mmbench_dev_20230712}"
    local answer_dir="./playground/data/eval/mmbench/answers/$split"
    local upload_dir="./playground/data/eval/mmbench/answers_upload/$split"
    mkdir -p "$answer_dir" "$upload_dir"

    python -m llava.eval.model_vqa_mmbench \
        "${COMMON_ARGS[@]}" \
        --question-file "./playground/data/eval/mmbench/$split.tsv" \
        --answers-file "$answer_dir/$CKPT.jsonl" \
        --single-pred-prompt

    python scripts/convert_mmbench_for_submission.py \
        --annotation-file "./playground/data/eval/mmbench/$split.tsv" \
        --result-dir "$answer_dir" \
        --upload-dir "$upload_dir" \
        --experiment "$CKPT"
}

run_mmbench_cn() {
    local split="${MMBENCH_CN_SPLIT:-mmbench_dev_cn_20231003}"
    local answer_dir="./playground/data/eval/mmbench_cn/answers/$split"
    local upload_dir="./playground/data/eval/mmbench_cn/answers_upload/$split"
    mkdir -p "$answer_dir" "$upload_dir"

    python -m llava.eval.model_vqa_mmbench \
        "${COMMON_ARGS[@]}" \
        --question-file "./playground/data/eval/mmbench_cn/$split.tsv" \
        --answers-file "$answer_dir/$CKPT.jsonl" \
        --single-pred-prompt \
        --lang cn

    python scripts/convert_mmbench_for_submission.py \
        --annotation-file "./playground/data/eval/mmbench_cn/$split.tsv" \
        --result-dir "$answer_dir" \
        --upload-dir "$upload_dir" \
        --experiment "$CKPT"
}

run_sqa() {
    local answer_file="./playground/data/eval/scienceqa/answers/$CKPT.jsonl"
    python -m llava.eval.model_vqa_science \
        "${COMMON_ARGS[@]}" \
        --question-file "./playground/data/eval/scienceqa/llava_test_CQM-A.json" \
        --image-folder "./playground/data/eval/scienceqa/images/test" \
        --answers-file "$answer_file" \
        --single-pred-prompt

    if [ "$RUN_EVAL" = "1" ]; then
        python -m llava.eval.eval_science_qa \
            --base-dir "./playground/data/eval/scienceqa" \
            --result-file "$answer_file" \
            --output-file "./playground/data/eval/scienceqa/answers/${CKPT}_output.jsonl" \
            --output-result "./playground/data/eval/scienceqa/answers/${CKPT}_result.json"
    fi
}

if [ "$DATASETS" = "all" ]; then
    DATASETS="gqa,mmbench,mme,pope,sqa,vqav2,textvqa"
fi

for dataset in ${DATASETS//,/ }; do
    case "$dataset" in
        gqa) run_gqa ;;
        mmb|mmbench) run_mmbench ;;
        mmb_cn|mmbench_cn|mmbcn) run_mmbench_cn ;;
        mme) run_mme ;;
        pope) run_pope ;;
        sqa|scienceqa) run_sqa ;;
        vqav2|vqa_v2) run_vqav2 ;;
        textvqa|vqa_text) run_textvqa ;;
        *)
            echo "Unknown dataset: $dataset" >&2
            exit 1
            ;;
    esac
done
