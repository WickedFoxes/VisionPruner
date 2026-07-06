# VisionPruner 학습 가이드

이 문서는 `scripts/v1_5/finetune_vision_pruner.sh`로 VisionPruner를 학습할 때 필요한 환경 구성, 데이터 준비, 실행 방법을 정리한 참고 문서입니다.

## 1. Conda 환경 생성

권장 conda env 이름은 `visionpruner`입니다. CUDA 버전에 맞는 PyTorch wheel만 필요하면 바꿔서 설치하세요. 아래 예시는 CUDA 12.1 기준입니다.

```bash
conda create -n visionpruner python=3.11 -y
conda activate visionpruner

pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers tokenizers accelerate deepspeed peft bitsandbytes
pip install sentencepiece protobuf pillow numpy scipy scikit-learn tqdm einops timm wandb
```

선택 설치:

```bash
# flash attention을 쓸 경우. CUDA/PyTorch 조합이 맞지 않으면 생략하세요.
pip install flash-attn --no-build-isolation

# xformers attention monkey patch를 쓸 경우.
pip install xformers
```

GPU가 bf16을 지원하지 않으면 스크립트의 `--bf16 True`를 `--bf16 False --fp16 True`로 바꾸는 편이 안전합니다.

## 2. Repo 준비

스크립트는 반드시 repo 루트에서 실행합니다.

```bash
cd /home/seokhun/VisionPruner
```

`finetune_vision_pruner.sh`는 `./scripts/zero2.json`을 사용합니다. 파일이 없다면 아래 예시로 생성할 수 있습니다.

```json
{
  "bf16": {
    "enabled": "auto"
  },
  "fp16": {
    "enabled": "auto"
  },
  "zero_optimization": {
    "stage": 2,
    "allgather_partitions": true,
    "allgather_bucket_size": 200000000,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 200000000,
    "contiguous_gradients": true
  },
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": "auto",
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto"
}
```

Hugging Face 모델을 처음 받는 환경이라면 로그인도 해두세요.

```bash
huggingface-cli login
```

## 3. 데이터 준비

기본 스크립트는 아래 경로를 사용합니다.

```bash
DATA_PATH=./playground/data/llava_instruct_150k.json
IMAGE_FOLDER=./playground/data/coco/train2017
```

`DATA_PATH`는 LLaVA instruction JSON 형식을 기대합니다. 각 샘플은 일반적으로 `image`와 `conversations` 필드를 포함합니다. `IMAGE_FOLDER`에는 JSON의 `image` 값과 매칭되는 이미지 파일이 있어야 합니다.

작게 먼저 돌려보고 싶으면 `DATA_FRACTION`을 낮추세요.

```bash
DATA_FRACTION=0.01 DATA_SEED=42 bash scripts/v1_5/finetune_vision_pruner.sh
```

## 4. 실행 방법

기본 실행:

```bash
conda activate visionpruner
cd /home/seokhun/VisionPruner
bash scripts/v1_5/finetune_vision_pruner.sh
```

출력 경로를 바꾸려면:

```bash
OUTPUT_DIR=./checkpoints/my-vision-pruner \
bash scripts/v1_5/finetune_vision_pruner.sh
```

학습률, 데이터 일부, 이미지 폴더를 지정하려면:

```bash
DATA_PATH=/path/to/llava_train.json \
IMAGE_FOLDER=/path/to/images \
DATA_FRACTION=0.2 \
LEARNING_RATE=2e-5 \
OUTPUT_DIR=./checkpoints/vp-qk-layer0-layer9 \
bash scripts/v1_5/finetune_vision_pruner.sh
```

기본 스크립트는 GPU 2장을 사용합니다.

```bash
deepspeed --num_gpus 2 llava/train/train_vision_pruner.py ...
```

GPU 수를 바꾸려면 `finetune_vision_pruner.sh`의 `--num_gpus`, `--per_device_train_batch_size`, `--gradient_accumulation_steps`를 함께 조정하세요.

## 5. 주요 설정

현재 `finetune_vision_pruner.sh`의 VisionPruner 설정:

- `--vision_pruner_value_layer_idx 0`: LLaVA decoder layer 0에서 fixed value path를 복사합니다.
- `--vision_pruner_context_layer_idx 9`: LLaVA decoder layer 9를 fixed context layer로 복사합니다.
- 학습되는 파라미터는 VisionPruner의 `text_q_proj`와 `image_k_proj`뿐입니다.
- score cutoff는 `0.0`, sparsity target `rho`는 `0.1`, sparsity weight는 `1.0`입니다.
- score noise와 텍스트 토큰 pruning은 사용하지 않습니다.

일반 학습 설정:

- `--model_name_or_path liuhaotian/llava-v1.5-7b`
- `--vision_tower openai/clip-vit-large-patch14-336`
- `--bf16 True`
- `--model_max_length 2048`
- `--gradient_checkpointing True`
- `--report_to wandb`

WandB를 쓰지 않으려면 스크립트 마지막의 `--report_to wandb`를 `--report_to none`으로 바꾸세요.

## 6. 저장되는 결과

`OUTPUT_DIR`에는 VisionPruner adapter 중심의 파일이 저장됩니다.

- `vision_pruner.bin`: 학습 가능한 Q/K scorer 가중치
- `config.json`: base model, vision tower, VisionPruner layer index 및 loss 설정 metadata
- tokenizer 관련 파일
- `vision_pruner_delta_stats.json`: 초기 Q/K scorer 대비 변화량 요약

현재 checkpoint는 base LLaVA 전체 weight를 포함하지 않습니다. 평가나 로드 시 base model 경로가 필요할 수 있습니다.

## 7. 자주 나는 문제

`./scripts/zero2.json`을 찾을 수 없음

- 이 README의 예시 JSON을 `scripts/zero2.json`으로 저장하세요.

CUDA out of memory

- `--per_device_train_batch_size`를 줄이세요.
- `--gradient_accumulation_steps`를 늘려 effective batch size를 유지하세요.
- `DATA_FRACTION`으로 먼저 작은 학습을 테스트하세요.

bf16 관련 오류

- GPU가 bf16을 지원하지 않는 경우 `--bf16 False --fp16 True`로 바꾸세요.

WandB 로그인 오류

- `wandb login`을 실행하거나 `--report_to none`으로 바꾸세요.
