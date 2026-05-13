import json
import os
from typing import Tuple


def add_model_loading_args(parser):
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--use-flash-attn", action="store_true")


def add_vision_pruner_args(parser):
    parser.add_argument(
        "--use-vision-pruner",
        action="store_true",
        help="Load LlavaLlamaForCausalLM_with_VisionPruner instead of the base LLaVA class.",
    )
    parser.add_argument(
        "--vision-pruner-top-p",
        type=float,
        default=None,
        help="Fraction of image tokens to keep during VisionPruner inference.",
    )
    parser.add_argument(
        "--vision-pruner-token-budget",
        type=int,
        default=None,
        help="Image token budget to keep. For LLaVA-v1.5, 128/64/32 correspond to Table 1.",
    )
    parser.add_argument(
        "--vision-pruner-num-image-tokens",
        type=int,
        default=576,
        help="Reference image-token count used to convert --vision-pruner-token-budget to a keep ratio.",
    )
    parser.add_argument(
        "--vision-pruner-verbose",
        action="store_true",
        help="Print per-sample VisionPruner score diagnostics.",
    )
    parser.add_argument(
        "--vision-pruner-debug-weights",
        action="store_true",
        help="Print checkpoint/live VisionPruner weight fingerprints after loading.",
    )
    parser.add_argument(
        "--vision-pruner-debug-weights-only",
        action="store_true",
        help="Print VisionPruner weight fingerprints and exit before evaluation.",
    )
    parser.add_argument(
        "--vision-pruner-debug-topk",
        action="store_true",
        help="Print compact fingerprints for the image tokens selected by VisionPruner.",
    )
    parser.add_argument(
        "--vision-pruner-debug-topk-limit",
        type=int,
        default=5,
        help="Maximum number of samples to print for --vision-pruner-debug-topk.",
    )
    parser.add_argument(
        "--use-attention-score-pruning",
        action="store_true",
        help="Prune image tokens by CLIP ViT [CLS] -> patch attention scores.",
    )
    parser.add_argument(
        "--attention-score-pruning-top-p",
        type=float,
        default=None,
        help="Fraction of image tokens to keep for ViT attention-score pruning.",
    )
    parser.add_argument(
        "--attention-score-pruning-token-budget",
        type=int,
        default=None,
        help="Image token budget to keep for ViT attention-score pruning.",
    )
    parser.add_argument(
        "--attention-score-pruning-num-image-tokens",
        type=int,
        default=576,
        help="Reference image-token count used to convert token budget to keep ratio.",
    )
    parser.add_argument(
        "--attention-score-pruning-layer",
        type=int,
        default=None,
        help="CLIP ViT layer used for [CLS] attention scores. Defaults to mm_vision_select_layer.",
    )
    parser.add_argument(
        "--attention-score-pruning-head-reduction",
        type=str,
        default="mean",
        choices=("mean", "sum"),
        help="How to reduce ViT attention heads before top-k selection.",
    )
    parser.add_argument(
        "--attention-score-pruning-debug-topk",
        action="store_true",
        help="Print compact fingerprints for tokens selected by attention-score pruning.",
    )
    parser.add_argument(
        "--attention-score-pruning-debug-topk-limit",
        type=int,
        default=5,
        help="Maximum number of samples to print for --attention-score-pruning-debug-topk.",
    )


def _local_config(model_path):
    expanded = os.path.expanduser(model_path)
    config_path = os.path.join(expanded, "config.json")
    if not os.path.isfile(config_path):
        return {}
    with open(config_path, "r") as f:
        return json.load(f)


def should_use_vision_pruner(args, model_path):
    if getattr(args, "use_attention_score_pruning", False):
        return False

    if getattr(args, "use_vision_pruner", False):
        return True

    expanded = os.path.expanduser(model_path)
    if not os.path.isdir(expanded):
        return False

    if os.path.exists(os.path.join(expanded, "vision_pruner.bin")):
        return True

    config = _local_config(expanded)
    return (
        config.get("model_type") == "llava_llama_with_vision_pruner"
        or "vision_pruner_decoder_layer_idx" in config
    )


def resolve_vision_pruner_keep_ratio(args):
    token_budget = getattr(args, "vision_pruner_token_budget", None)
    keep_ratio = getattr(args, "vision_pruner_top_p", None)

    if token_budget is not None:
        if token_budget <= 0:
            raise ValueError("--vision-pruner-token-budget must be positive.")
        num_image_tokens = getattr(args, "vision_pruner_num_image_tokens", 576)
        if num_image_tokens <= 0:
            raise ValueError("--vision-pruner-num-image-tokens must be positive.")
        keep_ratio = token_budget / float(num_image_tokens)

    if keep_ratio is None:
        keep_ratio = 1.0

    if keep_ratio <= 0 or keep_ratio > 1:
        raise ValueError(
            f"VisionPruner keep ratio must be in (0, 1], got {keep_ratio}."
        )

    return keep_ratio


def resolve_attention_score_pruning_keep_ratio(args):
    token_budget = getattr(args, "attention_score_pruning_token_budget", None)
    keep_ratio = getattr(args, "attention_score_pruning_top_p", None)

    if token_budget is not None:
        if token_budget <= 0:
            raise ValueError("--attention-score-pruning-token-budget must be positive.")
        num_image_tokens = getattr(args, "attention_score_pruning_num_image_tokens", 576)
        if num_image_tokens <= 0:
            raise ValueError("--attention-score-pruning-num-image-tokens must be positive.")
        keep_ratio = token_budget / float(num_image_tokens)

    if keep_ratio is None:
        keep_ratio = 1.0

    if keep_ratio <= 0 or keep_ratio > 1:
        raise ValueError(
            f"Attention-score pruning keep ratio must be in (0, 1], got {keep_ratio}."
        )

    return keep_ratio


def _sha256_file(path):
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_vision_pruner_key(key):
    if "vision_pruner." in key:
        return key.split("vision_pruner.", 1)[1]
    return key


def _hash_tensor_state(state_dict):
    import hashlib
    import torch

    digest = hashlib.sha256()
    for key in sorted(state_dict):
        tensor = state_dict[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        try:
            raw = tensor.view(torch.uint8).numpy().tobytes()
        except RuntimeError:
            raw = tensor.float().numpy().tobytes()
        digest.update(raw)
    return digest.hexdigest()


def debug_vision_pruner_weights(model, model_path):
    import torch

    expanded = os.path.expanduser(model_path)
    vision_pruner_bin = os.path.join(expanded, "vision_pruner.bin")
    model_class = type(model).__name__
    print(f"[VisionPruner debug] model_class={model_class}")
    print(f"[VisionPruner debug] checkpoint={expanded}")

    if not os.path.exists(vision_pruner_bin):
        print(f"[VisionPruner debug] no vision_pruner.bin at {vision_pruner_bin}")
        return

    print(
        "[VisionPruner debug] "
        f"vision_pruner.bin sha256={_sha256_file(vision_pruner_bin)}"
    )

    get_model = getattr(model, "get_model", None)
    inner_model = get_model() if get_model is not None else model
    get_vision_pruner = getattr(inner_model, "get_vision_pruner", None)
    vision_pruner = get_vision_pruner() if get_vision_pruner is not None else None
    if vision_pruner is None:
        print("[VisionPruner debug] live vision_pruner=None")
        return

    source_state = torch.load(vision_pruner_bin, map_location="cpu")
    source_state = {
        _normalize_vision_pruner_key(k): v
        for k, v in source_state.items()
        if "vision_pruner" in k or k.startswith("layer.")
    }
    live_state = {
        _normalize_vision_pruner_key(k): v.detach()
        for k, v in vision_pruner.state_dict().items()
    }

    source_keys = set(source_state)
    live_keys = set(live_state)
    common_keys = sorted(source_keys & live_keys)
    missing_keys = sorted(source_keys - live_keys)
    unexpected_keys = sorted(live_keys - source_keys)
    shape_mismatch = [
        key
        for key in common_keys
        if tuple(source_state[key].shape) != tuple(live_state[key].shape)
    ]

    max_abs_diff = 0.0
    abs_diff_sum = 0.0
    abs_diff_count = 0
    for key in common_keys:
        if key in shape_mismatch:
            continue
        source_tensor = source_state[key].to(dtype=live_state[key].dtype)
        live_tensor = live_state[key].detach().cpu()
        diff = (live_tensor.float() - source_tensor.float()).abs()
        max_abs_diff = max(max_abs_diff, float(diff.max().item()))
        abs_diff_sum += float(diff.sum().item())
        abs_diff_count += diff.numel()

    mean_abs_diff = abs_diff_sum / abs_diff_count if abs_diff_count else 0.0
    source_dtypes = sorted({str(v.dtype) for v in source_state.values()})
    live_dtypes = sorted({str(v.dtype) for v in live_state.values()})

    print(
        "[VisionPruner debug] "
        f"live_pruner_class={type(vision_pruner).__name__} "
        f"tensors={len(live_state)} sha256={_hash_tensor_state(live_state)}"
    )
    print(
        "[VisionPruner debug] "
        f"matched={len(common_keys)} missing={len(missing_keys)} "
        f"unexpected={len(unexpected_keys)} shape_mismatch={len(shape_mismatch)}"
    )
    print(
        "[VisionPruner debug] "
        f"source_dtypes={source_dtypes} live_dtypes={live_dtypes} "
        f"max_abs_diff_after_dtype_cast={max_abs_diff:.8g} "
        f"mean_abs_diff_after_dtype_cast={mean_abs_diff:.8g}"
    )
    if missing_keys:
        print(f"[VisionPruner debug] missing_keys={missing_keys[:5]}")
    if unexpected_keys:
        print(f"[VisionPruner debug] unexpected_keys={unexpected_keys[:5]}")
    if shape_mismatch:
        print(f"[VisionPruner debug] shape_mismatch={shape_mismatch[:5]}")


def load_eval_model(args, model_path, model_name) -> Tuple[object, object, object, int]:
    from llava_lp.model.builder import (
        load_pretrained_model,
        load_pretrained_model_with_VisionPruner,
    )

    use_vision_pruner = should_use_vision_pruner(args, model_path)
    use_attention_score_pruning = getattr(args, "use_attention_score_pruning", False)
    args._use_vision_pruner = use_vision_pruner
    args._use_attention_score_pruning = use_attention_score_pruning
    args._vision_pruner_keep_ratio = resolve_vision_pruner_keep_ratio(args)
    args._attention_score_pruning_keep_ratio = resolve_attention_score_pruning_keep_ratio(args)

    if use_attention_score_pruning:
        os.environ["LLAVA_ATTENTION_SCORE_PRUNING"] = "1"
        os.environ["LLAVA_ATTENTION_SCORE_PRUNING_TRACE_TOPK"] = (
            "1" if getattr(args, "attention_score_pruning_debug_topk", False) else "0"
        )
        os.environ["LLAVA_ATTENTION_SCORE_PRUNING_TRACE_TOPK_LIMIT"] = str(
            getattr(args, "attention_score_pruning_debug_topk_limit", 5)
        )
        loader = load_pretrained_model
    elif use_vision_pruner:
        os.environ["LLAVA_VISION_PRUNER_VERBOSE"] = (
            "1" if getattr(args, "vision_pruner_verbose", False) else "0"
        )
        os.environ["LLAVA_VISION_PRUNER_TRACE_TOPK"] = (
            "1" if getattr(args, "vision_pruner_debug_topk", False) else "0"
        )
        os.environ["LLAVA_VISION_PRUNER_TRACE_TOPK_LIMIT"] = str(
            getattr(args, "vision_pruner_debug_topk_limit", 5)
        )
        loader = load_pretrained_model_with_VisionPruner
    else:
        loader = load_pretrained_model

    loaded = loader(
        model_path,
        args.model_base,
        model_name,
        load_8bit=getattr(args, "load_8bit", False),
        load_4bit=getattr(args, "load_4bit", False),
        device_map=getattr(args, "device_map", "auto"),
        device=getattr(args, "device", "cuda"),
        use_flash_attn=getattr(args, "use_flash_attn", False),
    )
    if use_vision_pruner and (
        getattr(args, "vision_pruner_debug_weights", False)
        or getattr(args, "vision_pruner_debug_weights_only", False)
    ):
        debug_vision_pruner_weights(loaded[1], model_path)
    return loaded


def build_generation_kwargs(args, max_new_tokens):
    kwargs = {
        "do_sample": True if args.temperature > 0 else False,
        "temperature": args.temperature,
        "num_beams": getattr(args, "num_beams", 1),
        "max_new_tokens": max_new_tokens,
        "use_cache": True,
    }

    if getattr(args, "_use_attention_score_pruning", False):
        kwargs["attention_score_pruning"] = True
        kwargs["attention_score_pruning_top_p"] = getattr(
            args, "_attention_score_pruning_keep_ratio", 1.0
        )
        kwargs["attention_score_pruning_layer"] = getattr(
            args, "attention_score_pruning_layer", None
        )
        kwargs["attention_score_pruning_head_reduction"] = getattr(
            args, "attention_score_pruning_head_reduction", "mean"
        )
    elif getattr(args, "_use_vision_pruner", False):
        kwargs["top_p"] = getattr(args, "_vision_pruner_keep_ratio", 1.0)
    elif getattr(args, "top_p", None) is not None:
        kwargs["top_p"] = args.top_p

    return kwargs
