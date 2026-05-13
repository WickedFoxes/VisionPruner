import hashlib
import os
from typing import List, Optional

import torch

from .multimodal_pruner.pruner import get_image_token_ranges, prune_tokens_for_inference


def _resolve_layer_index(layer_index: int, num_layers: int) -> int:
    resolved = layer_index if layer_index >= 0 else num_layers + layer_index
    if resolved < 0 or resolved >= num_layers:
        raise ValueError(
            f"ViT attention layer index {layer_index} is out of range for {num_layers} layers."
        )
    return resolved


def _flatten_images(images: torch.Tensor) -> torch.Tensor:
    if isinstance(images, list):
        flattened = []
        for image in images:
            flattened.append(image.unsqueeze(0) if image.ndim == 3 else image)
        return torch.cat(flattened, dim=0)

    if images.ndim == 3:
        return images.unsqueeze(0)
    if images.ndim == 4:
        return images
    if images.ndim == 5:
        return images.flatten(0, 1)

    raise ValueError(f"Expected image tensor/list with 3D, 4D, or 5D images, got {tuple(images.shape)}.")


def _reduce_heads(cls_to_patch_attention: torch.Tensor, head_reduction: str) -> torch.Tensor:
    if head_reduction == "mean":
        return cls_to_patch_attention.mean(dim=1)
    if head_reduction == "sum":
        return cls_to_patch_attention.sum(dim=1)
    raise ValueError(f"Unsupported attention head reduction: {head_reduction}.")


def compute_vit_cls_attention_scores(
    model,
    images,
    select_layer: Optional[int] = None,
    head_reduction: str = "mean",
    cls_token_idx: int = 0,
) -> torch.Tensor:
    """Return per-image CLIP ViT [CLS] -> patch attention scores."""
    vision_tower = model.get_vision_tower()
    if vision_tower is None:
        raise RuntimeError("Cannot run attention-score pruning without a vision tower.")
    if not getattr(vision_tower, "is_loaded", False):
        vision_tower.load_model()

    raw_vision_model = getattr(vision_tower, "vision_tower", None)
    if raw_vision_model is None:
        raise RuntimeError("The current vision tower does not expose the raw CLIP vision model.")

    flat_images = _flatten_images(images)
    flat_images = flat_images.to(device=vision_tower.device, dtype=vision_tower.dtype)

    outputs = raw_vision_model(
        flat_images,
        output_hidden_states=True,
        output_attentions=True,
        return_dict=True,
    )
    attentions = outputs.attentions
    if attentions is None or any(attn is None for attn in attentions):
        raise RuntimeError(
            "Vision tower did not return attentions. Load the CLIP vision tower with eager attention."
        )

    layer = getattr(model.config, "mm_vision_select_layer", -2) if select_layer is None else select_layer
    layer_idx = _resolve_layer_index(int(layer), len(attentions))
    selected_attention = attentions[layer_idx]

    # selected_attention: [num_images, num_heads, 1 + num_patches, 1 + num_patches]
    cls_to_patch = selected_attention[:, :, cls_token_idx, cls_token_idx + 1 :].float()
    return _reduce_heads(cls_to_patch, head_reduction)


def _align_scores_to_length(scores: torch.Tensor, target_len: int) -> torch.Tensor:
    if scores.numel() == target_len:
        return scores
    if scores.numel() > target_len:
        return scores[:target_len]
    pad_len = target_len - scores.numel()
    return torch.cat([scores, scores.new_zeros(pad_len)], dim=0)


def _scores_by_batch_image_ranges(
    scores_by_image: torch.Tensor,
    batch_image_ranges: List[List[tuple]],
    device: torch.device,
) -> List[torch.Tensor]:
    score_rows = scores_by_image.to(device=device)
    score_row_idx = 0
    image_scores_list = []

    for ranges in batch_image_ranges:
        per_sample_scores = []
        for start, end in ranges:
            target_len = int(end - start)
            if score_row_idx >= score_rows.shape[0]:
                per_sample_scores.append(torch.zeros(target_len, device=device, dtype=torch.float32))
            else:
                per_sample_scores.append(_align_scores_to_length(score_rows[score_row_idx], target_len))
            score_row_idx += 1

        if per_sample_scores:
            image_scores_list.append(torch.cat(per_sample_scores, dim=0).unsqueeze(0))
        else:
            image_scores_list.append(torch.empty((1, 0), device=device, dtype=torch.float32))

    return image_scores_list


def _digest_indices(indices: torch.Tensor) -> str:
    indices = indices.detach().to(torch.int64).cpu().contiguous()
    return hashlib.sha256(indices.numpy().tobytes()).hexdigest()[:16]


def _maybe_trace_scores(image_scores_list, keep_ratio: float):
    if os.environ.get("LLAVA_ATTENTION_SCORE_PRUNING_TRACE_TOPK", "0") != "1":
        return

    trace_limit = int(os.environ.get("LLAVA_ATTENTION_SCORE_PRUNING_TRACE_TOPK_LIMIT", "5"))
    if not hasattr(_maybe_trace_scores, "_trace_count"):
        _maybe_trace_scores._trace_count = 0

    for batch_idx, score_tensor in enumerate(image_scores_list):
        if _maybe_trace_scores._trace_count >= trace_limit:
            break
        scores = score_tensor.squeeze(0)
        num_tokens = int(scores.numel())
        if num_tokens == 0:
            continue
        keep_count = max(1, min(num_tokens, int(num_tokens * float(keep_ratio))))
        topk_indices = torch.topk(scores, k=keep_count).indices.sort().values
        selected_scores = scores[topk_indices]
        score_std = scores.float().std(unbiased=False).item() if num_tokens > 1 else 0.0
        print(
            "[AttentionScorePruning topk] "
            f"sample={_maybe_trace_scores._trace_count} batch={batch_idx} "
            f"image_tokens={num_tokens} keep={keep_count} keep_ratio={float(keep_ratio):.6f} "
            f"rel_sha={_digest_indices(topk_indices)} "
            f"score_min={scores.min().item():.6f} "
            f"score_max={scores.max().item():.6f} "
            f"score_mean={scores.float().mean().item():.6f} "
            f"score_std={score_std:.6f} "
            f"selected_score_min={selected_scores.min().item():.6f} "
            f"selected_score_max={selected_scores.max().item():.6f} "
            f"first10_rel={topk_indices[:10].tolist()}"
        )
        _maybe_trace_scores._trace_count += 1


def apply_vit_attention_score_pruning(
    model,
    orig_input_ids: torch.Tensor,
    orig_attention_mask: Optional[torch.Tensor],
    images,
    inputs_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    position_ids: Optional[torch.Tensor],
    keep_ratio: float,
    select_layer: Optional[int] = None,
    head_reduction: str = "mean",
):
    scores_by_image = compute_vit_cls_attention_scores(
        model,
        images,
        select_layer=select_layer,
        head_reduction=head_reduction,
    )
    batch_image_ranges = get_image_token_ranges(
        orig_input_ids,
        inputs_embeds,
        orig_attention_mask=orig_attention_mask,
        exp_attention_mask=attention_mask,
    )
    image_scores_list = _scores_by_batch_image_ranges(
        scores_by_image,
        batch_image_ranges,
        device=inputs_embeds.device,
    )
    _maybe_trace_scores(image_scores_list, keep_ratio)
    return prune_tokens_for_inference(
        image_scores_list,
        batch_image_ranges,
        inputs_embeds,
        attention_mask,
        position_ids,
        top_p=keep_ratio,
    )
