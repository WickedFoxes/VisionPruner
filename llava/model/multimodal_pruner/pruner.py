import copy
import hashlib
import inspect
import os
import torch
import torch.nn as nn
from llava.constants import IMAGE_TOKEN_INDEX, IGNORE_INDEX
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask


class LlavaImagePruner(nn.Module):
    def __init__(self, value_layer, context_layer, rotary_emb=None):
        super().__init__()
        value_attn = value_layer.self_attn
        context_attn = context_layer.self_attn

        self.value_layernorm = copy.deepcopy(value_layer.input_layernorm)
        self.value_v_proj = copy.deepcopy(value_attn.v_proj)
        self.value_o_proj = copy.deepcopy(value_attn.o_proj)
        self.rotary_emb = copy.deepcopy(
            rotary_emb if rotary_emb is not None else getattr(value_attn, "rotary_emb", None)
        )
        self.context_layer = copy.deepcopy(context_layer)
        self.text_q_proj = copy.deepcopy(context_attn.q_proj)
        self.image_k_proj = copy.deepcopy(context_attn.k_proj)
        self.verbose = os.environ.get("LLAVA_VISION_PRUNER_VERBOSE", "1") == "1"

        config = getattr(context_attn, "config", None)
        hidden_size = getattr(config, "hidden_size", context_attn.q_proj.in_features)
        self.num_attention_heads = int(
            getattr(
                context_attn,
                "num_heads",
                getattr(config, "num_attention_heads", 1),
            )
        )
        self.num_key_value_heads = int(
            getattr(
                context_attn,
                "num_key_value_heads",
                getattr(config, "num_key_value_heads", self.num_attention_heads),
            )
        )
        self.head_dim = int(
            getattr(
                context_attn,
                "head_dim",
                context_attn.q_proj.out_features // max(1, self.num_attention_heads),
            )
        )
        self.num_key_value_groups = max(1, self.num_attention_heads // max(1, self.num_key_value_heads))
        self.hidden_size = int(hidden_size)

        self._freeze_fixed_modules()

    def _freeze_fixed_modules(self):
        for module in (
            self.value_layernorm,
            self.value_v_proj,
            self.value_o_proj,
            self.rotary_emb,
            self.context_layer,
        ):
            if module is None:
                continue
            for param in module.parameters():
                param.requires_grad_(False)

        for param in self.text_q_proj.parameters():
            param.requires_grad_(True)
        for param in self.image_k_proj.parameters():
            param.requires_grad_(True)

    @staticmethod
    def _stable_l2_normalize(tensor: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
        tensor = tensor.float()
        scale = tensor.detach().abs().amax(dim=dim, keepdim=True).clamp_min(eps)
        scaled = tensor / scale
        inv_norm = torch.rsqrt(scaled.pow(2).sum(dim=dim, keepdim=True).clamp_min(eps * eps))
        return scaled * inv_norm

    @staticmethod
    def _rotate_half(tensor: torch.Tensor) -> torch.Tensor:
        first_half = tensor[..., : tensor.shape[-1] // 2]
        second_half = tensor[..., tensor.shape[-1] // 2 :]
        return torch.cat((-second_half, first_half), dim=-1)

    @staticmethod
    def _layer_output_hidden_states(outputs):
        if isinstance(outputs, tuple):
            return outputs[0]
        return outputs

    def _build_position_ids(
        self,
        attention_mask_2d: torch.Tensor,
        position_ids: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        if position_ids is not None:
            return position_ids
        return torch.arange(seq_len, device=attention_mask_2d.device, dtype=torch.long).unsqueeze(0).expand(
            attention_mask_2d.shape[0],
            -1,
        )

    def _build_attention_mask_4d(
        self,
        attention_mask: torch.Tensor,
        hidden_states: torch.Tensor,
        past_key_value: tuple = None,
    ) -> torch.Tensor:
        if attention_mask is None:
            return None
        if attention_mask.dim() != 2:
            return attention_mask.float()
        return _prepare_4d_causal_attention_mask(
            attention_mask,
            (hidden_states.shape[0], hidden_states.shape[1]),
            hidden_states,
            past_key_value[0].shape[-2] if past_key_value is not None else 0,
        )

    def _value_path(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.to(dtype=self.value_v_proj.weight.dtype)
        normalized = self.value_layernorm(hidden_states)
        value_states = self.value_v_proj(normalized)
        value_states = value_states.view(
            hidden_states.shape[0],
            hidden_states.shape[1],
            self.num_key_value_heads,
            self.head_dim,
        ).transpose(1, 2)
        value_states = self._repeat_kv(value_states)
        value_states = value_states.transpose(1, 2).contiguous().view(
            hidden_states.shape[0],
            hidden_states.shape[1],
            self.num_attention_heads * self.head_dim,
        )
        return self.value_o_proj(value_states)

    def _repeat_kv(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.num_key_value_groups == 1:
            return hidden_states
        batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch,
            num_key_value_heads,
            self.num_key_value_groups,
            seq_len,
            head_dim,
        )
        return hidden_states.reshape(batch, num_key_value_heads * self.num_key_value_groups, seq_len, head_dim)

    def _run_context_layer(
        self,
        hidden_states: torch.Tensor,
        attention_mask_4d: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_value: tuple = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        forward_params = inspect.signature(self.context_layer.forward).parameters
        layer_kwargs = {
            "hidden_states": hidden_states,
            "attention_mask": attention_mask_4d,
            "position_ids": position_ids,
        }

        if "past_key_values" in forward_params:
            layer_kwargs["past_key_values"] = past_key_value
        elif "past_key_value" in forward_params:
            layer_kwargs["past_key_value"] = past_key_value
        if "output_attentions" in forward_params:
            layer_kwargs["output_attentions"] = output_attentions
        if "use_cache" in forward_params:
            layer_kwargs["use_cache"] = use_cache
        if "position_embeddings" in forward_params and self.rotary_emb is not None:
            layer_kwargs["position_embeddings"] = self.rotary_emb(hidden_states, position_ids)

        for key, value in kwargs.items():
            if key in forward_params:
                layer_kwargs[key] = value

        outputs = self.context_layer(**layer_kwargs)
        return self._layer_output_hidden_states(outputs)

    def _apply_rotary(self, states: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        if self.rotary_emb is None:
            return states
        cos, sin = self.rotary_emb(states, positions)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        return (states * cos) + (self._rotate_half(states) * sin)

    def _project_query(self, token_states: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        query = self.text_q_proj(token_states)
        query = query.view(1, 1, self.num_attention_heads, self.head_dim).transpose(1, 2)
        return self._apply_rotary(query, token_positions)

    def _project_keys(self, token_states: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        keys = self.image_k_proj(token_states)
        keys = keys.view(1, token_states.shape[1], self.num_key_value_heads, self.head_dim).transpose(1, 2)
        keys = self._apply_rotary(keys, token_positions)
        return self._repeat_kv(keys)

    def forward(
        self,
        input_ids: torch.LongTensor,
        hidden_states: torch.Tensor,
        batch_image_ranges = None,
        batch_text_indices = None,
        attention_mask: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        past_key_value: tuple = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        labels: torch.LongTensor = None,
        **kwargs
    ):
        if attention_mask is None:
            attention_mask_2d = torch.ones(
                hidden_states.shape[:2], device=hidden_states.device, dtype=torch.long
            )
        else:
            attention_mask_2d = attention_mask.clone()
        position_ids = self._build_position_ids(attention_mask_2d, position_ids, hidden_states.shape[1])
        attention_mask_4d = self._build_attention_mask_4d(attention_mask, hidden_states, past_key_value)

        value_states = self._value_path(hidden_states)
        layer_output_states = self._run_context_layer(
            hidden_states=value_states,
            attention_mask_4d=attention_mask_4d,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )

        batch_size = layer_output_states.shape[0]
        seq_len = layer_output_states.shape[1]

        if labels is not None:
            last_token_indices = []
            for b in range(batch_size):
                valid_idx = torch.where(labels[b] != IGNORE_INDEX)[0]
                if len(valid_idx) > 0:
                    idx = valid_idx[0].item() - 1
                    last_token_indices.append(max(0, idx))
                else:
                    idx = attention_mask_2d[b].cumsum(dim=0).argmax().item()
                    last_token_indices.append(idx)
            last_token_indices = torch.tensor(last_token_indices, device=layer_output_states.device)
        else:
            last_token_indices = attention_mask_2d.cumsum(dim=1).argmax(dim=1)
        
        image_scores_list = []
        
        for b in range(batch_size):
            ranges = batch_image_ranges[b] if batch_image_ranges is not None else []
            if len(ranges) == 0:
                image_scores_list.append(torch.empty((1, 0), device=hidden_states.device, dtype=torch.float32))
                continue

            all_img_indices = torch.cat([
                torch.arange(start, end, device=hidden_states.device) for start, end in ranges
            ])
            all_img_indices = all_img_indices[(all_img_indices >= 0) & (all_img_indices < seq_len)]
            if all_img_indices.numel() == 0:
                image_scores_list.append(torch.empty((1, 0), device=hidden_states.device, dtype=torch.float32))
                continue

            query_index = last_token_indices[b].view(1, 1)
            query_states = layer_output_states[b:b+1, query_index.squeeze(0), :]
            image_states = layer_output_states[b:b+1, all_img_indices, :]

            query = self._project_query(query_states, position_ids[b:b+1, query_index.squeeze(0)])
            keys = self._project_keys(image_states, position_ids[b:b+1, all_img_indices])
            query = self._stable_l2_normalize(query, dim=-1, eps=1e-6)
            keys = self._stable_l2_normalize(keys, dim=-1, eps=1e-6)
            scores = (query * keys).sum(dim=-1).mean(dim=1).clamp(min=-1.0, max=1.0)
            image_scores_list.append(scores)
        
        if self.verbose:
            for b, s in enumerate(image_scores_list):
                scores = s.squeeze(0)
                n = scores.shape[0]
                if n == 0:
                    print(f"[batch {b}] image tokens: 0")
                    continue
                k = max(1, int(n * 0.1))
                # topk_idx = torch.topk(scores, k=k).indices.sort().values
                topk_idx = torch.topk(scores, k=k).indices  # 점수 내림차순 유지
                score_cutoff = 0.0
                print(f"[batch {b}] image tokens: {n}, top 10% ({k}) indices: {topk_idx.tolist()}")
                print(f"scores : {scores[topk_idx]}")
                print(f"         score range: min={scores.min().item():.4f}, max={scores.max().item():.4f}")
                print(f"         scores >= {score_cutoff}: {(scores >= score_cutoff).sum().item()} / {n}")

        return {"image": image_scores_list}


def get_image_and_text_token_indices(input_ids,
                                     hidden_states,
                                     orig_attention_mask=None,
                                     exp_attention_mask=None,
                                     labels=None,
                                     text_prune_mask=None,
                                     image_token_id=IMAGE_TOKEN_INDEX):
    """
    배치 내 각 데이터별로 이미지 토큰 범위와 질문 텍스트 토큰 인덱스를 계산합니다.
    Args:
        input_ids: [batch_size, seq_len] (이미지 패치로 확장되기 전, 패딩 포함)
        hidden_states: [batch_size, expanded_seq_len, hidden_dim] (이미지 패치로 확장된 후, 오른쪽 패딩)
        orig_attention_mask: [batch_size, seq_len] input_ids에 대응하는 원본 attention mask
        exp_attention_mask: [batch_size, expanded_seq_len] hidden_states에 대응하는 확장 후 attention mask
        labels: [batch_size, expanded_seq_len] 답변 토큰은 유효 label, 프롬프트는 IGNORE_INDEX
        text_prune_mask: [batch_size, seq_len] 원본 input_ids 기준 pruning 후보 텍스트 토큰 마스크.
            주어지면 USER:/ASSISTANT: 같은 템플릿 토큰을 제외하고 True 위치만 질문
            텍스트 후보로 사용합니다.
    Returns:
        batch_image_ranges: 배치 각 항목별 이미지 시작/끝 인덱스 튜플 리스트
        batch_text_indices: 배치 각 항목별 질문 텍스트 토큰 인덱스 텐서 리스트
    """
    batch_size = len(input_ids)
    batch_image_ranges = []
    batch_text_indices = []
    device = hidden_states.device

    for b in range(batch_size):
        # 1. 패딩 제거 후 실제 input_ids에서 이미지 토큰 위치 찾기
        # orig_attention_mask 없으면 패딩이 없다고 가정 (단일 샘플 등)
        if orig_attention_mask is not None:
            orig_unpad_mask = orig_attention_mask[b].bool()
            unpadded_ids = input_ids[b][orig_unpad_mask]
            if text_prune_mask is not None:
                unpadded_text_prune_mask = text_prune_mask[b].to(device=device).bool()[orig_unpad_mask]
            else:
                unpadded_text_prune_mask = None
        else:
            unpadded_ids = input_ids[b]
            if text_prune_mask is not None:
                unpadded_text_prune_mask = text_prune_mask[b].to(device=device).bool()
            else:
                unpadded_text_prune_mask = None

        image_positions = torch.where(unpadded_ids == image_token_id)[0]

        # 2. 이미지 패치 개수 계산
        # hidden_states는 오른쪽 패딩이므로 exp_attention_mask로 실제 확장 길이 구함
        num_image_tokens = len(image_positions)
        num_text_tokens = len(unpadded_ids) - num_image_tokens

        if exp_attention_mask is not None:
            actual_expanded_indices = torch.where(exp_attention_mask[b].bool())[0]
        else:
            actual_expanded_indices = torch.arange(hidden_states.shape[1], device=device)

        if actual_expanded_indices.numel() == 0:
            batch_image_ranges.append([])
            batch_text_indices.append(torch.empty(0, device=device, dtype=torch.long))
            continue

        actual_expanded_len = int(actual_expanded_indices.numel())
        expanded_offset = int(actual_expanded_indices[0].item())
        expanded_limit = min(expanded_offset + actual_expanded_len, hidden_states.shape[1])

        if num_image_tokens > 0:
            total_patch_embeddings = actual_expanded_len - num_text_tokens
            patches_per_image = total_patch_embeddings // num_image_tokens
        else:
            patches_per_image = 1

        ranges = []
        for i, pos in enumerate(image_positions):
            offset = i * (patches_per_image - 1)
            start_idx = expanded_offset + pos.item() + offset
            end_idx = min(start_idx + patches_per_image, expanded_limit)
            if start_idx < end_idx:
                ranges.append((start_idx, end_idx))

        text_indices = []
        seen_images = 0
        for pos, token_id in enumerate(unpadded_ids.tolist()):
            if token_id == image_token_id:
                seen_images += 1
                continue
            if (
                unpadded_text_prune_mask is not None
                and (
                    pos >= unpadded_text_prune_mask.numel()
                    or not unpadded_text_prune_mask[pos].item()
                )
            ):
                continue
            expanded_idx = expanded_offset + pos + seen_images * (patches_per_image - 1)
            if expanded_idx < expanded_limit:
                text_indices.append(expanded_idx)

        text_indices = torch.tensor(text_indices, device=device, dtype=torch.long)

        # 학습에서는 답변 시작 직전 토큰을 score query로 사용하므로, 그 이전의 텍스트만
        # 질문 텍스트 후보로 둔다. 추론에서는 현재 프롬프트의 마지막 토큰을 query로 쓴다.
        if labels is not None:
            valid_idx = torch.where(labels[b] != IGNORE_INDEX)[0]
            if len(valid_idx) > 0:
                last_prompt_idx = max(0, valid_idx[0].item() - 1)
            else:
                last_prompt_idx = expanded_limit - 1
            if text_indices.numel() > 0:
                text_indices = text_indices[
                    (text_indices < last_prompt_idx)
                    & (labels[b, text_indices] == IGNORE_INDEX)
                ]
        elif text_indices.numel() > 0:
            text_indices = text_indices[text_indices < (expanded_limit - 1)]

        batch_image_ranges.append(ranges)
        batch_text_indices.append(text_indices)

    return batch_image_ranges, batch_text_indices


def get_image_token_ranges(input_ids,
                           hidden_states,
                           orig_attention_mask=None,
                           exp_attention_mask=None,
                           image_token_id=IMAGE_TOKEN_INDEX):
    batch_image_ranges, _ = get_image_and_text_token_indices(
        input_ids,
        hidden_states,
        orig_attention_mask=orig_attention_mask,
        exp_attention_mask=exp_attention_mask,
        image_token_id=image_token_id,
    )
    return batch_image_ranges


def get_image_slot_counts(input_ids, orig_attention_mask=None, image_token_id=IMAGE_TOKEN_INDEX):
    """Return the number of vision-tower rows consumed by each batch item."""
    counts = []
    batch_size = len(input_ids)
    for b in range(batch_size):
        if orig_attention_mask is not None:
            unpadded_ids = input_ids[b][orig_attention_mask[b].bool()]
        else:
            unpadded_ids = input_ids[b]
        num_images = int((unpadded_ids == image_token_id).sum().item())
        counts.append(max(1, num_images))
    return counts


def _validate_keep_ratio(keep_ratio: float) -> float:
    keep_ratio = float(keep_ratio)
    if keep_ratio <= 0.0 or keep_ratio > 1.0:
        raise ValueError(f"keep ratio must be in (0, 1], got {keep_ratio}.")
    return keep_ratio


def _get_keep_count(num_tokens: int, keep_ratio: float) -> int:
    keep_ratio = _validate_keep_ratio(keep_ratio)
    return max(1, min(num_tokens, int(num_tokens * keep_ratio)))


def _align_scores_to_length(scores: torch.Tensor, target_len: int) -> torch.Tensor:
    if scores.numel() == target_len:
        return scores
    if scores.numel() > target_len:
        return scores[:target_len]
    pad_len = target_len - scores.numel()
    return torch.cat([scores, scores.new_zeros(pad_len)], dim=0)


def _digest_indices(indices: torch.Tensor) -> str:
    indices = indices.detach().to(torch.int64).cpu().contiguous()
    return hashlib.sha256(indices.numpy().tobytes()).hexdigest()[:16]


def scores_by_image_to_batch_token_scores(
    scores_by_image: torch.Tensor,
    batch_image_ranges,
    device: torch.device,
    image_slot_counts=None,
):
    """Map one ViT score row per image to one concatenated image-token score tensor per batch item."""
    score_rows = scores_by_image.to(device=device)
    score_row_idx = 0
    image_scores_list = []

    for batch_idx, ranges in enumerate(batch_image_ranges):
        per_sample_scores = []
        for image_idx, (start, end) in enumerate(ranges):
            target_len = int(end - start)
            row_idx = score_row_idx + image_idx
            if row_idx >= score_rows.shape[0]:
                per_sample_scores.append(torch.zeros(target_len, device=device, dtype=torch.float32))
            else:
                per_sample_scores.append(_align_scores_to_length(score_rows[row_idx], target_len))

        rows_to_consume = len(ranges)
        if image_slot_counts is not None and batch_idx < len(image_slot_counts):
            rows_to_consume = max(rows_to_consume, int(image_slot_counts[batch_idx]))
        score_row_idx += rows_to_consume

        if per_sample_scores:
            image_scores_list.append(torch.cat(per_sample_scores, dim=0).unsqueeze(0))
        else:
            image_scores_list.append(torch.empty((1, 0), device=device, dtype=torch.float32))

    return image_scores_list


def prune_image_tokens_by_score(
    image_scores_list,
    batch_image_ranges,
    inputs_embeds,
    attention_mask,
    position_ids,
    top_p=0.75,
    labels=None,
    batch_text_indices=None,
    return_image_token_indices=False,
):
    """
    Hard-prune image tokens by score and return updated token ranges.

    This is used as the fixed ViT [CLS]-attention pre-pruning stage before
    VisionPruner scores the remaining image tokens.
    """
    batch_size = inputs_embeds.shape[0]
    device = inputs_embeds.device
    _validate_keep_ratio(top_p)

    new_embeds_list = []
    new_labels_list = [] if labels is not None else None
    new_masks_list = [] if attention_mask is not None else None
    new_pos_ids_list = [] if position_ids is not None else None
    new_batch_image_ranges = []
    new_batch_text_indices = [] if batch_text_indices is not None else None
    new_batch_image_token_indices = [] if return_image_token_indices else None
    trace_pre_prune = (
        os.environ.get("LLAVA_VISION_PRUNER_TRACE_PRE_PRUNE", "0") == "1"
        or os.environ.get("LLAVA_ATTENTION_SCORE_PRUNING_TRACE_TOPK", "0") == "1"
    )
    trace_limit = int(os.environ.get(
        "LLAVA_VISION_PRUNER_TRACE_PRE_PRUNE_LIMIT",
        os.environ.get("LLAVA_ATTENTION_SCORE_PRUNING_TRACE_TOPK_LIMIT", "5"),
    ))
    if not hasattr(prune_image_tokens_by_score, "_trace_count"):
        prune_image_tokens_by_score._trace_count = 0

    for b in range(batch_size):
        total_seq_len = inputs_embeds.shape[1]
        keep_mask = torch.ones(total_seq_len, dtype=torch.bool, device=device)
        ranges = batch_image_ranges[b] if batch_image_ranges is not None else []

        if b < len(image_scores_list):
            image_scores = image_scores_list[b].squeeze(0)
        else:
            image_scores = torch.empty(0, device=device, dtype=torch.float32)

        if ranges and image_scores.numel() > 0:
            all_img_indices = torch.cat([
                torch.arange(start, end, device=device) for start, end in ranges
            ])
            token_count = min(int(image_scores.numel()), int(all_img_indices.numel()))
            if token_count > 0:
                image_scores = image_scores[:token_count]
                all_img_indices = all_img_indices[:token_count]
                k = _get_keep_count(token_count, top_p)
                topk_indices = torch.topk(image_scores, k=k).indices.sort().values
                selected_img_indices = all_img_indices[topk_indices]

                if trace_pre_prune and prune_image_tokens_by_score._trace_count < trace_limit:
                    selected_scores = image_scores[topk_indices]
                    score_std = image_scores.float().std(unbiased=False).item() if token_count > 1 else 0.0
                    print(
                        "[VisionPruner attention pre-prune] "
                        f"sample={prune_image_tokens_by_score._trace_count} batch={b} "
                        f"image_tokens={token_count} keep={k} keep_ratio={float(top_p):.6f} "
                        f"rel_sha={_digest_indices(topk_indices)} "
                        f"abs_seq_sha={_digest_indices(selected_img_indices)} "
                        f"score_min={image_scores.min().item():.6f} "
                        f"score_max={image_scores.max().item():.6f} "
                        f"score_mean={image_scores.float().mean().item():.6f} "
                        f"score_std={score_std:.6f} "
                        f"selected_score_min={selected_scores.min().item():.6f} "
                        f"selected_score_max={selected_scores.max().item():.6f} "
                        f"first10_rel={topk_indices[:10].tolist()}"
                    )
                    prune_image_tokens_by_score._trace_count += 1

                keep_mask[all_img_indices] = False
                keep_mask[selected_img_indices] = True

        final_indices = torch.where(keep_mask)[0]
        old_to_new = torch.full((total_seq_len,), -1, device=device, dtype=torch.long)
        old_to_new[final_indices] = torch.arange(final_indices.numel(), device=device)

        mapped_ranges = []
        for start, end in ranges:
            cur_indices = torch.arange(start, end, device=device)
            cur_indices = cur_indices[keep_mask[cur_indices]]
            if cur_indices.numel() == 0:
                continue
            mapped = old_to_new[cur_indices]
            mapped = mapped[mapped >= 0]
            if mapped.numel() > 0:
                mapped_ranges.append((int(mapped[0].item()), int(mapped[-1].item()) + 1))
        new_batch_image_ranges.append(mapped_ranges)

        if new_batch_text_indices is not None:
            text_indices = batch_text_indices[b].to(device=device, dtype=torch.long)
            if text_indices.numel() > 0:
                text_indices = text_indices[(text_indices >= 0) & (text_indices < total_seq_len)]
                mapped_text_indices = old_to_new[text_indices]
                mapped_text_indices = mapped_text_indices[mapped_text_indices >= 0]
            else:
                mapped_text_indices = torch.empty(0, device=device, dtype=torch.long)
            new_batch_text_indices.append(mapped_text_indices)

        if new_batch_image_token_indices is not None:
            if ranges:
                all_img_indices = torch.cat([
                    torch.arange(start, end, device=device) for start, end in ranges
                ])
                local_indices = torch.arange(all_img_indices.numel(), device=device, dtype=torch.long)
                kept_local_indices = local_indices[keep_mask[all_img_indices]]
            else:
                kept_local_indices = torch.empty(0, device=device, dtype=torch.long)
            new_batch_image_token_indices.append(kept_local_indices)

        new_embeds_list.append(inputs_embeds[b, final_indices])
        if labels is not None:
            new_labels_list.append(labels[b, final_indices])
        if attention_mask is not None:
            new_masks_list.append(attention_mask[b, final_indices])
        if position_ids is not None:
            new_pos_ids_list.append(position_ids[b, final_indices])

    max_len = max(x.shape[0] for x in new_embeds_list)

    final_embeds = []
    final_labels = [] if labels is not None else None
    final_masks = [] if attention_mask is not None else None
    final_pos_ids = [] if position_ids is not None else None

    for i in range(batch_size):
        cur_len = new_embeds_list[i].shape[0]
        pad_len = max_len - cur_len

        if pad_len > 0:
            pad_embeds = torch.zeros((pad_len, inputs_embeds.shape[-1]), device=device, dtype=inputs_embeds.dtype)
            final_embeds.append(torch.cat([new_embeds_list[i], pad_embeds], dim=0))

            if labels is not None:
                pad_labels = torch.full((pad_len,), IGNORE_INDEX, device=device, dtype=labels.dtype)
                final_labels.append(torch.cat([new_labels_list[i], pad_labels], dim=0))

            if attention_mask is not None:
                pad_masks = torch.zeros(pad_len, device=device, dtype=attention_mask.dtype)
                final_masks.append(torch.cat([new_masks_list[i], pad_masks], dim=0))

            if position_ids is not None:
                pad_pos = torch.zeros(pad_len, device=device, dtype=position_ids.dtype)
                final_pos_ids.append(torch.cat([new_pos_ids_list[i], pad_pos], dim=0))
        else:
            final_embeds.append(new_embeds_list[i])
            if labels is not None:
                final_labels.append(new_labels_list[i])
            if attention_mask is not None:
                final_masks.append(new_masks_list[i])
            if position_ids is not None:
                final_pos_ids.append(new_pos_ids_list[i])

    final_embeds = torch.stack(final_embeds)
    if labels is not None:
        final_labels = torch.stack(final_labels)
    if attention_mask is not None:
        final_masks = torch.stack(final_masks)
    if position_ids is not None:
        final_pos_ids = torch.stack(final_pos_ids)

    return (
        final_embeds,
        final_labels,
        final_masks,
        final_pos_ids,
        new_batch_image_ranges,
        new_batch_text_indices,
        new_batch_image_token_indices,
    ) if return_image_token_indices else (
        final_embeds,
        final_labels,
        final_masks,
        final_pos_ids,
        new_batch_image_ranges,
        new_batch_text_indices,
    )


def prune_tokens_for_training(
    scores,
    batch_image_ranges,
    batch_text_indices,
    inputs_embeds,
    labels,
    attention_mask,
    position_ids,
    tau=0.0
):
    """
    Applies STE-based token masking using a fixed score cutoff TAU.

    Masking rule:
        M_i = 1  if S_i > TAU,  else 0
        M_ste_i = S_i + stop_grad(M_i − S_i)   ← forward: M_i, backward: S_i

    Returns selected and complement image-token masks for the task and contrastive losses.
    """
    batch_size = inputs_embeds.shape[0]
    device = inputs_embeds.device

    # Keep the STE mask in fp32 because cosine scores are fp32 even when embeddings are bf16.
    selected_mask = torch.ones((batch_size, inputs_embeds.shape[1]), device=device, dtype=torch.float32)
    unselected_mask = torch.ones((batch_size, inputs_embeds.shape[1]), device=device, dtype=torch.float32)

    if isinstance(scores, dict):
        image_scores_list = scores.get("image", [])
    else:
        image_scores_list = scores

    all_image_ste_masks = []

    def build_ste_mask(score_tensor):
        M = (score_tensor > tau).to(score_tensor.dtype)
        return score_tensor + (M - score_tensor).detach()

    for b in range(batch_size):
        image_scores = image_scores_list[b].squeeze(0)  # [total_image_tokens]
        num_img_tokens = image_scores.shape[0]
        ranges = batch_image_ranges[b] if batch_image_ranges is not None else []
        if num_img_tokens == 0 or not ranges:
            continue

        all_img_indices = torch.cat([
            torch.arange(start, end, device=device) for start, end in ranges
        ])
        token_count = min(num_img_tokens, all_img_indices.numel())
        if token_count == 0:
            continue

        image_scores = image_scores[:token_count]
        all_img_indices = all_img_indices[:token_count]
        image_M_ste = build_ste_mask(image_scores)
        unselected_M_ste = (1.0 - image_scores) + ((1.0 - image_M_ste) - (1.0 - image_scores)).detach()
        all_image_ste_masks.append(image_M_ste)

        selected_mask[b, all_img_indices] = image_M_ste
        unselected_mask[b, all_img_indices] = unselected_M_ste

    image_ste_masks_flat = (
        torch.cat(all_image_ste_masks) if all_image_ste_masks else inputs_embeds.new_empty(0)
    )

    # Differentiable masks: pruned image tokens become zero vectors.
    # Cast only at the multiplication boundary so the downstream model keeps its input dtype.
    selected_embeds = inputs_embeds * selected_mask.to(inputs_embeds.dtype).unsqueeze(-1)
    unselected_embeds = inputs_embeds * unselected_mask.to(inputs_embeds.dtype).unsqueeze(-1)

    if attention_mask is not None:
        selected_attention_mask = attention_mask * selected_mask.to(attention_mask.dtype)
        unselected_attention_mask = attention_mask * unselected_mask.to(attention_mask.dtype)
    else:
        selected_attention_mask = None
        unselected_attention_mask = None

    selected_labels = labels.clone() if labels is not None else None
    unselected_labels = labels.clone() if labels is not None else None

    return (
        selected_embeds,
        selected_labels,
        selected_attention_mask,
        position_ids,
        unselected_embeds,
        unselected_labels,
        unselected_attention_mask,
        image_ste_masks_flat,
    )

def prune_tokens_for_inference(
    scores,
    batch_image_ranges,
    inputs_embeds,
    attention_mask,
    position_ids,
    top_p=None,
    image_keep_counts=None,
    batch_text_indices=None,
    text_top_p=None,
    image_token_indices=None,
    tau=0.0,
    selection_mode="threshold",
):
    batch_size = inputs_embeds.shape[0]
    device = inputs_embeds.device
    trace_topk = os.environ.get("LLAVA_VISION_PRUNER_TRACE_TOPK", "0") == "1"
    trace_limit = int(os.environ.get("LLAVA_VISION_PRUNER_TRACE_TOPK_LIMIT", "5"))
    trace_full_indices = os.environ.get("LLAVA_VISION_PRUNER_TRACE_FULL_INDICES", "0") == "1"
    if not hasattr(prune_tokens_for_inference, "_trace_count"):
        prune_tokens_for_inference._trace_count = 0

    if isinstance(scores, dict):
        image_scores_list = scores.get("image", [])
    else:
        image_scores_list = scores

    new_embeds_list = []
    new_masks_list = []   # attention_mask가 None이면 사용하지 않음
    new_pos_ids_list = [] # position_ids가 None이면 사용하지 않음

    for b in range(batch_size):
        total_seq_len = inputs_embeds.shape[1]
        keep_mask = torch.ones(total_seq_len, dtype=torch.bool, device=device)

        image_scores = image_scores_list[b].squeeze(0) # [total_image_tokens]
        num_img_tokens = image_scores.shape[0]
        if num_img_tokens > 0:
            ranges = batch_image_ranges[b] if batch_image_ranges is not None else []
            if not ranges:
                final_indices = torch.where(keep_mask)[0]
                new_embeds_list.append(inputs_embeds[b, final_indices])
                if attention_mask is not None:
                    new_masks_list.append(attention_mask[b, final_indices])
                if position_ids is not None:
                    new_pos_ids_list.append(position_ids[b, final_indices])
                continue
            all_img_indices = torch.cat([
                torch.arange(start, end, device=device) for start, end in ranges
            ])
            token_count = min(num_img_tokens, all_img_indices.numel())
            if token_count == 0:
                final_indices = torch.where(keep_mask)[0]
                new_embeds_list.append(inputs_embeds[b, final_indices])
                if attention_mask is not None:
                    new_masks_list.append(attention_mask[b, final_indices])
                if position_ids is not None:
                    new_pos_ids_list.append(position_ids[b, final_indices])
                continue
            image_scores = image_scores[:token_count]
            all_img_indices = all_img_indices[:token_count]

            if selection_mode == "topk":
                if image_keep_counts is not None and b < len(image_keep_counts):
                    k = max(1, min(token_count, int(image_keep_counts[b])))
                else:
                    keep_ratio = 1.0 if top_p is None else float(top_p)
                    k = _get_keep_count(token_count, keep_ratio)
                selected_local_indices = torch.topk(image_scores, k=k).indices
                selected_local_indices, _ = torch.sort(selected_local_indices)
            else:
                selected_local_indices = torch.where(image_scores > tau)[0]
                k = int(selected_local_indices.numel())

            selected_img_indices = all_img_indices[selected_local_indices]

            if trace_topk and prune_tokens_for_inference._trace_count < trace_limit:
                selected_scores = image_scores[selected_local_indices]
                score_std = image_scores.float().std(unbiased=False).item() if token_count > 1 else 0.0
                original_local_indices = None
                token_index_map = None
                if image_token_indices is not None and b < len(image_token_indices):
                    token_index_map = image_token_indices[b].to(device=device, dtype=torch.long)
                    if token_index_map.numel() >= token_count and selected_local_indices.numel() > 0:
                        original_local_indices = token_index_map[:token_count][selected_local_indices]
                original_index_trace = ""
                if original_local_indices is not None and token_index_map is not None:
                    original_index_trace = (
                        f" pre_prune_orig_count={int(token_index_map.numel())} "
                        f"pre_prune_orig_sha={_digest_indices(token_index_map)} "
                        f"orig_local_sha={_digest_indices(original_local_indices)} "
                        f"first10_orig_local={original_local_indices[:10].tolist()}"
                    )
                    if trace_full_indices:
                        original_index_trace += (
                            f" pre_prune_orig_local={token_index_map.tolist()} "
                            f"selected_orig_local={original_local_indices.tolist()}"
                        )
                print(
                    "[VisionPruner select] "
                    f"sample={prune_tokens_for_inference._trace_count} batch={b} "
                    f"mode={selection_mode} image_tokens={token_count} keep={k} tau={float(tau):.6f} "
                    f"rel_sha={_digest_indices(selected_local_indices)} "
                    f"abs_seq_sha={_digest_indices(selected_img_indices)} "
                    f"score_min={image_scores.min().item():.6f} "
                    f"score_max={image_scores.max().item():.6f} "
                    f"score_mean={image_scores.float().mean().item():.6f} "
                    f"score_std={score_std:.6f} "
                    f"selected_score_min={(selected_scores.min().item() if selected_scores.numel() > 0 else float('nan')):.6f} "
                    f"selected_score_max={(selected_scores.max().item() if selected_scores.numel() > 0 else float('nan')):.6f} "
                    f"first10_rel={selected_local_indices[:10].tolist()}"
                    f"{original_index_trace}"
                )
                prune_tokens_for_inference._trace_count += 1

            keep_mask[all_img_indices] = False
            if selected_img_indices.numel() > 0:
                keep_mask[selected_img_indices] = True

        final_indices = torch.where(keep_mask)[0]

        new_embeds_list.append(inputs_embeds[b, final_indices])
        if attention_mask is not None:
            new_masks_list.append(attention_mask[b, final_indices])
        if position_ids is not None:
            new_pos_ids_list.append(position_ids[b, final_indices])

    # If different items have different lengths, we need to pad.
    max_len = max(x.shape[0] for x in new_embeds_list)

    final_embeds = []
    final_masks = [] if attention_mask is not None else None
    final_pos_ids = [] if position_ids is not None else None

    for i in range(batch_size):
        cur_len = new_embeds_list[i].shape[0]
        if cur_len < max_len:
            pad_len = max_len - cur_len

            pad_embeds = torch.zeros((pad_len, inputs_embeds.shape[-1]), device=device, dtype=inputs_embeds.dtype)
            final_embeds.append(torch.cat([new_embeds_list[i], pad_embeds], dim=0))

            if attention_mask is not None:
                pad_masks = torch.zeros(pad_len, device=device, dtype=attention_mask.dtype)
                final_masks.append(torch.cat([new_masks_list[i], pad_masks], dim=0))

            if position_ids is not None:
                pad_pos = torch.zeros(pad_len, device=device, dtype=position_ids.dtype)
                final_pos_ids.append(torch.cat([new_pos_ids_list[i], pad_pos], dim=0))
        else:
            final_embeds.append(new_embeds_list[i])
            if attention_mask is not None:
                final_masks.append(new_masks_list[i])
            if position_ids is not None:
                final_pos_ids.append(new_pos_ids_list[i])

    final_embeds = torch.stack(final_embeds)
    if attention_mask is not None:
        final_masks = torch.stack(final_masks)
    if position_ids is not None:
        final_pos_ids = torch.stack(final_pos_ids)

    return final_embeds, final_masks, final_pos_ids
