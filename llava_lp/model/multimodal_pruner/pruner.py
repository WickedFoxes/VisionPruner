import copy
import hashlib
import os
import torch
import torch.nn as nn
from llava.constants import IMAGE_TOKEN_INDEX, IGNORE_INDEX
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask


class LlavaImagePruner(nn.Module):
    def __init__(self, llava_decoder_layer):
        super().__init__()
        self.layer = copy.deepcopy(llava_decoder_layer)
        self.verbose = os.environ.get("LLAVA_VISION_PRUNER_VERBOSE", "1") == "1"
        self.score_noise_std = 0.0
        self.score_noise_variance = 0.0
        self.score_noise_tail_threshold = 0.1
        self.score_noise_last_std = 0.0
        self.score_noise_last_abs_ge_threshold_ratio = 0.0

    @staticmethod
    def _stable_l2_normalize(tensor: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
        tensor = tensor.float()
        scale = tensor.detach().abs().amax(dim=dim, keepdim=True).clamp_min(eps)
        scaled = tensor / scale
        inv_norm = torch.rsqrt(scaled.pow(2).sum(dim=dim, keepdim=True).clamp_min(eps * eps))
        return scaled * inv_norm

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
        if attention_mask is not None and attention_mask.dim() == 2:
            attention_mask_4d = _prepare_4d_causal_attention_mask(
                attention_mask,
                (hidden_states.shape[0], hidden_states.shape[1]),
                hidden_states,
                past_key_value[0].shape[-2] if past_key_value is not None else 0,
            )
        else:
            attention_mask_4d = attention_mask.float() if attention_mask is not None else None

        # self.layer 통과하여 outputs 구하기 (fp32 연산)
        outputs = self.layer(
            hidden_states=hidden_states,
            attention_mask=attention_mask_4d,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs
        )
        
        layer_output_states = outputs[0] # [batch_size, expanded_seq_len, hidden_size]

        batch_size = layer_output_states.shape[0]
        seq_len = layer_output_states.shape[1]
        hidden_size = layer_output_states.shape[-1]

        # 스코어 계산을 위한 "마지막 토큰" 구하기
        if labels is not None:
            # [훈련 단계] Prompt의 마지막 토큰 찾기
            # Prompt는 labels에서 IGNORE_INDEX(-100)로 마스킹되어 있으므로, 
            # 첫 번째 유효 레이블 바로 이전의 토큰을 Prompt의 마지막 토큰으로 간주함.
            last_token_indices = []
            for b in range(batch_size):
                valid_idx = torch.where(labels[b] != IGNORE_INDEX)[0]
                if len(valid_idx) > 0:
                    # 답변이 시작되기 직전의 토큰 (일반적으로 'Assistant:' 토큰)
                    idx = valid_idx[0].item() - 1
                    last_token_indices.append(max(0, idx))
                else:
                    # 만약 유효한 레이블이 없다면(에러 상황 등), 패딩을 고려한 시퀀스 끝 사용
                    idx = attention_mask_2d[b].cumsum(dim=0).argmax().item()
                    last_token_indices.append(idx)
            last_token_indices = torch.tensor(last_token_indices, device=layer_output_states.device)
        else:
            # [추론 단계] 현재 입력 시퀀스의 마지막 토큰 찾기
            # 패딩을 고려하여 실제 마지막 토큰의 위치를 찾음
            last_token_indices = attention_mask_2d.cumsum(dim=1).argmax(dim=1)
        
        last_token_output = layer_output_states[torch.arange(batch_size), last_token_indices].unsqueeze(1)
        
        image_scores_list = []
        text_scores_list = []
        score_noise_samples = []
        
        # 배치 내 각 아이템별로 처리
        for b in range(batch_size):
            ranges = batch_image_ranges[b] if batch_image_ranges is not None else []
            last_token_b = last_token_output[b:b+1, :, :] # [1, 1, hidden_size]

            def compute_scores(target_tokens):
                query = self._stable_l2_normalize(last_token_b, dim=-1, eps=1e-6)
                keys = self._stable_l2_normalize(target_tokens, dim=-1, eps=1e-6)
                scores = torch.matmul(query, keys.transpose(-1, -2)).squeeze(1)
                scores = scores.clamp(min=-1.0, max=1.0)

                if self.training:
                    score_noise_std = float(getattr(self, "score_noise_std", 0.0) or 0.0)
                    if score_noise_std > 0.0:
                        noise = torch.randn_like(scores) * score_noise_std
                        scores = scores + noise
                        scores = scores.clamp(min=-1.0, max=1.0)
                        score_noise_samples.append(noise.detach().float().reshape(-1))
                return scores

            if len(ranges) == 0:
                # 이미지가 없는 예외 상황의 경우 빈 스코어 생성
                empty_score = torch.empty((1, 0), device=hidden_states.device)
                image_scores_list.append(empty_score)
            else:
                # 해당 배치 아이템의 모든 이미지 패치 추출 및 병합: [1, total_image_tokens, hidden_size]
                # fp32로 캐스팅된 hidden_states 사용 (last_token_b와 dtype 일치)
                image_tokens_target = torch.cat(
                    [hidden_states[b:b+1, s:e, :] for s, e in ranges], dim=1
                )
                image_scores_list.append(compute_scores(image_tokens_target))

            if batch_text_indices is None:
                text_indices = torch.empty(0, device=hidden_states.device, dtype=torch.long)
            else:
                text_indices = batch_text_indices[b].to(device=hidden_states.device, dtype=torch.long)
                if text_indices.numel() > 0:
                    text_indices = text_indices[(text_indices >= 0) & (text_indices < seq_len)]
                if text_indices.numel() > 0:
                    text_indices = text_indices[attention_mask_2d[b, text_indices].bool()]

            if text_indices.numel() == 0:
                text_scores_list.append(torch.empty((1, 0), device=hidden_states.device))
            else:
                text_tokens_target = hidden_states[b:b+1, text_indices, :]
                text_scores_list.append(compute_scores(text_tokens_target))

        score_noise_std = float(getattr(self, "score_noise_std", 0.0) or 0.0)
        score_noise_tail_threshold = float(getattr(self, "score_noise_tail_threshold", 0.1) or 0.1)
        self.score_noise_variance = score_noise_std ** 2
        if score_noise_samples:
            noise_samples = torch.cat(score_noise_samples)
            self.score_noise_last_std = noise_samples.std(unbiased=False).item()
            self.score_noise_last_abs_ge_threshold_ratio = (
                noise_samples.abs() >= score_noise_tail_threshold
            ).float().mean().item()
        elif self.training:
            self.score_noise_last_std = 0.0
            self.score_noise_last_abs_ge_threshold_ratio = 0.0
        
        # 스코어가 잘 나오고 있는지 출력
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
            for b, s in enumerate(text_scores_list):
                scores = s.squeeze(0)
                n = scores.shape[0]
                if n == 0:
                    print(f"[batch {b}] question text tokens: 0")
                    continue
                k = max(1, int(n * 0.1))
                topk_idx = torch.topk(scores, k=k).indices
                score_cutoff = 0.0
                print(f"[batch {b}] question text tokens: {n}, top 10% ({k}) indices: {topk_idx.tolist()}")
                print(f"scores : {scores[topk_idx]}")
                print(f"         score range: min={scores.min().item():.4f}, max={scores.max().item():.4f}")
                print(f"         scores >= {score_cutoff}: {(scores >= score_cutoff).sum().item()} / {n}")

        # q_proj의 가중치를 간략 출력
        # target_layer = self.layer.self_attn.q_proj.weight
        # print("self_attn.q_proj.weight", target_layer[:5, :5])

        return {"image": image_scores_list, "text": text_scores_list}


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

    Returns image/text ste mask tensors so the caller can compute separate sparsity losses:
        L_sparse_* = (mean(M_ste_*) − ρ)²
    """
    batch_size = inputs_embeds.shape[0]
    device = inputs_embeds.device

    # Keep the STE mask in fp32 because cosine scores are fp32 even when embeddings are bf16.
    diff_mask = torch.ones((batch_size, inputs_embeds.shape[1]), device=device, dtype=torch.float32)

    if isinstance(scores, dict):
        image_scores_list = scores.get("image", [])
        text_scores_list = scores.get("text", [])
    else:
        image_scores_list = scores
        text_scores_list = []

    all_image_ste_masks = []
    all_text_ste_masks = []

    def build_ste_mask(score_tensor):
        M = (score_tensor > tau).to(score_tensor.dtype)
        return score_tensor + (M - score_tensor).detach()

    for b in range(batch_size):
        image_scores = image_scores_list[b].squeeze(0)  # [total_image_tokens]
        num_img_tokens = image_scores.shape[0]
        if num_img_tokens == 0:
            pass
        else:
            # STE: numerically equals the hard binary mask in forward, but gradient flows through scores.
            image_M_ste = build_ste_mask(image_scores)
            all_image_ste_masks.append(image_M_ste)

            ranges = batch_image_ranges[b]
            all_img_indices = torch.cat([
                torch.arange(start, end, device=device) for start, end in ranges
            ])
            diff_mask[b, all_img_indices] = image_M_ste

        if text_scores_list:
            text_scores = text_scores_list[b].squeeze(0)
            text_indices = batch_text_indices[b].to(device=device, dtype=torch.long)
            if text_scores.numel() > 0 and text_indices.numel() > 0:
                token_count = min(text_scores.numel(), text_indices.numel())
                text_scores = text_scores[:token_count]
                text_indices = text_indices[:token_count]
                text_M_ste = build_ste_mask(text_scores)
                all_text_ste_masks.append(text_M_ste)
                diff_mask[b, text_indices] = text_M_ste

    # Flat tensors of all M_ste values — used by the caller for separate L_sparse terms.
    image_ste_masks_flat = (
        torch.cat(all_image_ste_masks) if all_image_ste_masks else inputs_embeds.new_empty(0)
    )
    text_ste_masks_flat = (
        torch.cat(all_text_ste_masks) if all_text_ste_masks else inputs_embeds.new_empty(0)
    )

    # Differentiable mask: pruned image/question-text tokens become zero vectors.
    # Cast only at the multiplication boundary so the downstream model keeps its input dtype.
    final_embeds = inputs_embeds * diff_mask.to(inputs_embeds.dtype).unsqueeze(-1)

    # Attention mask: exclude pruned tokens from attention
    if attention_mask is not None:
        final_masks = attention_mask * diff_mask.to(attention_mask.dtype)
    else:
        final_masks = None

    # Labels: ignore pruned positions in the task loss
    final_labels = labels.clone() if labels is not None else None
    if final_labels is not None:
        # diff_mask is numerically 0 or 1 in the forward pass (STE property)
        final_labels[diff_mask.detach() == 0] = IGNORE_INDEX

    return final_embeds, final_labels, final_masks, position_ids, image_ste_masks_flat, text_ste_masks_flat

def prune_tokens_for_inference(
    scores,
    batch_image_ranges,
    inputs_embeds,
    attention_mask,
    position_ids,
    top_p=0.05,
    image_keep_counts=None,
    batch_text_indices=None,
    text_top_p=None,
    image_token_indices=None,
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
        text_scores_list = scores.get("text", [])
    else:
        image_scores_list = scores
        text_scores_list = []

    new_embeds_list = []
    new_masks_list = []   # attention_mask가 None이면 사용하지 않음
    new_pos_ids_list = [] # position_ids가 None이면 사용하지 않음

    for b in range(batch_size):
        total_seq_len = inputs_embeds.shape[1]
        keep_mask = torch.ones(total_seq_len, dtype=torch.bool, device=device)

        image_scores = image_scores_list[b].squeeze(0) # [total_image_tokens]
        num_img_tokens = image_scores.shape[0]
        if num_img_tokens > 0:
            if image_keep_counts is not None and b < len(image_keep_counts):
                k = max(1, min(num_img_tokens, int(image_keep_counts[b])))
            else:
                k = _get_keep_count(num_img_tokens, top_p)
            topk_indices = torch.topk(image_scores, k=k).indices
            topk_indices, _ = torch.sort(topk_indices)

            ranges = batch_image_ranges[b]
            all_img_indices = torch.cat([
                torch.arange(start, end, device=device) for start, end in ranges
            ])
            selected_img_indices = all_img_indices[topk_indices]

            if trace_topk and prune_tokens_for_inference._trace_count < trace_limit:
                selected_scores = image_scores[topk_indices]
                score_std = image_scores.float().std(unbiased=False).item() if num_img_tokens > 1 else 0.0
                original_local_indices = None
                token_index_map = None
                if image_token_indices is not None and b < len(image_token_indices):
                    token_index_map = image_token_indices[b].to(device=device, dtype=torch.long)
                    if token_index_map.numel() >= num_img_tokens:
                        original_local_indices = token_index_map[:num_img_tokens][topk_indices]
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
                    "[VisionPruner topk] "
                    f"sample={prune_tokens_for_inference._trace_count} batch={b} "
                    f"image_tokens={num_img_tokens} keep={k} keep_ratio={float(top_p):.6f} "
                    f"rel_sha={_digest_indices(topk_indices)} "
                    f"abs_seq_sha={_digest_indices(selected_img_indices)} "
                    f"score_min={image_scores.min().item():.6f} "
                    f"score_max={image_scores.max().item():.6f} "
                    f"score_mean={image_scores.float().mean().item():.6f} "
                    f"score_std={score_std:.6f} "
                    f"selected_score_min={selected_scores.min().item():.6f} "
                    f"selected_score_max={selected_scores.max().item():.6f} "
                    f"first10_rel={topk_indices[:10].tolist()}"
                    f"{original_index_trace}"
                )
                prune_tokens_for_inference._trace_count += 1

            keep_mask[all_img_indices] = False
            keep_mask[selected_img_indices] = True

        if text_top_p is not None and text_scores_list and batch_text_indices is not None:
            text_scores = text_scores_list[b].squeeze(0)
            text_indices = batch_text_indices[b].to(device=device, dtype=torch.long)
            if text_scores.numel() > 0 and text_indices.numel() > 0:
                token_count = min(text_scores.numel(), text_indices.numel())
                text_scores = text_scores[:token_count]
                text_indices = text_indices[:token_count]
                k = _get_keep_count(token_count, text_top_p)
                topk_indices = torch.topk(text_scores, k=k).indices
                topk_indices, _ = torch.sort(topk_indices)
                selected_text_indices = text_indices[topk_indices]

                keep_mask[text_indices] = False
                keep_mask[selected_text_indices] = True

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
