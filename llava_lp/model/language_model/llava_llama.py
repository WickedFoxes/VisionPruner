#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from ..llava_arch import LlavaMetaModel_with_VisionPruner, LlavaMetaForCausalLM_with_VisionPruner
from ..multimodal_pruner.pruner import (
    get_image_and_text_token_indices,
    get_image_slot_counts,
    prune_image_tokens_by_score,
    prune_tokens_for_training,
    prune_tokens_for_inference,
    scores_by_image_to_batch_token_scores,
)
from ..attention_score_pruning import (
    apply_vit_attention_score_pruning,
    compute_vit_cls_attention_scores,
)

class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaConfig_with_VisionPruner(LlavaConfig):
    model_type = "llava_llama_with_vision_pruner"

    def __init__(
        self,
        *args,
        vision_pruner_decoder_layer_idx: int = 0,
        vision_pruner_attention_pre_prune_keep_ratio: float = 0.50,
        vision_pruner_attention_pre_prune_layer: Optional[int] = None,
        vision_pruner_attention_pre_prune_head_reduction: str = "mean",
        **kwargs,
    ):
        # Older checkpoints may contain this key. TAU/RHO are fixed code constants now.
        kwargs.pop("vision_pruner_threshold", None)
        super().__init__(*args, **kwargs)
        self.vision_pruner_decoder_layer_idx = vision_pruner_decoder_layer_idx
        self.vision_pruner_attention_pre_prune_keep_ratio = vision_pruner_attention_pre_prune_keep_ratio
        self.vision_pruner_attention_pre_prune_layer = vision_pruner_attention_pre_prune_layer
        self.vision_pruner_attention_pre_prune_head_reduction = vision_pruner_attention_pre_prune_head_reduction


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        text_prune_mask: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
        ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        attention_score_pruning = kwargs.pop("attention_score_pruning", False)
        attention_score_pruning_top_p = kwargs.pop("attention_score_pruning_top_p", 1.0)
        attention_score_pruning_layer = kwargs.pop("attention_score_pruning_layer", None)
        attention_score_pruning_head_reduction = kwargs.pop(
            "attention_score_pruning_head_reduction", "mean"
        )
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            orig_input_ids = inputs.clone()
            orig_attention_mask = attention_mask.clone() if attention_mask is not None else None
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
            if attention_score_pruning:
                inputs_embeds, attention_mask, position_ids = apply_vit_attention_score_pruning(
                    self,
                    orig_input_ids=orig_input_ids,
                    orig_attention_mask=orig_attention_mask,
                    images=images,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    keep_ratio=attention_score_pruning_top_p,
                    select_layer=attention_score_pruning_layer,
                    head_reduction=attention_score_pruning_head_reduction,
                )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs


###################### VisionPruner ######################

# Sparsity-loss hyper-parameters
# L_total = L_task + LAMBDA_SPARSE * L_sparse
# L_sparse = (mean(M_ste) - RHO)^2
LAMBDA_SPARSE: float = 1.0   # weight of the sparsity term
TAU: float = 0.0             # score cutoff (0.0 이하이면 마스킹)
RHO: float = 0.05            # target keep-rate (최소한 5%는 0.0 이상이 되도록 학습)
IMAGE_SPARSE_LOSS_WEIGHT: float = 0.5
TEXT_SPARSE_LOSS_WEIGHT: float = 0.5
VISION_PRUNER_ATTENTION_PRE_PRUNE_KEEP_RATIO: float = 0.50
VISION_PRUNER_ATTENTION_PRE_PRUNE_HEAD_REDUCTION: str = "mean"


def _maybe_enable_vit_attention_scores_for_vision_pruner(config):
    keep_ratio = getattr(
        config,
        "vision_pruner_attention_pre_prune_keep_ratio",
        VISION_PRUNER_ATTENTION_PRE_PRUNE_KEEP_RATIO,
    )
    if keep_ratio is None:
        keep_ratio = VISION_PRUNER_ATTENTION_PRE_PRUNE_KEEP_RATIO
    keep_ratio = float(keep_ratio)
    if 0.0 < keep_ratio < 1.0:
        os.environ["LLAVA_ATTENTION_SCORE_PRUNING"] = "1"


class LlavaLlamaModel_with_VisionPruner(LlavaMetaModel_with_VisionPruner, LlamaModel):
    config_class = LlavaConfig_with_VisionPruner

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel_with_VisionPruner, self).__init__(config)


class LlavaLlamaForCausalLM_with_VisionPruner(LlamaForCausalLM, LlavaMetaForCausalLM_with_VisionPruner):
    config_class = LlavaConfig_with_VisionPruner

    def __init__(self, config):
        _maybe_enable_vit_attention_scores_for_vision_pruner(config)
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel_with_VisionPruner(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def _apply_vision_pruner_attention_pre_pruning(
        self,
        images,
        batch_image_ranges,
        batch_text_indices,
        inputs_embeds,
        labels,
        attention_mask,
        position_ids,
        image_slot_counts=None,
        min_keep_ratio=None,
        return_image_token_indices=False,
    ):
        keep_ratio = getattr(
            self.config,
            "vision_pruner_attention_pre_prune_keep_ratio",
            VISION_PRUNER_ATTENTION_PRE_PRUNE_KEEP_RATIO,
        )
        if keep_ratio is None:
            keep_ratio = VISION_PRUNER_ATTENTION_PRE_PRUNE_KEEP_RATIO
        keep_ratio = float(keep_ratio)
        if min_keep_ratio is not None:
            keep_ratio = max(keep_ratio, float(min_keep_ratio))
        if keep_ratio >= 1.0:
            if return_image_token_indices:
                image_token_indices = []
                for ranges in batch_image_ranges:
                    num_image_tokens = sum(max(0, int(end) - int(start)) for start, end in ranges)
                    image_token_indices.append(
                        torch.arange(num_image_tokens, device=inputs_embeds.device, dtype=torch.long)
                    )
                return (
                    inputs_embeds,
                    labels,
                    attention_mask,
                    position_ids,
                    batch_image_ranges,
                    batch_text_indices,
                    image_token_indices,
                )
            return inputs_embeds, labels, attention_mask, position_ids, batch_image_ranges, batch_text_indices
        if keep_ratio <= 0.0:
            raise ValueError(
                "vision_pruner_attention_pre_prune_keep_ratio must be in (0, 1], "
                f"got {keep_ratio}."
            )

        select_layer = getattr(self.config, "vision_pruner_attention_pre_prune_layer", None)
        head_reduction = getattr(
            self.config,
            "vision_pruner_attention_pre_prune_head_reduction",
            VISION_PRUNER_ATTENTION_PRE_PRUNE_HEAD_REDUCTION,
        )

        with torch.no_grad():
            scores_by_image = compute_vit_cls_attention_scores(
                self,
                images,
                select_layer=select_layer,
                head_reduction=head_reduction,
            )
            pre_prune_scores = scores_by_image_to_batch_token_scores(
                scores_by_image,
                batch_image_ranges,
                device=inputs_embeds.device,
                image_slot_counts=image_slot_counts,
            )

        return prune_image_tokens_by_score(
            pre_prune_scores,
            batch_image_ranges,
            inputs_embeds,
            attention_mask,
            position_ids,
            top_p=keep_ratio,
            labels=labels,
            batch_text_indices=batch_text_indices,
            return_image_token_indices=return_image_token_indices,
        )

    @staticmethod
    def _image_keep_counts_for_final_ratio(batch_image_ranges, keep_ratio: float):
        keep_ratio = float(keep_ratio)
        if keep_ratio <= 0.0 or keep_ratio > 1.0:
            raise ValueError(f"VisionPruner keep ratio must be in (0, 1], got {keep_ratio}.")

        keep_counts = []
        for ranges in batch_image_ranges:
            num_image_tokens = sum(max(0, int(end) - int(start)) for start, end in ranges)
            if num_image_tokens == 0:
                keep_counts.append(0)
            else:
                keep_counts.append(max(1, min(num_image_tokens, int(num_image_tokens * keep_ratio))))
        return keep_counts

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        text_prune_mask: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        image_ste_masks_flat = None  # image-token M_ste values collected for L_sparse
        text_ste_masks_flat = None   # question-text-token M_ste values collected for L_sparse

        if inputs_embeds is None:
            orig_input_ids = input_ids.clone()
            orig_attention_mask = attention_mask.clone() if attention_mask is not None else None
            orig_text_prune_mask = text_prune_mask.clone() if text_prune_mask is not None else None
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )

            image_pruner = self.get_vision_pruner()
            if (image_pruner is not None) and (inputs_embeds is not None):
                batch_image_ranges, batch_text_indices = get_image_and_text_token_indices(
                    orig_input_ids,
                    inputs_embeds,
                    orig_attention_mask=orig_attention_mask,
                    exp_attention_mask=attention_mask,
                    labels=labels,
                    text_prune_mask=orig_text_prune_mask,
                )
                image_slot_counts = get_image_slot_counts(
                    orig_input_ids,
                    orig_attention_mask=orig_attention_mask,
                )
                (
                    inputs_embeds,
                    labels,
                    attention_mask,
                    position_ids,
                    batch_image_ranges,
                    batch_text_indices,
                ) = self._apply_vision_pruner_attention_pre_pruning(
                    images,
                    batch_image_ranges,
                    batch_text_indices,
                    inputs_embeds,
                    labels,
                    attention_mask,
                    position_ids,
                    image_slot_counts=image_slot_counts,
                )

                # 1. 이미지 토큰 score와 질문 텍스트 토큰 score를 따로 계산
                scores = image_pruner(
                    input_ids=orig_input_ids,
                    hidden_states=inputs_embeds.detach(),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    batch_image_ranges=batch_image_ranges,
                    batch_text_indices=batch_text_indices,
                    labels=labels,
                )
                # 2. 이미지/질문 텍스트 토큰 모두 STE 마스킹 (fixed TAU 기반)
                #    M_i = 1 if S_i > τ else 0
                #    M_ste_i = S_i + stop_grad(M_i − S_i)
                #    *_ste_masks_flat: L_sparse 계산을 위해 반환된 M_ste 값
                (
                    inputs_embeds,
                    labels,
                    attention_mask,
                    position_ids,
                    image_ste_masks_flat,
                    text_ste_masks_flat,
                ) = prune_tokens_for_training(
                    scores,
                    batch_image_ranges,
                    batch_text_indices,
                    inputs_embeds,
                    labels,
                    attention_mask,
                    position_ids,
                    tau=TAU
                )

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        # 3. Sparsity loss 계산 및 합산
        #    L_sparse_img/text = (mean(M_ste) − rho)²
        #    L_sparse = 0.5 * L_sparse_img + 0.5 * L_sparse_text
        #    L_total  = L_task + λ_sparse * L_sparse   [λ_sparse = 1]
        has_image_sparse = image_ste_masks_flat is not None and image_ste_masks_flat.numel() > 0
        has_text_sparse = text_ste_masks_flat is not None and text_ste_masks_flat.numel() > 0
        if outputs.loss is not None and (has_image_sparse or has_text_sparse):
            l_task = outputs.loss
            l_sparse_image = (
                (image_ste_masks_flat.mean() - RHO) ** 2
                if has_image_sparse
                else l_task.new_zeros(())
            )
            l_sparse_text = (
                (text_ste_masks_flat.mean() - RHO) ** 2
                if has_text_sparse
                else l_task.new_zeros(())
            )
            weighted_l_sparse = LAMBDA_SPARSE * (
                IMAGE_SPARSE_LOSS_WEIGHT * l_sparse_image
                + TEXT_SPARSE_LOSS_WEIGHT * l_sparse_text
            )
            print(f"l_task : {l_task.item():.8f}")
            print(
                "weighted_l_sparse "
                f"image={l_sparse_image.item():.8f}*{IMAGE_SPARSE_LOSS_WEIGHT:.1f}, "
                f"text={l_sparse_text.item():.8f}*{TEXT_SPARSE_LOSS_WEIGHT:.1f}, "
                f"total={weighted_l_sparse.item():.8f}"
            )
            total_loss = l_task + weighted_l_sparse
            outputs = CausalLMOutputWithPast(
                loss=total_loss,
                logits=outputs.logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        
        return outputs

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        top_p: float = 1.0,
        text_top_p: Optional[float] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        prune_text_tokens = kwargs.pop("prune_text_tokens", None)
        text_prune_mask = kwargs.pop("text_prune_mask", None)
        # Ignore legacy callers; pruning now uses fixed TAU/RHO constants.
        kwargs.pop("threshold", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        if prune_text_tokens is False:
            text_top_p = None
        elif prune_text_tokens is True and text_top_p is None:
            text_top_p = top_p

        if images is not None:
            orig_input_ids = inputs.clone()
            orig_attention_mask = attention_mask.clone() if attention_mask is not None else None
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )

            image_pruner = self.get_vision_pruner()
            if (image_pruner is not None) and (inputs_embeds is not None):
                batch_image_ranges, batch_text_indices = get_image_and_text_token_indices(
                    orig_input_ids,
                    inputs_embeds,
                    orig_attention_mask=orig_attention_mask,
                    exp_attention_mask=attention_mask,
                    text_prune_mask=text_prune_mask,
                )
                image_slot_counts = get_image_slot_counts(
                    orig_input_ids,
                    orig_attention_mask=orig_attention_mask,
                )
                final_image_keep_counts = self._image_keep_counts_for_final_ratio(
                    batch_image_ranges,
                    top_p,
                )
                (
                    inputs_embeds,
                    _,
                    attention_mask,
                    position_ids,
                    batch_image_ranges,
                    batch_text_indices,
                    pre_prune_image_token_indices,
                ) = self._apply_vision_pruner_attention_pre_pruning(
                    images,
                    batch_image_ranges,
                    batch_text_indices,
                    inputs_embeds,
                    None,
                    attention_mask,
                    position_ids,
                    image_slot_counts=image_slot_counts,
                    min_keep_ratio=top_p,
                    return_image_token_indices=True,
                )

                # 1. 이미지 토큰 score와 질문 텍스트 토큰 score를 따로 계산
                scores = image_pruner(
                    input_ids=orig_input_ids,
                    hidden_states=inputs_embeds.detach(),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    batch_image_ranges=batch_image_ranges,
                    batch_text_indices=batch_text_indices,
                )

                # 2. 이미지 토큰은 원본 image token 수 기준 top_p 최종 개수로 프루닝한다.
                #    텍스트 토큰은 text_top_p가 있을 때만 top-k 프루닝한다.
                inputs_embeds, attention_mask, position_ids = prune_tokens_for_inference(
                    scores,
                    batch_image_ranges,
                    inputs_embeds,
                    attention_mask,
                    position_ids,
                    top_p=top_p,
                    image_keep_counts=final_image_keep_counts,
                    batch_text_indices=batch_text_indices,
                    text_top_p=text_top_p,
                    image_token_indices=pre_prune_image_token_indices,
                )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

##########################################################

AutoConfig.register("llava_llama", LlavaConfig)
AutoConfig.register("llava_llama_with_vision_pruner", LlavaConfig_with_VisionPruner)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
AutoModelForCausalLM.register(LlavaConfig_with_VisionPruner, LlavaLlamaForCausalLM_with_VisionPruner)
