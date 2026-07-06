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
        vision_pruner_value_layer_idx: int = 0,
        vision_pruner_context_layer_idx: int = 9,
        vision_pruner_decoder_layer_idx: Optional[int] = None,
        vision_pruner_tau: float = 0.0,
        vision_pruner_rho: float = 0.1,
        vision_pruner_lambda_sparse: float = 1.0,
        vision_pruner_attention_pre_prune_keep_ratio: float = 1.0,
        vision_pruner_attention_pre_prune_layer: Optional[int] = None,
        vision_pruner_attention_pre_prune_head_reduction: str = "mean",
        **kwargs,
    ):
        # Older checkpoints may contain this key. TAU/RHO are fixed code constants now.
        kwargs.pop("vision_pruner_threshold", None)
        super().__init__(*args, **kwargs)
        if vision_pruner_decoder_layer_idx is not None:
            vision_pruner_value_layer_idx = vision_pruner_decoder_layer_idx
        self.vision_pruner_value_layer_idx = vision_pruner_value_layer_idx
        self.vision_pruner_context_layer_idx = vision_pruner_context_layer_idx
        self.vision_pruner_decoder_layer_idx = vision_pruner_value_layer_idx
        self.vision_pruner_tau = vision_pruner_tau
        self.vision_pruner_rho = vision_pruner_rho
        self.vision_pruner_lambda_sparse = vision_pruner_lambda_sparse
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
RHO: float = 0.1             # target image-token keep-rate
VISION_PRUNER_ATTENTION_PRE_PRUNE_KEEP_RATIO: float = 1.0
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
        unselected_inputs_embeds = None
        unselected_labels = None
        unselected_attention_mask = None

        if inputs_embeds is None:
            orig_input_ids = input_ids.clone()
            orig_attention_mask = attention_mask.clone() if attention_mask is not None else None
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

                scores = image_pruner(
                    input_ids=orig_input_ids,
                    hidden_states=inputs_embeds.detach(),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    batch_image_ranges=batch_image_ranges,
                    labels=labels,
                )
                (
                    inputs_embeds,
                    labels,
                    attention_mask,
                    position_ids,
                    unselected_inputs_embeds,
                    unselected_labels,
                    unselected_attention_mask,
                    image_ste_masks_flat,
                ) = prune_tokens_for_training(
                    scores,
                    batch_image_ranges,
                    batch_text_indices,
                    inputs_embeds,
                    labels,
                    attention_mask,
                    position_ids,
                    tau=getattr(self.config, "vision_pruner_tau", TAU),
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
            return_dict=True
        )

        has_image_sparse = image_ste_masks_flat is not None and image_ste_masks_flat.numel() > 0
        if outputs.loss is not None and has_image_sparse:
            l_select = outputs.loss
            rho = float(getattr(self.config, "vision_pruner_rho", RHO))
            lambda_sparse = float(getattr(self.config, "vision_pruner_lambda_sparse", LAMBDA_SPARSE))
            l_sparse = (image_ste_masks_flat.mean() - rho) ** 2
            if unselected_inputs_embeds is not None:
                unselected_outputs = super().forward(
                    input_ids=input_ids,
                    attention_mask=unselected_attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=unselected_inputs_embeds,
                    labels=unselected_labels,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=True,
                )
                l_unselect = (
                    unselected_outputs.loss
                    if unselected_outputs.loss is not None
                    else l_select.detach()
                )
            else:
                l_unselect = l_select.detach()

            l_contrast = torch.relu(l_select - l_unselect)
            weighted_l_sparse = lambda_sparse * l_sparse
            total_loss = l_select + weighted_l_sparse + l_contrast
            print(
                "vision_pruner_loss "
                f"task={l_select.item():.8f} "
                f"sparse={weighted_l_sparse.item():.8f} "
                f"contrast={l_contrast.item():.8f} "
                f"unselect={l_unselect.item():.8f} "
                f"total={total_loss.item():.8f}"
            )
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
        top_p: Optional[float] = None,
        text_top_p: Optional[float] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        kwargs.pop("prune_text_tokens", None)
        kwargs.pop("text_prune_mask", None)
        kwargs.pop("threshold", None)
        selection_mode = kwargs.pop(
            "vision_pruner_selection_mode",
            "topk" if top_p is not None else "threshold",
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

            image_pruner = self.get_vision_pruner()
            if (image_pruner is not None) and (inputs_embeds is not None):
                batch_image_ranges, batch_text_indices = get_image_and_text_token_indices(
                    orig_input_ids,
                    inputs_embeds,
                    orig_attention_mask=orig_attention_mask,
                    exp_attention_mask=attention_mask,
                )
                image_slot_counts = get_image_slot_counts(
                    orig_input_ids,
                    orig_attention_mask=orig_attention_mask,
                )
                final_image_keep_counts = None
                if selection_mode == "topk":
                    keep_ratio = 1.0 if top_p is None else top_p
                    final_image_keep_counts = self._image_keep_counts_for_final_ratio(
                        batch_image_ranges,
                        keep_ratio,
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
                    min_keep_ratio=(top_p if selection_mode == "topk" and top_p is not None else None),
                    return_image_token_indices=True,
                )

                scores = image_pruner(
                    input_ids=orig_input_ids,
                    hidden_states=inputs_embeds.detach(),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    batch_image_ranges=batch_image_ranges,
                )

                inputs_embeds, attention_mask, position_ids = prune_tokens_for_inference(
                    scores,
                    batch_image_ranges,
                    inputs_embeds,
                    attention_mask,
                    position_ids,
                    top_p=top_p,
                    image_keep_counts=final_image_keep_counts,
                    batch_text_indices=batch_text_indices,
                    text_top_p=None,
                    image_token_indices=pre_prune_image_token_indices,
                    tau=getattr(self.config, "vision_pruner_tau", TAU),
                    selection_mode=selection_mode,
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
