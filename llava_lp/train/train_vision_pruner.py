# Adopted from llava_lp/train/train.py
# Modified for vision_pruner-only fine-tuning:
#   - All parameters are frozen except LlavaMetaModel_with_VisionPruner.vision_pruner
#   - Checkpoints save only VisionPruner weights plus lightweight metadata

import os
import copy
from dataclasses import dataclass, field
import json
import logging
import random
from typing import Dict, Optional, Sequence, Set

import torch

import transformers
import tokenizers

from llava.constants import (
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from torch.utils.data import Dataset
from llava.train.llava_trainer import LLaVATrainer

VISION_PRUNER_SCORE_NOISE_TAIL_THRESHOLD = 0.1
VISION_PRUNER_DEFAULT_SCORE_NOISE_STD = VISION_PRUNER_SCORE_NOISE_TAIL_THRESHOLD / 1.959963984540054


class VisionPrunerTrainer(LLaVATrainer):
    def create_optimizer(self):
        optimizer = super().create_optimizer()
        self._register_optimizer_nan_guard(optimizer)
        return optimizer

    def _register_optimizer_nan_guard(self, optimizer):
        if optimizer is None or getattr(optimizer, "_vision_pruner_nan_guard_registered", False):
            return

        def pre_hook(opt, args, kwargs):
            self._check_optimizer_tensors(opt, "pre_step")
            return None

        def post_hook(opt, args, kwargs):
            self._check_optimizer_tensors(opt, "post_step")
            return None

        if hasattr(optimizer, "register_step_pre_hook"):
            optimizer.register_step_pre_hook(pre_hook)
        if hasattr(optimizer, "register_step_post_hook"):
            optimizer.register_step_post_hook(post_hook)
        optimizer._vision_pruner_nan_guard_registered = True

    def _optimizer_param_name_by_id(self):
        return {id(param): name for name, param in self.model.named_parameters()}

    def _record_tensor_issue(self, issues, label: str, tensor, max_abs_allowed: float = 0.0):
        if tensor is None or not torch.is_tensor(tensor) or not tensor.is_floating_point():
            return
        detached = tensor.detach()
        finite = torch.isfinite(detached)
        if not finite.all():
            nonfinite_count = int((~finite).sum().item())
            issues.append(f"{label}: nonfinite={nonfinite_count}/{detached.numel()}")
            return
        if max_abs_allowed > 0.0:
            max_abs = detached.abs().max().item()
            if max_abs > max_abs_allowed:
                issues.append(f"{label}: max_abs={max_abs:.6e} > {max_abs_allowed:.6e}")

    def _check_optimizer_tensors(self, optimizer, location: str):
        issues = []
        name_by_id = self._optimizer_param_name_by_id()
        max_param_abs = float(getattr(self.args, "vision_pruner_max_param_abs", 0.0) or 0.0)

        with torch.no_grad():
            for group_idx, group in enumerate(optimizer.param_groups):
                for param_idx, param in enumerate(group.get("params", [])):
                    name = name_by_id.get(id(param), f"optimizer_group_{group_idx}.param_{param_idx}")
                    self._record_tensor_issue(
                        issues,
                        f"param {name}",
                        param.data,
                        max_abs_allowed=max_param_abs,
                    )
                    self._record_tensor_issue(issues, f"grad {name}", param.grad)

            for state_idx, (param, state) in enumerate(optimizer.state.items()):
                name = name_by_id.get(id(param), f"optimizer_state_param_{state_idx}")
                for state_name, value in state.items():
                    self._record_tensor_issue(issues, f"state {name}.{state_name}", value)

        if issues:
            preview = "\n  ".join(issues[:12])
            if len(issues) > 12:
                preview += f"\n  ... and {len(issues) - 12} more"
            raise FloatingPointError(
                f"[vision_pruner] optimizer tensor check failed at {location}.\n"
                f"  {preview}\n"
                "Training stopped before silently zeroing/corrupting VisionPruner weights."
            )

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        """Save only the trainable VisionPruner adapter, not the full base model."""
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        if not self.args.should_save:
            return

        self.model.config.save_pretrained(output_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
        save_vision_pruner_checkpoint(self, output_dir)

    def _save_checkpoint(self, model, trial, metrics=None):
        from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)

        if hasattr(self, "store_flos"):
            self.store_flos()
        self._save(output_dir)

        if self.args.should_save:
            self.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))
            self._rotate_checkpoints(use_mtime=False, output_dir=run_dir)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        vision_pruner_bin = os.path.join(resume_from_checkpoint, "vision_pruner.bin")
        if os.path.exists(vision_pruner_bin):
            load_vision_pruner_checkpoint_into_model(
                model if model is not None else self.model,
                vision_pruner_bin,
            )
            return

        return super()._load_from_checkpoint(resume_from_checkpoint, model)

    def _load_optimizer_and_scheduler(self, checkpoint):
        vision_pruner_bin = os.path.join(checkpoint, "vision_pruner.bin") if checkpoint else None
        if vision_pruner_bin and os.path.exists(vision_pruner_bin):
            rank0_print(
                "[vision_pruner] optimizer/scheduler state is not saved for "
                "adapter-only checkpoints; continuing with fresh optimizer state."
            )
            return

        return super()._load_optimizer_and_scheduler(checkpoint)

    def _get_vision_pruner_module(self, model):
        module = model.module if hasattr(model, "module") else model
        if hasattr(module, "get_model"):
            module = module.get_model()
        return getattr(module, "vision_pruner", None)

    def _update_vision_pruner_score_noise(self, model):
        vision_pruner = self._get_vision_pruner_module(model)
        if vision_pruner is None:
            return None

        std = float(self.args.vision_pruner_score_noise_std)
        vision_pruner.score_noise_std = std
        vision_pruner.score_noise_variance = std ** 2
        vision_pruner.score_noise_tail_threshold = VISION_PRUNER_SCORE_NOISE_TAIL_THRESHOLD
        return std

    def _vision_pruner_stats(self, model):
        stats = []
        for name, param in model.named_parameters():
            if "vision_pruner" not in name or not param.requires_grad:
                continue
            with torch.no_grad():
                data = maybe_zero_3(param, ignore_status=True, name=name)
                stats.append((name, data.float().norm().item(), data.float().abs().max().item()))
        return stats

    def _log_vision_pruner_training_status(self, model):
        if self.args.local_rank not in [-1, 0]:
            return

        trainable_count = 0
        grad_none_count = 0
        total_grad_norm_sq = 0.0
        max_grad_abs = 0.0

        for name, param in model.named_parameters():
            if "vision_pruner" not in name or not param.requires_grad:
                continue
            trainable_count += param.numel()
            if param.grad is None:
                grad_none_count += 1
                continue
            grad = maybe_zero_3(param.grad, ignore_status=True, name=f"{name}.grad").float()
            grad_norm = grad.norm().item()
            total_grad_norm_sq += grad_norm ** 2
            max_grad_abs = max(max_grad_abs, grad.abs().max().item())

        total_grad_norm = total_grad_norm_sq ** 0.5
        stats = self._vision_pruner_stats(model)
        param_norm = sum(norm ** 2 for _, norm, _ in stats) ** 0.5
        max_param_abs = max((max_abs for _, _, max_abs in stats), default=0.0)
        vision_pruner = self._get_vision_pruner_module(model)
        score_noise_std = float(getattr(vision_pruner, "score_noise_std", 0.0) or 0.0)
        score_noise_variance = float(getattr(vision_pruner, "score_noise_variance", 0.0) or 0.0)
        score_noise_last_std = float(getattr(vision_pruner, "score_noise_last_std", 0.0) or 0.0)
        score_noise_tail_threshold = float(
            getattr(vision_pruner, "score_noise_tail_threshold", VISION_PRUNER_SCORE_NOISE_TAIL_THRESHOLD)
            or VISION_PRUNER_SCORE_NOISE_TAIL_THRESHOLD
        )
        score_noise_tail_ratio = float(
            getattr(vision_pruner, "score_noise_last_abs_ge_threshold_ratio", 0.0) or 0.0
        )

        print(
            f"[vision_pruner] step={self.state.global_step} "
            f"trainable_params={trainable_count:,} grad_none_tensors={grad_none_count} "
            f"grad_norm={total_grad_norm:.6e} max_grad_abs={max_grad_abs:.6e} "
            f"param_norm={param_norm:.6e} max_param_abs={max_param_abs:.6e} "
            f"score_noise_std={score_noise_std:.6f} score_noise_variance={score_noise_variance:.6f} "
            f"last_noise_std={score_noise_last_std:.6f} "
            f"last_abs_noise>={score_noise_tail_threshold:.1f}={score_noise_tail_ratio:.4f}"
        )

    def training_step(self, model, inputs, *args, **kwargs):
        self._update_vision_pruner_score_noise(model)
        loss = super().training_step(model, inputs, *args, **kwargs)

        if self.state.global_step % self.args.logging_steps == 0 or self.state.global_step == 1:
            self._log_vision_pruner_training_status(model)
        return loss


from llava import conversation as conversation_lib
from llava.mm_utils import tokenizer_image_token

from llava_lp.model.language_model.llava_llama import (
    IMAGE_SPARSE_LOSS_WEIGHT,
    TEXT_SPARSE_LOSS_WEIGHT,
    VISION_PRUNER_ATTENTION_PRE_PRUNE_HEAD_REDUCTION,
    VISION_PRUNER_ATTENTION_PRE_PRUNE_KEEP_RATIO,
    LlavaLlamaForCausalLM_with_VisionPruner,
)

from PIL import Image


local_rank = None


def rank0_print(*args):
    if local_rank in [0, -1, None]:
        print(*args)


from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=0)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")
    # VisionPruner-specific
    vision_pruner_decoder_layer_idx: int = field(
        default=0,
        metadata={"help": "Index of the LLaMA decoder layer to deepcopy for LlavaImagePruner."}
    )
    vision_pruner_train_mode: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "VisionPruner training mode. "
                "Use 'llm_full' for LLM-initialized full training, "
                "'random_full' for fully reinitialized full training, "
                "'llm_freeze_q' for LLM-initialized training with Q frozen, "
                "'llm_freeze_k' for LLM-initialized training with K frozen, "
                "'llm_freeze_kv' for LLM-initialized training with K/V frozen, "
                "'llm_freeze_kv_random_rest' for LLM K/V frozen and all other weights reinitialized, "
                "'llm_freeze_v_ffn' for LLM-initialized training with V and FFN frozen, "
                "or 'llm_freeze_v_ffn_random_rest' for LLM V/FFN frozen and all other weights reinitialized. "
                "If omitted, vision_pruner_init_mode selects llm_full or random_full."
            )
        },
    )
    vision_pruner_init_mode: str = field(
        default="llm",
        metadata={
            "help": (
                "Legacy VisionPruner initialization mode used when vision_pruner_train_mode is omitted. "
                "'llm' starts from the selected LLM decoder layer. "
                "'random' reinitializes VisionPruner."
            )
        },
    )

@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    data_fraction: float = field(
        default=1.0,
        metadata={"help": "Fraction of the training data to use. Must be in (0, 1]."},
    )
    data_subset_seed: int = field(
        default=42,
        metadata={"help": "Seed used when selecting a data_fraction subset."},
    )
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    single_turn_only: bool = field(
        default=True,
        metadata={
            "help": (
                "Use only the first human/gpt exchange from each training sample. "
                "This makes VisionPruner training single-turn even when the source data is multi-turn."
            )
        },
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    group_by_modality_length: bool = field(default=False)
    mm_projector_lr: Optional[float] = field(default=None)
    vision_pruner_score_noise_std: float = field(
        default=VISION_PRUNER_DEFAULT_SCORE_NOISE_STD,
        metadata={
            "help": (
                "Fixed standard deviation for Gaussian noise added to VisionPruner scores during "
                "training. The default is 0.1 / z_0.975 ~= 0.051021, so about 5% of sampled "
                "noise values have absolute magnitude greater than or equal to 0.1."
            )
        },
    )
    vision_pruner_score_noise_start: float = field(
        default=2.0,
        metadata={
            "help": (
                "Deprecated compatibility argument. VisionPruner score noise now uses the fixed "
                "Gaussian std from vision_pruner_score_noise_std instead of a decaying schedule."
            )
        },
    )
    vision_pruner_score_noise_end: float = field(
        default=0.02,
        metadata={
            "help": (
                "Deprecated compatibility argument. VisionPruner score noise now uses the fixed "
                "Gaussian std from vision_pruner_score_noise_std instead of a decaying schedule."
            )
        },
    )
    vision_pruner_max_param_abs: float = field(
        default=1e6,
        metadata={
            "help": (
                "Abort VisionPruner training when any optimizer parameter exceeds this absolute "
                "value. Set to 0 to disable the finite-value explosion check."
            )
        },
    )


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(model, keys_to_match):
    named_params = model.named_parameters()
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    
    # Add buffers (excluding rotary_emb related ones that cause loading issues)
    keys_to_ignore = ['rotary_emb.inv_freq', 'rotary_emb.cos_cached', 'rotary_emb.sin_cached']
    named_buffers = model.named_buffers()
    for k, v in named_buffers:
        if any(key_match in k for key_match in keys_to_match):
            if not any(ignore_key in k for ignore_key in keys_to_ignore):
                to_return[k] = v.detach().cpu().clone()
            
    return to_return


VISION_PRUNER_INIT_ALIASES = {
    "llm": "llm",
    "llm_init": "llm",
    "default": "llm",
    "random": "random",
    "random_init": "random",
    "scratch": "random",
}

@dataclass
class VisionPrunerTrainingMode:
    name: str
    reinitialize: bool
    preserve_components: Set[str]
    frozen_components: Set[str]


VISION_PRUNER_TRAIN_MODES = {
    "llm_full": VisionPrunerTrainingMode(
        name="llm_full",
        reinitialize=False,
        preserve_components=set(),
        frozen_components=set(),
    ),
    "random_full": VisionPrunerTrainingMode(
        name="random_full",
        reinitialize=True,
        preserve_components=set(),
        frozen_components=set(),
    ),
    "llm_freeze_q": VisionPrunerTrainingMode(
        name="llm_freeze_q",
        reinitialize=False,
        preserve_components=set(),
        frozen_components={"q_proj"},
    ),
    "llm_freeze_k": VisionPrunerTrainingMode(
        name="llm_freeze_k",
        reinitialize=False,
        preserve_components=set(),
        frozen_components={"k_proj"},
    ),
    "llm_freeze_kv": VisionPrunerTrainingMode(
        name="llm_freeze_kv",
        reinitialize=False,
        preserve_components=set(),
        frozen_components={"k_proj", "v_proj"},
    ),
    "llm_freeze_kv_random_rest": VisionPrunerTrainingMode(
        name="llm_freeze_kv_random_rest",
        reinitialize=True,
        preserve_components={"k_proj", "v_proj"},
        frozen_components={"k_proj", "v_proj"},
    ),
    "llm_freeze_v_ffn": VisionPrunerTrainingMode(
        name="llm_freeze_v_ffn",
        reinitialize=False,
        preserve_components=set(),
        frozen_components={"v_proj", "ffn"},
    ),
    "llm_freeze_v_ffn_random_rest": VisionPrunerTrainingMode(
        name="llm_freeze_v_ffn_random_rest",
        reinitialize=True,
        preserve_components={"v_proj", "ffn"},
        frozen_components={"v_proj", "ffn"},
    ),
}
VISION_PRUNER_TRAIN_MODE_ALIASES = {
    "llm_full": "llm_full",
    "random_full": "random_full",
    "llm_freeze_q": "llm_freeze_q",
    "llm_freeze_k": "llm_freeze_k",
    "llm_freeze_kv": "llm_freeze_kv",
    "llm_freeze_kv_random_rest": "llm_freeze_kv_random_rest",
    "llm_freeze_v_ffn": "llm_freeze_v_ffn",
    "llm_freeze_v_ffn_random_rest": "llm_freeze_v_ffn_random_rest",
}


def _copy_vision_pruner_training_mode(mode: VisionPrunerTrainingMode) -> VisionPrunerTrainingMode:
    return VisionPrunerTrainingMode(
        name=mode.name,
        reinitialize=mode.reinitialize,
        preserve_components=set(mode.preserve_components),
        frozen_components=set(mode.frozen_components),
    )


def _parse_vision_pruner_train_mode(train_mode: Optional[str]) -> Optional[VisionPrunerTrainingMode]:
    if train_mode is None:
        return None

    train_mode = train_mode.lower().strip()
    if not train_mode or train_mode == "none":
        return None
    if train_mode not in VISION_PRUNER_TRAIN_MODE_ALIASES:
        raise ValueError(
            f"Unknown vision_pruner_train_mode='{train_mode}'. "
            f"Expected one of {sorted(VISION_PRUNER_TRAIN_MODE_ALIASES)}."
        )

    canonical_mode = VISION_PRUNER_TRAIN_MODE_ALIASES[train_mode]
    return _copy_vision_pruner_training_mode(VISION_PRUNER_TRAIN_MODES[canonical_mode])


def _validate_vision_pruner_init_mode(init_mode: str) -> str:
    init_mode = init_mode.lower().strip()
    if init_mode not in VISION_PRUNER_INIT_ALIASES:
        raise ValueError(
            f"Unknown vision_pruner_init_mode='{init_mode}'. "
            f"Expected one of {sorted(VISION_PRUNER_INIT_ALIASES)}."
        )
    return VISION_PRUNER_INIT_ALIASES[init_mode]


def _resolve_vision_pruner_training_mode(
    train_mode: Optional[str],
    init_mode: str,
) -> VisionPrunerTrainingMode:
    parsed_train_mode = _parse_vision_pruner_train_mode(train_mode)
    if parsed_train_mode is not None:
        return parsed_train_mode

    init_mode = _validate_vision_pruner_init_mode(init_mode)
    if init_mode == "llm":
        return _copy_vision_pruner_training_mode(VISION_PRUNER_TRAIN_MODES["llm_full"])
    return _copy_vision_pruner_training_mode(VISION_PRUNER_TRAIN_MODES["random_full"])


def _is_vision_pruner_component_param(name: str, components: Set[str]) -> bool:
    normalized_name = f".{name}."
    return (
        ("q_proj" in components and ".self_attn.q_proj." in normalized_name)
        or ("k_proj" in components and ".self_attn.k_proj." in normalized_name)
        or ("v_proj" in components and ".self_attn.v_proj." in normalized_name)
        or ("ffn" in components and ".mlp." in normalized_name)
    )


def _format_vision_pruner_components(components: Set[str]) -> str:
    ordered_components = [component for component in ("q_proj", "k_proj", "v_proj", "ffn") if component in components]
    return ",".join(ordered_components) if ordered_components else "none"


def _clone_vision_pruner_component_state(vision_pruner, components: Set[str]) -> Dict[str, torch.Tensor]:
    return {
        name: param.detach().cpu().clone()
        for name, param in vision_pruner.named_parameters()
        if _is_vision_pruner_component_param(name, components)
    }


def _restore_vision_pruner_component_state(vision_pruner, component_state: Dict[str, torch.Tensor]) -> int:
    restored_count = 0
    named_params = dict(vision_pruner.named_parameters())
    for name, value in component_state.items():
        if name not in named_params:
            raise ValueError(f"Cannot restore preserved VisionPruner parameter '{name}': parameter not found.")
        param = named_params[name]
        if param.shape != value.shape:
            raise ValueError(
                f"Cannot restore preserved VisionPruner parameter '{name}': "
                f"shape mismatch preserved={tuple(value.shape)} current={tuple(param.shape)}."
            )
        with torch.no_grad():
            param.copy_(value.to(device=param.device, dtype=param.dtype))
        restored_count += param.numel()
    return restored_count


def _register_vision_pruner_gradient_nan_guard(vision_pruner) -> int:
    hook_count = 0
    for name, param in vision_pruner.named_parameters():
        if not param.requires_grad:
            continue

        def check_gradient(grad, param_name=name):
            if grad is None:
                return grad
            if torch.is_tensor(grad) and grad.is_floating_point():
                finite = torch.isfinite(grad)
                if not finite.all():
                    nonfinite_count = int((~finite).sum().item())
                    raise FloatingPointError(
                        f"[vision_pruner] non-finite gradient in vision_pruner.{param_name}: "
                        f"{nonfinite_count}/{grad.numel()}"
                    )
            return grad

        param.register_hook(check_gradient)
        hook_count += 1
    return hook_count


def _reinitialize_vision_pruner_parameters(model):
    vision_pruner = model.get_model().get_vision_pruner()
    if vision_pruner is None:
        raise ValueError("VisionPruner must be initialized before its parameters can be reinitialized.")

    init_weights = getattr(model, "_init_weights", None)
    if callable(init_weights):
        vision_pruner.apply(init_weights)
    else:
        for module in vision_pruner.modules():
            reset_parameters = getattr(module, "reset_parameters", None)
            if callable(reset_parameters):
                reset_parameters()

    for module in vision_pruner.modules():
        module_name = module.__class__.__name__.lower()
        if isinstance(module, torch.nn.LayerNorm) or "rmsnorm" in module_name:
            if getattr(module, "weight", None) is not None:
                module.weight.data.fill_(1.0)
            if getattr(module, "bias", None) is not None:
                module.bias.data.zero_()


def _normalize_vision_pruner_key(key: str) -> str:
    if "vision_pruner" not in key:
        return key
    return "vision_pruner" + key.split("vision_pruner", 1)[1]


def _summarize_vision_pruner_delta(initial_state: Dict[str, torch.Tensor],
                                   current_state: Dict[str, torch.Tensor]) -> Dict[str, float]:
    normalized_initial = {
        _normalize_vision_pruner_key(k): v.float()
        for k, v in initial_state.items()
        if "vision_pruner" in k and torch.is_tensor(v) and v.is_floating_point()
    }
    normalized_current = {
        _normalize_vision_pruner_key(k): v.float()
        for k, v in current_state.items()
        if "vision_pruner" in k and torch.is_tensor(v) and v.is_floating_point()
    }

    changed_tensors = 0
    total_tensors = 0
    total_params = 0
    l2_delta_sq = 0.0
    max_abs_delta = 0.0
    mean_abs_delta_num = 0.0

    for key, current in normalized_current.items():
        initial = normalized_initial.get(key)
        if initial is None or initial.shape != current.shape:
            continue
        total_tensors += 1
        delta = current - initial
        abs_delta = delta.abs()
        numel = delta.numel()
        total_params += numel
        tensor_max = abs_delta.max().item() if numel > 0 else 0.0
        if tensor_max > 0:
            changed_tensors += 1
        max_abs_delta = max(max_abs_delta, tensor_max)
        l2_delta_sq += delta.pow(2).sum().item()
        mean_abs_delta_num += abs_delta.sum().item()

    return {
        "matched_tensors": total_tensors,
        "changed_tensors": changed_tensors,
        "matched_params": total_params,
        "l2_delta": l2_delta_sq ** 0.5,
        "max_abs_delta": max_abs_delta,
        "mean_abs_delta": mean_abs_delta_num / total_params if total_params else 0.0,
    }


def save_vision_pruner_checkpoint(trainer: transformers.Trainer, output_dir: str) -> Dict[str, torch.Tensor]:
    weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model, ['vision_pruner'])
    torch.save(weight_to_save, os.path.join(output_dir, 'vision_pruner.bin'))

    initial_state = getattr(trainer, "vision_pruner_initial_state", None)
    if initial_state is not None:
        delta_stats = _summarize_vision_pruner_delta(initial_state, weight_to_save)
        with open(os.path.join(output_dir, "vision_pruner_delta_stats.json"), "w") as f:
            json.dump(delta_stats, f, indent=2)
        rank0_print(
            "[vision_pruner] saved "
            f"{os.path.join(output_dir, 'vision_pruner.bin')} "
            f"delta_l2={delta_stats['l2_delta']:.6e} "
            f"max_abs_delta={delta_stats['max_abs_delta']:.6e} "
            f"mean_abs_delta={delta_stats['mean_abs_delta']:.6e} "
            f"changed_tensors={delta_stats['changed_tensors']}/{delta_stats['matched_tensors']}"
        )
    else:
        rank0_print(f"[vision_pruner] saved {os.path.join(output_dir, 'vision_pruner.bin')}")

    return weight_to_save


def _clean_vision_pruner_state_dict(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        if "vision_pruner" in key:
            cleaned_key = "model.vision_pruner" + key.split("vision_pruner", 1)[1]
            cleaned_state_dict[cleaned_key] = value
        else:
            cleaned_state_dict[key] = value
    return cleaned_state_dict


def load_vision_pruner_checkpoint_into_model(model, checkpoint_path: str):
    rank0_print(f"[vision_pruner] loading {checkpoint_path}")
    vision_pruner_state = torch.load(checkpoint_path, map_location="cpu")
    cleaned_state = _clean_vision_pruner_state_dict(vision_pruner_state)
    load_result = model.load_state_dict(cleaned_state, strict=False)

    missing_vp = [key for key in load_result.missing_keys if "vision_pruner" in key]
    unexpected_vp = [key for key in load_result.unexpected_keys if "vision_pruner" in key]
    rank0_print(
        "[vision_pruner] loaded "
        f"tensors={len(cleaned_state)} "
        f"missing_vision_pruner_keys={len(missing_vp)} "
        f"unexpected_vision_pruner_keys={len(unexpected_vp)}"
    )
    return load_result


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " + sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(sources: Sequence[str], data_args: DataArguments) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources


def preprocess_llama_2(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            source = source[1:]
        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    if has_image:
        input_ids = torch.stack(
            [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        _total_len = int(target.ne(tokenizer.pad_token_id).sum())
        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break
            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def _tokenized_len_for_text_prune_mask(
    text: str,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool,
) -> int:
    if has_image:
        return len(tokenizer_image_token(text, tokenizer))
    return len(tokenizer(text).input_ids)


def _build_user_text_prune_mask(
    conversation: str,
    source: Sequence[Dict[str, str]],
    tokenizer: transformers.PreTrainedTokenizer,
    input_ids: torch.Tensor,
    has_image: bool,
) -> torch.Tensor:
    text_prune_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    search_start = 0

    for sentence in source:
        if sentence.get("from") != "human":
            continue

        value = sentence["value"]
        if isinstance(value, tuple):
            value = value[0]
        start = conversation.find(value, search_start)
        if start < 0:
            start = conversation.find(value)
        if start < 0:
            continue

        end = start + len(value)
        start_idx = _tokenized_len_for_text_prune_mask(
            conversation[:start].rstrip(), tokenizer, has_image
        )
        end_idx = _tokenized_len_for_text_prune_mask(
            conversation[:end], tokenizer, has_image
        )
        start_idx = max(0, min(start_idx, text_prune_mask.numel()))
        end_idx = max(start_idx, min(end_idx, text_prune_mask.numel()))
        text_prune_mask[start_idx:end_idx] = True
        search_start = end

    special_token_ids = {
        token_id
        for token_id in (
            tokenizer.pad_token_id,
            tokenizer.bos_token_id,
            tokenizer.eos_token_id,
            tokenizer.unk_token_id,
        )
        if token_id is not None
    }

    for idx in torch.where(text_prune_mask)[0].tolist():
        token_id = int(input_ids[idx].item())
        if token_id == IMAGE_TOKEN_INDEX or token_id in special_token_ids:
            text_prune_mask[idx] = False
            continue
        if tokenizer.decode([token_id], skip_special_tokens=False).strip() == "":
            text_prune_mask[idx] = False

    return text_prune_mask


def preprocess_v1(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    conversations = []
    normalized_sources = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            source = source[1:]
        normalized_sources.append(source)
        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    if has_image:
        input_ids = torch.stack(
            [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    text_prune_masks = torch.stack([
        _build_user_text_prune_mask(conversation, source, tokenizer, input_id, has_image)
        for conversation, source, input_id in zip(conversations, normalized_sources, input_ids)
    ])
    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        _total_len = int(target.ne(tokenizer.pad_token_id).sum())
        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break
            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2
            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets, text_prune_mask=text_prune_masks)


def preprocess_plain(sources: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=targets)


def preprocess(sources: Sequence[str], tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)

    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


def _single_turn_conversations(conversations: Sequence[Dict[str, str]]) -> Sequence[Dict[str, str]]:
    first_human_idx = None
    for idx, sentence in enumerate(conversations):
        if sentence.get("from", "").lower() == "human":
            first_human_idx = idx
            break

    if first_human_idx is None:
        raise ValueError("Cannot build single-turn sample: no human message found.")

    first_assistant_idx = None
    for idx in range(first_human_idx + 1, len(conversations)):
        if conversations[idx].get("from", "").lower() == "gpt":
            first_assistant_idx = idx
            break

    if first_assistant_idx is None:
        raise ValueError("Cannot build single-turn sample: no gpt message found after the first human message.")

    return [conversations[first_human_idx], conversations[first_assistant_idx]]


def _make_single_turn_sample(sample: Dict) -> Dict:
    sample = copy.deepcopy(sample)
    sample["conversations"] = _single_turn_conversations(sample["conversations"])
    return sample


class LazySupervisedDataset(Dataset):

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        if not 0 < data_args.data_fraction <= 1:
            raise ValueError(f"data_fraction must be in (0, 1], got {data_args.data_fraction}")

        if data_args.data_fraction < 1:
            original_size = len(list_data_dict)
            subset_size = max(1, int(original_size * data_args.data_fraction))
            rng = random.Random(data_args.data_subset_seed)
            selected_indices = set(rng.sample(range(original_size), subset_size))
            list_data_dict = [sample for idx, sample in enumerate(list_data_dict) if idx in selected_indices]
            rank0_print(
                f"Using {len(list_data_dict):,}/{original_size:,} training samples "
                f"({data_args.data_fraction:.2%}, seed={data_args.data_subset_seed})"
            )

        if data_args.single_turn_only:
            original_turn_counts = [len(sample["conversations"]) for sample in list_data_dict]
            list_data_dict = [_make_single_turn_sample(sample) for sample in list_data_dict]
            truncated_count = sum(turn_count > 2 for turn_count in original_turn_counts)
            rank0_print(
                "Using single-turn training samples: "
                f"kept first human/gpt exchange for {truncated_count:,}/{len(list_data_dict):,} multi-turn samples"
            )

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(sources, self.tokenizer, has_image=('image' in self.list_data_dict[i]))
        if isinstance(i, int):
            sample_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])
            if "text_prune_mask" in data_dict:
                sample_dict["text_prune_mask"] = data_dict["text_prune_mask"][0]
            data_dict = sample_dict

        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        if all("text_prune_mask" in instance for instance in instances):
            text_prune_mask = torch.nn.utils.rnn.pad_sequence(
                [instance["text_prune_mask"] for instance in instances],
                batch_first=True,
                padding_value=False,
            )
            batch["text_prune_mask"] = text_prune_mask[:, :self.tokenizer.model_max_length]
        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images
        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Save the VisionPruner adapter without materializing the full model state."""

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    if trainer.args.should_save:
        trainer._save(output_dir)  # noqa


def train(attn_implementation=None):
    global local_rank

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    if training_args.vision_pruner_score_noise_std < 0:
        raise ValueError(
            "vision_pruner_score_noise_std must be non-negative, "
            f"got {training_args.vision_pruner_score_noise_std}"
        )
    if training_args.vision_pruner_max_param_abs < 0:
        raise ValueError(
            "vision_pruner_max_param_abs must be non-negative, "
            f"got {training_args.vision_pruner_max_param_abs}"
        )
    vision_pruner_mode = _resolve_vision_pruner_training_mode(
        model_args.vision_pruner_train_mode,
        model_args.vision_pruner_init_mode,
    )
    os.environ["LLAVA_ATTENTION_SCORE_PRUNING"] = "1"
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type
            )
        ))

    if model_args.vision_tower is not None:
        model = LlavaLlamaForCausalLM_with_VisionPruner.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            **bnb_model_from_pretrained_args
        )
    else:
        raise ValueError("vision_tower must be specified for VisionPruner training.")

    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(_module, _input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    # ── Vision modules setup ─────────────────────────────────────────────────
    model.get_model().initialize_vision_modules(model_args=model_args, fsdp=training_args.fsdp)

    vision_tower = model.get_vision_tower()
    vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

    data_args.image_processor = vision_tower.image_processor
    data_args.is_multimodal = True

    model.config.image_aspect_ratio = data_args.image_aspect_ratio
    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length
    model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
    model.config.vision_pruner_decoder_layer_idx = model_args.vision_pruner_decoder_layer_idx
    model.config.vision_pruner_train_mode = vision_pruner_mode.name
    model.config.vision_pruner_init_mode = "random" if vision_pruner_mode.reinitialize else "llm"
    model.config.vision_pruner_preserve_components = _format_vision_pruner_components(
        vision_pruner_mode.preserve_components
    )
    model.config.vision_pruner_freeze_components = _format_vision_pruner_components(
        vision_pruner_mode.frozen_components
    )
    model.config.vision_pruner_score_noise_std = training_args.vision_pruner_score_noise_std
    model.config.vision_pruner_score_noise_tail_threshold = VISION_PRUNER_SCORE_NOISE_TAIL_THRESHOLD
    model.config.vision_pruner_score_noise_start = training_args.vision_pruner_score_noise_start
    model.config.vision_pruner_score_noise_end = training_args.vision_pruner_score_noise_end
    model.config.vision_pruner_max_param_abs = training_args.vision_pruner_max_param_abs
    model.config.vision_pruner_image_sparse_loss_weight = IMAGE_SPARSE_LOSS_WEIGHT
    model.config.vision_pruner_text_sparse_loss_weight = TEXT_SPARSE_LOSS_WEIGHT
    model.config.vision_pruner_attention_pre_prune_keep_ratio = VISION_PRUNER_ATTENTION_PRE_PRUNE_KEEP_RATIO
    model.config.vision_pruner_attention_pre_prune_layer = None
    model.config.vision_pruner_attention_pre_prune_head_reduction = VISION_PRUNER_ATTENTION_PRE_PRUNE_HEAD_REDUCTION
    model.config.vision_pruner_prune_question_text_during_training = True
    model.config.vision_pruner_base_model_name_or_path = model_args.model_name_or_path
    model.config.vision_pruner_vision_tower = model_args.vision_tower

    # ── Freeze everything ────────────────────────────────────────────────────
    model.requires_grad_(False)

    # ── Initialize VisionPruner from a decoder layer ─────────────────────────
    # deepcopy happens here (before DeepSpeed sharding) on the full CPU model
    decoder_layer_idx = model_args.vision_pruner_decoder_layer_idx
    rank0_print(
        f"Initializing VisionPruner from decoder layer {decoder_layer_idx} "
        f"(train_mode={vision_pruner_mode.name})..."
    )
    model.get_model().initialize_vision_pruner(model.model.layers[decoder_layer_idx])
    preserved_component_state = {}
    if vision_pruner_mode.preserve_components:
        preserved_component_state = _clone_vision_pruner_component_state(
            model.get_model().vision_pruner,
            vision_pruner_mode.preserve_components,
        )
        rank0_print(
            "Preserving LLM-initialized VisionPruner components before reinit: "
            f"{_format_vision_pruner_components(vision_pruner_mode.preserve_components)} "
            f"({sum(t.numel() for t in preserved_component_state.values()):,} params)"
        )
    if vision_pruner_mode.reinitialize:
        rank0_print("Reinitializing VisionPruner parameters with the model initializer...")
        _reinitialize_vision_pruner_parameters(model)
    if preserved_component_state:
        restored_param_count = _restore_vision_pruner_component_state(
            model.get_model().vision_pruner,
            preserved_component_state,
        )
        rank0_print(
            "Restored preserved LLM-initialized VisionPruner components: "
            f"{_format_vision_pruner_components(vision_pruner_mode.preserve_components)} "
            f"({restored_param_count:,} params)"
        )
    model.get_model().vision_pruner.score_noise_std = training_args.vision_pruner_score_noise_std
    model.get_model().vision_pruner.score_noise_variance = training_args.vision_pruner_score_noise_std ** 2
    model.get_model().vision_pruner.score_noise_tail_threshold = VISION_PRUNER_SCORE_NOISE_TAIL_THRESHOLD
    model.get_model().vision_pruner.score_noise_last_std = 0.0
    model.get_model().vision_pruner.score_noise_last_abs_ge_threshold_ratio = 0.0
    model.get_model().vision_pruner.to(dtype=torch.float32, device=training_args.device)
    # Train every parameter inside vision_pruner. The rest of the model remains frozen
    # because model.requires_grad_(False) was applied before the pruner was re-enabled.
    frozen_param_count = 0
    frozen_tensor_names = []
    for name, param in model.get_model().vision_pruner.named_parameters():
        if _is_vision_pruner_component_param(name, vision_pruner_mode.frozen_components):
            param.requires_grad_(False)
            frozen_param_count += param.numel()
            frozen_tensor_names.append(name)
        else:
            param.requires_grad_(True)

    nan_guard_hook_count = _register_vision_pruner_gradient_nan_guard(model.get_model().vision_pruner)

    initial_vision_pruner_state = get_mm_adapter_state_maybe_zero_3(model, ['vision_pruner'])

    if vision_pruner_mode.frozen_components:
        rank0_print(
            "Frozen VisionPruner components: "
            f"{_format_vision_pruner_components(vision_pruner_mode.frozen_components)} "
            f"({frozen_param_count:,} params)"
        )
        for name in frozen_tensor_names:
            rank0_print(f"  frozen: vision_pruner.{name}")
    rank0_print(f"VisionPruner gradient NaN guard registered on {nan_guard_hook_count} trainable tensors.")

    rank0_print("Trainable parameters:")
    trainable_param_count = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_param_count += param.numel()
            rank0_print(f"  {name}: {param.shape}")
    rank0_print(f"Total trainable parameters: {trainable_param_count:,}")
    rank0_print(
        "VisionPruner config: "
        f"decoder_layer_idx={model.config.vision_pruner_decoder_layer_idx}, "
        f"train_mode={model.config.vision_pruner_train_mode}, "
        f"init_mode={model.config.vision_pruner_init_mode}, "
        f"preserve_components={model.config.vision_pruner_preserve_components}, "
        f"freeze_components={model.config.vision_pruner_freeze_components}, "
        f"score_noise_distribution=normal(mean=0,std={model.config.vision_pruner_score_noise_std:.6f}, "
        f"variance={model.config.vision_pruner_score_noise_std ** 2:.6f}, "
        f"tail_threshold={model.config.vision_pruner_score_noise_tail_threshold:.1f}), "
        f"sparse_loss_weights=image:{model.config.vision_pruner_image_sparse_loss_weight:.1f},"
        f"text:{model.config.vision_pruner_text_sparse_loss_weight:.1f}, "
        f"attention_pre_prune_keep_ratio={model.config.vision_pruner_attention_pre_prune_keep_ratio:.2f}, "
        f"attention_pre_prune_layer={model.config.vision_pruner_attention_pre_prune_layer}, "
        f"attention_pre_prune_head_reduction={model.config.vision_pruner_attention_pre_prune_head_reduction}, "
        "train_prunes_question_text_tokens=True, "
        f"max_param_abs={model.config.vision_pruner_max_param_abs}, "
        "pruning_uses_fixed_TAU_RHO, "
        "trainable_dtype=float32"
    )

    model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    # ── Data ─────────────────────────────────────────────────────────────────
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    # trainer = LLaVATrainer(
    #     model=model,
    #     tokenizer=tokenizer,
    #     args=training_args,
    #     **data_module,
    # )
    trainer = VisionPrunerTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module,
    )
    trainer.vision_pruner_initial_state = initial_vision_pruner_state

    last_checkpoint = None
    if (os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir):
        from transformers.trainer_utils import get_last_checkpoint
        last_checkpoint = get_last_checkpoint(training_args.output_dir)

    trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.save_state()

    model.config.use_cache = True

    # ── Save: VisionPruner weights only ────────────────────────────
    safe_save_model_for_hf_trainer(trainer=trainer,
                                   output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
