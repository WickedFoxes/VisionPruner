import os
from .pruner import LlavaImagePruner

def build_vision_pruner(value_layer, context_layer, rotary_emb=None):
    if value_layer is None or context_layer is None:
        return None
    return LlavaImagePruner(value_layer, context_layer, rotary_emb=rotary_emb)
