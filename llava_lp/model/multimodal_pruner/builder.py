import os
from .pruner import LlavaImagePruner

def build_vision_pruner(decoder_layer):
    if decoder_layer is None:
        return None
    return LlavaImagePruner(decoder_layer)