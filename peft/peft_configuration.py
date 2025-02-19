"""Implements the adapters and other parameter-efficient finetuning methods' configurations."""

from collections import OrderedDict
from dataclasses import dataclass

import torch.nn as nn


@dataclass
class AdapterConfig(object):
    """Implements the adapter configuration proposed by Houlsby et. al, 2019
    in https://arxiv.org/abs/1902.00751.
    We additionally pass all the configuration of parameter-efficient finetuning
    methods with this config."""
    add_layer_norm_before_adapter: bool = False
    add_layer_norm_after_adapter: bool = True
    non_linearity: str = "swish"
    task_reduction_factor: int = 16
    add_adapter_in_feed_forward = True
    add_adapter_in_self_attention = True
    hidden_dim = 128
    task_adapter_layers_encoder = None
    task_adapter_layers_decoder = None
    task_adapter_in_decoder = True
    intrinsic_dim = 100
    normalize_intrinsic_projections = False
    # This can be either random, or fastfood.
    intrinsic_projection = "random"
    steps = 1
    top_k = None
    
class PTConfig(object):
    # prefix-tuning parameters.
    prefix_dim = 100
    d_model = 512
    device = None
    prefix_random_range = 0.5
    prompt_embedding_path = None
    save_prompt_path =None
    init_prefix_from_vocab = True
    train_task_prompts = False
    tasks = None
    transfer_tasks = None
    ablation = None
    
class BitfitConfig(object):
    # BitFit configuration.
    bitfit = False

PEFT_CONFIG_MAPPING = OrderedDict([
("adapter", AdapterConfig),
("pt",PTConfig),
("bitfit",BitfitConfig)                                   
])


class AutoPeftConfig(nn.Module):
    """Generic Adapter config class to instantiate different adapter configs."""

    @classmethod
    def get(cls, config_name: str):
        config_name = "pt"
        if config_name in PEFT_CONFIG_MAPPING:
            return PEFT_CONFIG_MAPPING[config_name]()
        raise ValueError(
            "Unrecognized adapter config type identifier: {}. Should contain one of {}"
            .format(config_name, ", ".join(PEFT_CONFIG_MAPPING.keys())))
