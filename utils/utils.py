from third_party.models.t5 import T5LayerNorm
from peft import (AutoPeftConfig, AdapterController, Adapter, SoftPromptController, SoftPrompt)
import os
import regex as re
import logging
from dataclasses import fields
import torch.nn as nn
import json
import torch 

import sys
sys.path.append('..')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def create_dir(output_dir):
    """
    Checks whether to the output_dir already exists and creates it if not.
    Args:
      output_dir: path to the output_dir
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def get_peft_config(peft_args, data_args, training_args, config):
    if peft_args.train_task_adapters or peft_args.prefix_tuning or peft_args.bitfit:
        peft_config = AutoPeftConfig.get(
            peft_args.peft_config_name)
        if hasattr(config, "d_model"):
            peft_config.input_dim = config.d_model
        elif hasattr(config, "hidden_size"):
            peft_config.input_dim = config.hidden_size

        if peft_args.train_task_adapters:
            data_args.tasks = [data_args.task_name]
            peft_config.tasks = data_args.tasks
        peft_params = [field.name for field in fields(peft_args)]
        for p in peft_params:
            if hasattr(peft_args, p) and hasattr(peft_config, p) and\
                    getattr(peft_args, p) is not None:
                setattr(peft_config, p, getattr(peft_args, p))
            else:
                logger.warning(
                    f"({peft_config.__class__.__name__}) doesn't have a `{p}` attribute")
                
        peft_config.tasks = data_args.task_name # type: ignore
        peft_config.device = training_args.device
        peft_config.is_logger = training_args.is_logger
        if hasattr(config, "d_model"):
            peft_config.d_model = config.d_model
        elif hasattr(config, "hidden_size"):
            peft_config.d_model = config.hidden_size
        peft_config.task_name = data_args.task_name # type: ignore
        peft_config.multi_task_names = config.multi_task_names
        peft_config.multi_task = config.multi_task
        peft_config.output_dir = training_args.output_dir # type: ignore
        peft_config.prefix_tuning = config.prefix_tuning # type: ignore
        peft_config.prompt_embedding_path = config.prompt_embedding_path # type: ignore
        peft_config.target_prompt_embedding_path = config.target_prompt_embedding_path # type: ignore
        peft_config.load_target_task_name = config.load_target_task_name
        
    else:
        peft_config = None
    return peft_config


def freeze_params(model: nn.Module):
    """Set requires_grad=False for each of model.parameters()"""
    for par in model.parameters():
        par.requires_grad = False


def freeze_model_params(model, peft_args, peft_config):
    """
    Freezes the model parameters based on the given setting in the arguments.
    Args:
      model: the given model.
      peft_args: defines the pefts arguments.
    """
    # If we are training pefts, we freeze all parameters except the
    # peft parameters like peft controllers.
    if peft_args.train_task_adapters or peft_args.prefix_tuning :
        freeze_params(model)
        for name, sub_module in model.named_modules():
            if isinstance(sub_module, (AdapterController, Adapter)):
                for param_name, param in sub_module.named_parameters():
                    param.requires_grad = True
                    

    # Unfreezes last linear layer of decoder.
    if peft_args.unfreeze_lm_head:
        for param in model.lm_head.parameters():
            param.requires_grad = True

    # Unfreezes layer norms.
    if peft_args.unfreeze_layer_norms:
        for name, sub_module in model.named_modules():
            if isinstance(sub_module, (T5LayerNorm, nn.LayerNorm)):
                # this will not consider layer norms inside pefts then.
                if len(name.split(".")) < 7:
                    for param_name, param in sub_module.named_parameters():
                        param.requires_grad = True

    if peft_args.prefix_tuning:
        freeze_params(model)
        for name, sub_module in model.named_modules():
            if isinstance(sub_module, (SoftPromptController, SoftPrompt)):
                for param_name, param in sub_module.named_parameters():
                    param.requires_grad = True

    # For bitfit we freeze the whole model except for the biases and the final classifier layer.
    if peft_args.bitfit:
        freeze_params(model)
        # unfreeze bias terms.
        for n, p in model.named_parameters():
            if ".bias" in n:
                p.requires_grad = True

        # unfreeze the classifier.
        for param in model.lm_head.parameters():
            param.requires_grad = True
        if peft_args.freeze_bitfit_lm_head:
            for n, param in model.lm_head.named_parameters():
                if "bias" in n:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        if peft_args.freeze_bitfit_lm_head_all:
            for n, param in model.lm_head.named_parameters():
                param.requires_grad = False


def get_peft_params_names(model):
    """
    Returns peft related parameters names.
    Args:
      model: the given model.
    """
    params_names = []
    for name, sub_module in model.named_modules():
        if isinstance(sub_module, (AdapterController, Adapter)):
            for param_name, param in sub_module.named_parameters():
                params_names.append(name+"."+param_name)
        if isinstance(sub_module, (SoftPromptController, SoftPrompt)):
            for param_name, param in sub_module.named_parameters():
                params_names.append(name+"."+param_name)
    return params_names


def get_layer_norm_params_names(model):
    """Returns the layer norms parameters.
    Args:
        model: the given model.
    """
    params_names = []
    for name, sub_module in model.named_modules():
        if isinstance(sub_module,  (T5LayerNorm, nn.LayerNorm)):
            for param_name, param in sub_module.named_parameters():
                params_names.append(name+"."+param_name)
    return params_names


def get_last_checkpoint(output_dir):
    if os.path.exists(os.path.join(output_dir, 'pytorch_model.bin')):
        return output_dir
    return None



def modify_model_after_init(model, training_args, peft_args, peft_config):
    # Freezes model parameters.
    freeze_model_params(model, peft_args, peft_config)

    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    logger.info(
        "***** Model Trainable Parameters {} *****".format(trainable_params))
    if training_args.print_num_parameters:
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info("##### Parameter name %s", name)
        total_lm_head_params = sum(p.numel()
                                   for p in model.lm_head.parameters())
        total_trainable_params = sum(p.numel()
                                     for p in model.parameters() if p.requires_grad)
        total_trainable_bias_params = sum(p.numel(
        ) for n, p in model.named_parameters() if p.requires_grad and n.endswith(".b"))
        total_trainable_layernorm_params = sum(p.numel() for n, p in model.named_parameters(
        ) if p.requires_grad and ".layer_norm.weight" in n)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info("Total trainable parameters %s", total_trainable_params)
        logger.info("Total trainable bias parameters %s",
                    total_trainable_bias_params)
        logger.info("Total trainable layer norm parameters %s",
                    total_trainable_layernorm_params)
        logger.info("Total parameters %s", total_params)
        t5_base_params = 222882048
        # total params since we have 8 task, it is Y = 1*BERT + 8*peftS, and final number is Y/BERT ("1.3x")
        total_params_ratio = ((total_params-t5_base_params)
                              * 8+t5_base_params)/t5_base_params
        total_trainable_params_percent = (
            total_trainable_params/t5_base_params)*100
        total_trainable_bias_params_percent = (
            total_trainable_bias_params/total_trainable_params)*100
        total_trainable_layernorm_params_percent = (
            total_trainable_layernorm_params/total_trainable_params)*100
        total_trainable_lm_head_params_percent = (
            total_lm_head_params/t5_base_params)*100
        logger.info("For pefts/prompt-tuning, total params %s",
                    total_params_ratio)
        logger.info("For intrinsic, total params %s",
                    total_params/t5_base_params)
        logger.info("Total trainable params %s",
                    total_trainable_params_percent)
        logger.info("Total trainable bias params %s",
                    total_trainable_bias_params_percent)
        logger.info("Total trainable layer norm params %s",
                    total_trainable_layernorm_params_percent)
        logger.info("Total lm_head params %s",
                    total_trainable_lm_head_params_percent)
    return model


def save_json(filepath, dictionary):
    with open(filepath, "w") as outfile:
        json.dump(dictionary, outfile)


def read_json(filepath):
    f = open(filepath,)
    return json.load(f)


# def save_training_config(config_file, output_dir):
#     json_data = read_json(config_file)
#     save_json(os.path.join(output_dir, "training_config.json"), json_data)

def save_prompts(model, output_dir):
    soft_prompt_controller = model.get_encoder().soft_prompt_controller
    soft_prompt_controller.save_soft_prompt(output_dir)