# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""

# import os
# os.environ["http_proxy"] = "http://192.168.1.103:7890"
# os.environ["https_proxy"] = "http://192.168.1.103:7890"

# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.
# from utils import modify_model_after_init, save_training_config, save_prompts
from utils import modify_model_after_init, save_prompts
import glob
from accelerate import Accelerator
from data import AutoPostProcessor
from third_party.models import T5Config, T5ForConditionalGeneration, LlamaConfig, LlamaForCausalLM
from dataclasses import dataclass, field
from options import PeftTrainingArguments, ModelArguments, DataTrainingArguments, TrainingArguments
from third_party.trainers import OurTrainer
from data import TaskDataCollatorForSeq2Seq
from data import AutoTask
from utils import get_peft_config
from transformers.trainer_utils import is_main_process, get_last_checkpoint
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    default_data_collator,
    set_seed,
    AdamW,
    get_linear_schedule_with_warmup
)
import transformers
from datasets import concatenate_datasets
import subprocess
import sys
import functools
import logging
import numpy as np
import torch
import os
import enum
from metrics.metrics import TASK_TO_METRICS
from metrics.metrics import build_compute_metrics_fn

os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'


logger = logging.getLogger(__name__)

class TaskType(str, enum.Enum):
    ENCODER_ONLY = "ENCODER_ONLY"
    DECODER_ONLY = "EDCODER_ONLY"
    ENCODER_DECODER = "ENCODER_DECODER"

def run_command(command):
    output = subprocess.getoutput(command)
    return output

def main():
    # initlize accelerator
    accelerator = Accelerator()
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments,
                               PeftTrainingArguments)) # type: ignore
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, adapter_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, adapter_args = parser.parse_args_into_dataclasses()
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        print("#### last_checkpoint ", last_checkpoint)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            '''
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
            '''
            pass
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(
        training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    def encoder_preprocess_function(examples, max_length=None, task_id=None, tokenizer=None):
        model_inputs = tokenizer(examples['source'],
                                 max_length=data_args.max_source_length,
                                 padding=padding,
                                 truncation=True)
        # Setup the tokenizer for targets (use 0,1,2 as label)
        labels = torch.tensor([int(i) for i in examples['target']])
        model_inputs["labels"] = labels

        model_inputs["task"] = examples['task']
        if task_id is not None:
            model_inputs["task_ids"] = [
                task_id for _ in examples['task']]

        return model_inputs

    def seq2seq_preprocess_function(examples, max_target_length=None, task_id=None, tokenizer=None):
        model_inputs = tokenizer(examples['source'],
                                 max_length=data_args.max_source_length,
                                 padding=padding,
                                 truncation=True)
        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples['target'], max_length=max_target_length, padding=padding, truncation=True)
        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
        model_inputs["labels"] = labels["input_ids"]

        model_inputs["task"] = examples['task']
        if task_id is not None:
            model_inputs["task_ids"] = [
                task_id for _ in examples['task']]

        return model_inputs

    def decoder_preprocess_function(examples, max_target_length=None, task_id=None, tokenizer=None):
        batch_size = len(examples['source'])
        inputs = [f"{x} Label : " for x in examples['source']]
        model_inputs = tokenizer(inputs)
        labels = tokenizer(examples['target'])
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i] + [tokenizer.pad_token_id]
            model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
            labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
            model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])

        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i]
            model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
                data_args.max_source_length - len(sample_input_ids)
            ) + sample_input_ids
            model_inputs["attention_mask"][i] = [0] * (data_args.max_source_length - len(sample_input_ids)) + model_inputs[
                "attention_mask"
            ][i]
            labels["input_ids"][i] = [-100] * (data_args.max_source_length - len(sample_input_ids)) + label_input_ids
            model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:data_args.max_source_length])
            model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:data_args.max_source_length])
            labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:data_args.max_source_length])
        model_inputs["labels"] = labels["input_ids"]

        model_inputs["task"] = examples['task']
        if task_id is not None:
            model_inputs["task_ids"] = [
                task_id for _ in examples['task']]

        return model_inputs

    def decoder_test_preprocess_function(examples, max_target_length=None, task_id=None, tokenizer=None):
        batch_size = len(examples['source'])
        inputs = [f"{x} Label : " for x in examples['source']]
        model_inputs = tokenizer(inputs)
        labels = tokenizer(examples['target'])
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i]
            model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (data_args.max_source_length - len(sample_input_ids)) + sample_input_ids
            model_inputs["attention_mask"][i] = [0] * (data_args.max_source_length - len(sample_input_ids)) + model_inputs["attention_mask"][i]
            labels["input_ids"][i] = [-100] * (data_args.max_source_length - len(label_input_ids)) + label_input_ids
            model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:data_args.max_source_length])
            model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:data_args.max_source_length])
            labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:data_args.max_source_length])
        model_inputs["labels"] = labels["input_ids"]

        model_inputs["task"] = examples['task']
        if task_id is not None:
            model_inputs["task_ids"] = [
                task_id for _ in examples['task']]

        return model_inputs
    
    def compute_metrics_encoder(eval_preds,processor,eval_metrics):
        preds, labels, data_info = eval_preds
        num_logits = preds.shape[-1]
        if num_logits == 1:
            preds = np.squeeze(preds)
        else:
            preds = np.argmax(preds, axis=1)
        result = {}
        for metric in eval_metrics:
            result.update(metric(preds, labels))
        return result
    
    def compute_metrics_seq2seq(eval_preds,post_processor,eval_metrics):
        preds, labels, data_info = eval_preds
        decoded_preds, decoded_labels = post_processor.process(preds, labels, data_info)
        print(f"decoded_preds is {decoded_preds}")
        print(f"decoded_labels is {decoded_labels}")
        result = {}
        for metric in eval_metrics:
            result.update(metric(decoded_preds, decoded_labels))
        return result

    def compute_metrics_decoder(eval_preds,post_processor,eval_metrics):
        preds, labels, data_info = eval_preds
        preds = preds[:, data_args.max_source_length + peft_config.prefix_dim:]
        decoded_preds, decoded_labels = post_processor.process(preds, labels, data_info)
        print(f"decoded_preds is {decoded_preds}")
        print(f"decoded_labels is {decoded_labels}")
        result = {}
        for metric in eval_metrics:
            result.update(metric(decoded_preds, decoded_labels))
        return result


    if any(x in model_args.model_name_or_path for x in ["bert", "roberta", "albert"]):
        logger.info(f"\n\nLoading enocder model from {model_args.model_name_or_path}.\n\n")
        task_type = TaskType.ENCODER_ONLY
        preprocess_function = encoder_preprocess_function
        metrics_fn = compute_metrics_encoder
    elif any(x in model_args.model_name_or_path for x in ["t5"]):
        logger.info(f"\n\nLoading seq2seq model from {model_args.model_name_or_path}.\n\n")
        task_type = TaskType.ENCODER_DECODER
        preprocess_function = seq2seq_preprocess_function
        metrics_fn = compute_metrics_seq2seq
    elif any(x in model_args.model_name_or_path for x in ["llama"]):
        logger.info(f"\n\nLoading decoder model from {model_args.model_name_or_path}.\n\n")
        task_type = TaskType.DECODER_ONLY
        preprocess_function = decoder_preprocess_function
        metrics_fn = compute_metrics_decoder
    else:
        raise NotImplementedError

    # Load config
    if task_type == TaskType.ENCODER_ONLY:
        pass
    elif task_type == TaskType.ENCODER_DECODER:
        if 't5' in model_args.model_name_or_path:
            config = T5Config.from_pretrained(
                model_args.config_name if model_args.config_name else model_args.model_name_or_path,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
            )
    elif task_type == TaskType.DECODER_ONLY:
        if 'llama' in model_args.model_name_or_path:
            config = LlamaConfig.from_pretrained(
                model_args.config_name if model_args.config_name else model_args.model_name_or_path,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
            )
    config.train_task_adapters = adapter_args.train_task_adapters # type: ignore
    config.prefix_tuning = adapter_args.prefix_tuning # type: ignore
    config.prefix_num = model_args.prefix_num # type: ignore
    config.num_target = len(data_args.task_name) # type: ignore
    config.prompt_embedding_path = model_args.prompt_embedding_path # type: ignore
    config.target_prompt_embedding_path = model_args.target_prompt_embedding_path # type: ignore
    config.multi_task_names = model_args.multi_task_names # type: ignore
    config.multi_task = model_args.multi_task # type: ignore
    config.load_target_task_name = model_args.load_target_task_name
    config.max_length = data_args.max_source_length
    config.num_beams = data_args.num_beams
    peft_config = get_peft_config(adapter_args, data_args, training_args, config)
    config.ablation = peft_config.ablation

    # config generation_max_length
    if task_type == TaskType.ENCODER_ONLY:
        pass
    elif task_type == TaskType.ENCODER_DECODER:
        config.max_length = training_args.generation_max_length
    elif task_type == TaskType.DECODER_ONLY:
        config.max_length = data_args.max_source_length + training_args.generation_max_length
        if config.prefix_tuning:
             config.max_length = config.max_length + peft_config.prefix_dim
        
    #load model
    if task_type == TaskType.ENCODER_ONLY:
        pass
    elif task_type == TaskType.ENCODER_DECODER:
        if 't5' in model_args.model_name_or_path:
            model = T5ForConditionalGeneration.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                peft_config=peft_config
            )
    elif task_type == TaskType.DECODER_ONLY:
        if 'llama' in model_args.model_name_or_path:
            model = LlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                peft_config=peft_config
            )        
            
    # Set tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id  # TODO: check if this is correct
    padding = "max_length" if data_args.pad_to_max_length else False
    tokenizer.padding_side = tokenizer.padding_side if task_type != TaskType.DECODER_ONLY else "left"

    # Resize the model token embeddings
    model.resize_token_embeddings(len(tokenizer))
    
    # Update the model token embedding for soft prompt
    if config.prefix_tuning: # type: ignore
        model.update_encoder_soft_prompt_controller() # type: ignore

    # freeze model parameters
    model = modify_model_after_init(model, training_args, adapter_args, peft_config=peft_config,)

    data_args.dataset_name = data_args.task_name
    data_args.eval_dataset_name = data_args.eval_dataset_name
    data_args.test_dataset_name = data_args.test_dataset_name
    data_args.dataset_config_name = data_args.dataset_config_name
    data_args.eval_dataset_config_name = data_args.eval_dataset_config_name
    data_args.test_dataset_config_name = data_args.test_dataset_config_name
    assert len(data_args.dataset_name) == len(data_args.dataset_config_name)
    if data_args.eval_dataset_name is not None:
        assert len(data_args.eval_dataset_name) == len(
            data_args.eval_dataset_config_name)
    if data_args.test_dataset_name is not None:
        assert len(data_args.test_dataset_name) == len(
            data_args.test_dataset_config_name)

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False


    column_names = ['source', 'target', 'task']
    performance_metrics = {}
    if training_args.do_train:
        # Load datasets from files if your target datasets are not in huggingface datasets.
        if data_args.train_files is not None:
            train_datasets = [AutoTask.get(dataset_name,
                                           dataset_config_name,
                                           seed=training_args.data_seed).get(
                split="train",
                split_validation_test=training_args.split_validation_test,
                add_prefix=False if adapter_args.train_task_adapters else True,
                n_obs=data_args.max_train_samples, lang=data_args.lang_name, file_name=train_file)
                for dataset_name, dataset_config_name, train_file
                in zip(data_args.dataset_name, data_args.dataset_config_name, data_args.train_files)]
        else:
            train_datasets = [AutoTask.get(dataset_name,
                                           dataset_config_name,
                                           seed=training_args.data_seed).get(
                split="train",
                split_validation_test=training_args.split_validation_test,
                add_prefix=False if adapter_args.train_task_adapters else True,
                n_obs=data_args.max_train_samples, lang=data_args.lang_name, file_name=data_args.train_file)
                for dataset_name, dataset_config_name
                in zip(data_args.dataset_name, data_args.dataset_config_name)]

        max_target_lengths = [AutoTask.get(dataset_name, dataset_config_name).get_max_target_length(
            tokenizer=tokenizer, default_max_length=data_args.max_target_length, )
            for dataset_name, dataset_config_name in zip(data_args.dataset_name, data_args.dataset_config_name)]

        for i, train_dataset in enumerate(train_datasets):
            train_datasets[i] = train_datasets[i].map(
                functools.partial(preprocess_function,max_target_length=max_target_lengths[i],tokenizer=tokenizer),
                batched=True,
                num_proc=data_args.preprocessing_num_workers, # type: ignore
                # if train_dataset != "superglue-record" else column_names+["answers"],
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache, # type: ignore
            )
        train_dataset = concatenate_datasets(train_datasets) # type: ignore

    if training_args.do_eval:
        if data_args.validation_files is not None:
            eval_datasets = {eval_dataset: AutoTask.get(eval_dataset, eval_dataset_config,
                                                        seed=training_args.data_seed).get(
                split="validation",
                split_validation_test=training_args.split_validation_test,
                add_prefix=False if adapter_args.train_task_adapters else True,
                n_obs=data_args.max_val_samples, lang=data_args.lang_name, file_name=validation_file)
                for eval_dataset, eval_dataset_config, validation_file in zip(data_args.eval_dataset_name, data_args.eval_dataset_config_name, data_args.validation_files)}
        else:
            eval_datasets = {eval_dataset: AutoTask.get(eval_dataset, eval_dataset_config,
                                                        seed=training_args.data_seed).get(
                split="validation",
                split_validation_test=training_args.split_validation_test,
                add_prefix=False if adapter_args.train_task_adapters else True,
                n_obs=data_args.max_val_samples, lang=data_args.lang_name, file_name=data_args.validation_file)
                for eval_dataset, eval_dataset_config in zip(data_args.eval_dataset_name, data_args.eval_dataset_config_name)}

        max_target_lengths = [AutoTask.get(dataset_name, dataset_config_name).get_max_target_length(
            tokenizer=tokenizer, default_max_length=data_args.max_target_length)
            for dataset_name, dataset_config_name in zip(data_args.eval_dataset_name, data_args.eval_dataset_config_name)]

        for k, name in enumerate(eval_datasets):
            eval_datasets[name] = eval_datasets[name].map(
                functools.partial(decoder_test_preprocess_function if task_type == TaskType.DECODER_ONLY else preprocess_function,max_target_length=max_target_length,tokenizer=tokenizer),
                batched=True,
                num_proc=data_args.preprocessing_num_workers, # type: ignore
                # if name != "superglue-record" else column_names+["answers"],
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache, # type: ignore
            )

    if training_args.do_test:
        if data_args.test_files is not None:
            test_datasets = {test_dataset: AutoTask.get(test_dataset, test_dataset_config,
                                                        seed=training_args.data_seed).get(
                split="test",
                split_validation_test=training_args.split_validation_test,
                add_prefix=False if adapter_args.train_task_adapters else True,
                n_obs=data_args.max_test_samples, lang=data_args.lang_name, file_name=test_file)
                for test_dataset, test_dataset_config, test_file in zip(data_args.test_dataset_name, data_args.test_dataset_config_name, data_args.test_files)}
        else:
            test_datasets = {test_dataset: AutoTask.get(test_dataset, test_dataset_config,
                                                        seed=training_args.data_seed).get(
                split="test",
                split_validation_test=training_args.split_validation_test,
                add_prefix=False if adapter_args.train_task_adapters else True,
                n_obs=data_args.max_test_samples, lang=data_args.lang_name, file_name=data_args.test_file)
                for test_dataset, test_dataset_config in zip(data_args.test_dataset_name, data_args.test_dataset_config_name)}

        max_target_lengths = [AutoTask.get(dataset_name, dataset_config_name).get_max_target_length(
            tokenizer=tokenizer, default_max_length=data_args.max_target_length)
            for dataset_name, dataset_config_name in zip(data_args.test_dataset_name, data_args.test_dataset_config_name)]
        for k, name in enumerate(test_datasets):
            test_datasets[name] = test_datasets[name].map(
                functools.partial(decoder_test_preprocess_function if task_type == TaskType.DECODER_ONLY else preprocess_function,
                    max_target_length=max_target_length,
                    tokenizer=tokenizer),
                batched=True,
                num_proc=data_args.preprocessing_num_workers, # type: ignore
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache, # type: ignore
            )

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = TaskDataCollatorForSeq2Seq(
            tokenizer, # type: ignore
            label_pad_token_id=label_pad_token_id, # type: ignore
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )

    # Get the metric function
    eval_metrics = [AutoTask.get(dataset_name, dataset_config_name).metric
            for dataset_name, dataset_config_name in zip(data_args.dataset_name, data_args.dataset_config_name)]
    post_processors = [AutoPostProcessor.get(dataset_name, tokenizer, data_args.ignore_pad_token_for_loss)
            for dataset_name, dataset_config_name in zip(data_args.dataset_name, data_args.dataset_config_name)]

    compute_metrics_fn = build_compute_metrics_fn(metrics_fn,
                data_args.eval_dataset_name, 
                eval_metrics, 
                post_processors)  if training_args.predict_with_generate else None
    print(compute_metrics_fn)

    # use two learning rate for different modules.
    if (model_args.target_task_scale_learning_rate is not None or model_args.source_task_scale_learning_rate is not None) and training_args.do_train:
        print(training_args.learning_rate)
        print(model_args.target_task_scale_learning_rate)
        print(model_args.source_task_scale_learning_rate)

        all_parameters = set(model.parameters())
        target_params = []
        source_params = []
        for name, param in model.named_parameters():
            if "soft_prompt_TV_norm2" in name :
                if "target" in name:
                    target_params.append(param)
                else :
                    source_params.append(param)
        target_params = set(target_params)
        source_params = set(source_params)
        non_scale_params = all_parameters - target_params - source_params
        non_scale_params = list(non_scale_params)
        target_params = list(target_params)
        source_params = list(source_params)
        optim = AdamW([
            {'params': non_scale_params},
            {'params': target_params, 'lr': model_args.target_task_scale_learning_rate},
            {'params': source_params, 'lr': model_args.source_task_scale_learning_rate},
        ], lr=training_args.learning_rate,) # type: ignore
        num_training_steps = training_args.max_steps if training_args.max_steps!= -1 else len(train_dataset) * training_args.num_train_epochs // \
            (training_args.gradient_accumulation_steps * training_args.per_device_train_batch_size) 
        scheduler = get_linear_schedule_with_warmup(
            optim, num_warmup_steps=training_args.warmup_steps, num_training_steps=num_training_steps
        )
        print(num_training_steps)
        # Initialize our Trainer
        trainer = OurTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_datasets if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            multi_task_compute_metrics=compute_metrics_fn,
            evaluation_metrics={task_name:TASK_TO_METRICS[task_name] for task_name in data_args.dataset_name},
            optimizers=(optim, scheduler)
        )
    else:
        trainer = OurTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_datasets if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            evaluation_metrics={task_name:TASK_TO_METRICS[task_name] for task_name in data_args.dataset_name},
            multi_task_compute_metrics=compute_metrics_fn,)
    
    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        if training_args.compute_time:
            torch.cuda.synchronize()  # wait for move to complete
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record() # type: ignore

        train_result = trainer.train(resume_from_checkpoint=checkpoint) # type: ignore

        if training_args.compute_time:
            end.record() # type: ignore
            torch.cuda.synchronize()  # wait for all_reduce to complete
            total_time = start.elapsed_time(end)/(1000*60)
            performance_metrics.update({"total_time in minutes ": total_time})

        # # By setting the `save_prefix_only` True, you only save the attentions as well as the prompt components only.
        if training_args.load_best_model_at_end:
            if training_args.save_prefix_only:
                save_prompts(trainer.model, output_dir=training_args.output_dir) # type: ignore
            else:
                # save all model parameters and tokenizers regardless of whether they are updated or not.
                trainer.save_model() # type: ignore

        train_metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(
                train_dataset)
        )
        train_metrics["train_samples"] = min(
            max_train_samples, len(train_dataset))
        trainer.log_metrics("train", train_metrics) # type: ignore
        trainer.save_metrics("train", train_metrics) # type: ignore

        if not training_args.save_prefix_only:
            trainer.save_state() # type: ignore

    if torch.cuda.is_available() and training_args.compute_memory:
        peak_memory = (torch.cuda.max_memory_allocated() / 1024 ** 2)/1000
        print(
            "Memory utilization",
            peak_memory,
            "GB"
        )
        performance_metrics.update({"peak_memory": peak_memory})
    if training_args.compute_memory or training_args.compute_time:
        trainer.save_metrics("performance", performance_metrics) # type: ignore


    logger.info("*"*100)
    
    # Validation
    results = {}
    if training_args.do_eval:
        if training_args.load_best_model_at_end:
            logger.info("*** Evaluate For Best Training***")
        else :
            logger.info("*** Evaluate For Last Training***")
            
        for task, eval_dataset in eval_datasets.items():
            metrics = trainer.evaluate(eval_dataset=eval_dataset)
            trainer.log_metrics("eval", metrics) # type: ignore
            trainer.save_metrics("eval", metrics) # type: ignore

    # Test
    if training_args.do_test:
        if training_args.load_best_model_at_end:
            logger.info("*** Test For Best Training***")
        else:
            logger.info("*** Test For Last Training***")
            
        # multi-task evaluations
        results = {}
        for task, test_dataset in test_datasets.items():
            metrics = trainer.evaluate(eval_dataset=test_dataset,metric_key_prefix="test")
            trainer.log_metrics("test", metrics) # type: ignore
            trainer.save_metrics("test", metrics) # type: ignore
            
    logger.info("*"*100)

    # Evaluate all checkpoints on all tasks if training_args.eval_all_at_last==True
    results = {}
    if training_args.eval_all_at_last:
        if training_args.do_eval:
            print("OK IN")
            for checkpoint_dir in glob.glob(os.path.join(training_args.output_dir, "checkpoint-*")):
                print(checkpoint_dir)
                # load models here
                if model_args.load_layer_norm is True and "layer_norm_bias.pt" in checkpoint_dir:
                    trainer.model.update_layer_norm_weights(checkpoint_dir)
                dev_metrics_all = {}
                dev_avg = []
                logger.info("*** Evaluate ***") 
                trainer.model.update_encoder_soft_promp_final(checkpoint_dir, 
                    task = "final" if model_args.multi_task and peft_config.ablation!=10 else task)
                for idx, (task, eval_dataset) in enumerate(eval_datasets.items()):
                    # update task_name to evaluate here
                    trainer.set_task_name(task)
                    metrics = trainer.evaluate(eval_dataset=eval_dataset)
                    trainer.log_metrics("eval", metrics) # type: ignore
                    trainer.save_metrics("eval", metrics) # type: ignore
                    dev_metrics_all[task] = metrics
                    main_metric = list(metrics.values())[0]
                    dev_avg.append(main_metric)

                results.setdefault(checkpoint_dir, {})
                results[checkpoint_dir]["dev_avg"] = np.mean(dev_avg)
                results[checkpoint_dir]["dev_each"] = dev_metrics_all

        # Test
        if training_args.do_test:
            logger.info("*** Test ***")
            for checkpoint_dir in glob.glob(os.path.join(training_args.output_dir, "checkpoint-*")):
                # load models here
                if model_args.load_layer_norm is True and "layer_norm_bias.pt" in checkpoint_dir:
                    trainer.model.update_layer_norm_weights(checkpoint_dir)

                test_metrics_all = {}
                test_avg = []
                trainer.model.update_encoder_soft_promp_final(checkpoint_dir, 
                    task = "final" if model_args.multi_task and peft_config.ablation!=10 else task)
                for idx, (task, test_dataset) in enumerate(test_datasets.items()):
                    # update task_name to evaluate here
                    trainer.set_task_name(task)
                    metrics = trainer.evaluate(eval_dataset=test_dataset,metric_key_prefix="test")
                    trainer.log_metrics("test", metrics) # type: ignore
                    trainer.save_metrics("test", metrics) # type: ignore
                    test_metrics_all[task] = metrics
                    main_metric = list(metrics.values())[0]
                    test_avg.append(main_metric)
                results.setdefault(checkpoint_dir, {})
                results[checkpoint_dir]["test_avg"] = np.mean(test_avg)
                results[checkpoint_dir]["test_each"] = test_metrics_all

    print(results)
    return results

if __name__ == "__main__":
    main()
