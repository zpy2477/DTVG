#!/bin/bash

MODEL_NAME=t5-base
TASK_NAME=common_gen
for seed in {42,}
do
python run_seq2seq.py \
   --seed $seed \
   --output_dir "$MODEL_NAME/${TASK_NAME}_seed={$seed}" \
   --save_safetensors True \
   --only_save_best_checkpoint False \
   --do_train True \
   --do_eval False \
   --do_test True \
   --warmup_steps 500 \
   --num_train_epochs 20 \
   --evaluation_strategy 'epoch' \
   --save_strategy 'epoch' \
   --task_name $TASK_NAME \
   --eval_dataset_name $TASK_NAME \
   --test_dataset_name $TASK_NAME \
   --dataset_config_name "en" \
   --eval_dataset_config_name "en" \
   --test_dataset_config_name "en" \
   --model_name_or_path $MODEL_NAME \
   --tokenizer_name $MODEL_NAME \
   --greater_is_better True \
   --overwrite_output_dir True \
   --init_prefix_from_vocab True \
   --prefix_tuning True \
   --prefix_dim 100 \
   --generation_max_length 128 \
   --max_source_length 128 \
   --learning_rate 0.3 \
   --per_device_train_batch_size 32 \
   --per_device_eval_batch_size 32 \
   --split_validation_test True \
   --predict_with_generate True \
   --weight_decay 1e-5 \
   --compute_memory True \
   --metric_for_best_model "average_metric"
done
