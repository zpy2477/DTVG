#!/bin/bash
MODEL_NAME=llama-3.2-1B
TASK_NAME=cola
batch_size=16
lr=1e-3
for seed in {42,}
do
python run_seq2seq.py \
   --seed $seed \
   --output_dir "/llama-3.2-1B/cola_seed={42}" \
   --do_train False \
   --do_eval False \
   --do_test True \
   --warmup_steps 500 \
   --save_steps 1000 \
   --eval_steps 1000 \
   --max_steps 300000 \
   --evaluation_strategy "steps" \
   --prefix_tuning True \
   --prefix_dim 100 \
   --generation_max_length 20 \
   --max_source_length 256 \
   --learning_rate $lr \
   --per_device_train_batch_size $batch_size \
   --per_device_eval_batch_size $batch_size \
   --task_name $TASK_NAME \
   --eval_dataset_name $TASK_NAME \
   --test_dataset_name $TASK_NAME \
   --dataset_config_name "en" \
   --eval_dataset_config_name "en" \
   --test_dataset_config_name "en" \
   --model_name_or_path "/home/LAB/zhangpy/model/${MODEL_NAME}" \
   --tokenizer_name "/home/LAB/zhangpy/model/${MODEL_NAME}" \
   --greater_is_better True \
   --overwrite_output_dir True \
   --init_prefix_from_vocab True \
   --split_validation_test True \
   --predict_with_generate True \
   --weight_decay 1e-5 \
   --compute_memory True \
   --save_safetensors False \
   --only_save_best_checkpoint False \
   --load_best_model_at_end True \
   --metric_for_best_model "average_metric"
done
