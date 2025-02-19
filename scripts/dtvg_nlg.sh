#!/bin/bash

MODEL_NAME=t5-base
TASK_NAME=web_nlg
BATCH_SIZE=16
for seed in {42,}
do
for lr_1 in {1e-3,}
do
for lr_2 in {1e-3,1e-4,1e-2,}
do
python run_seq2seq.py \
   --output_dir "dtvg_${MODEL_NAME}_$seed/${TASK_NAME}_{$seed}_{$lr_1}_{$lr_2}" \
   --do_train True \
   --do_eval False \
   --do_test True \
   --warmup_steps 500 \
   --save_steps 1000 \
   --eval_steps 1000 \
   --max_steps 30000 \
   --evaluation_strategy "steps" \
   --multi_task True \
   --task_name $TASK_NAME \
   --eval_dataset_name $TASK_NAME \
   --test_dataset_name $TASK_NAME \
   --dataset_config_name "en" \
   --eval_dataset_config_name "en" \
   --test_dataset_config_name "en" \
   --learning_rate $lr_1 \
   --target_task_scale_learning_rate $lr_1 \
   --target_prompt_embedding_path "/${MODEL_NAME}/${TASK_NAME}_seed={$seed}" \
   --multi_task_names "mnli" "qnli" "qqp" "sst2" "superglue-record" "squad" \
   --source_task_scale_learning_rate $lr_2 \
   --prompt_embedding_path "/t5-base/mnli_seed={$seed}" "/t5-base/qnli_seed={$seed}" "/t5-base/qqp_seed={$seed}" "/t5-base/sst2_seed={$seed}" "/t5-base/superglue-record_seed={$seed}" "/t5-base/squad_seed={$seed}" \
   --model_name_or_path "/home/LAB/zhangpy/model/${MODEL_NAME}" \
   --tokenizer_name "/home/LAB/zhangpy/model/${MODEL_NAME}" \
   --overwrite_output_dir True \
   --init_prefix_from_vocab True \
   --prefix_tuning True \
   --prefix_dim 100 \
   --num_beams 1 \
   --generation_max_length 128 \
   --max_source_length 128 \
   --per_device_train_batch_size $BATCH_SIZE \
   --per_device_eval_batch_size $BATCH_SIZE \
   --split_validation_test True \
   --predict_with_generate True \
   --weight_decay 1e-5 \
   --greater_is_better True \
   --compute_memory True \
   --metric_for_best_model "average_metric" \
   --save_safetensors False \
   --only_save_best_checkpoint False \
   --load_best_model_at_end True
done done done
