# Dynamic Task Vector Grouping for Efficient Multi-Task Prompt Tuning

![Model](/pic/image.png)

In the paper,  we introduce the DTVG, a novel approach for addressing potential negative transfer in multi-task prompt tuning based on task prompt vectors. Compared to vanilla transfer of the soft prompt from all source tasks, we dynamically group a subset of source tasks and merge their task prompt vectors to avoid an unrelated source task inducing performance degradation of the target task. Extensive experiments demonstrate that DTVG effectively groups related source tasks to further optimize the performance of the target task.

## Quick links

1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Train](#license)
   - [Task Prompt Vector Learning](#task-prompt-vector-learning)
   - [Multi-Task Prompt Transfer](#multi-task-prompt-transfer)
4. [Baselines](#baselines)

## Requirements
Our framework requires Python 3.5 or higher. We do not support Python 2.X.
- Python 3.10
- PyTorch (currently tested on version 2.5.1)
- Transformers (version 4.45.2; unlikely to work with a different version)

## Installation
Run the following commands to clone the repository and install our framework:

Download this repository from [DTVG](https://anonymous.4open.science/r/DTVG-CD4E)

```
cd dtvg
pip install -r requirements.txt
```


## Train
<!-- Explain how to use your project -->
DTVG consists of two stages: 1) Task Prompt Vector Learning to obtain a tuned TPV for each source and target task and 2) Multi-Task Prompt Transfer to group source tasks' TPV and merge it with the target vector's TPV. Note that the first stage only needs to be performed once, while the second stage is iterative, and the source task group will be dynamically updated during the fine-tuning process of the target task.

### Task Prompt Vector Learning
We provide prompt_tuning_nlu.sh and prompt_tuning_nlg.sh in /scripts, with which you can easily run training on each source and target task to obatain task prompt vectors (TPV). 


```
bash /scripts/prompt_tuning_nlu.sh
bash /scripts/prompt_tuning_nlg.sh
```


### Multi-Task Prompt Transfer


We provide  dtvg_nlu.sh and  dtvg_nlg.sh in /scripts, with which you can easily run training on the target task. you need modify TASK_NAME in dtvg_nlu.sh or dtvg_nlg.sh if you need.
```
bash /scripts/dtvg_nlu.sh
bash /scripts/dtvg_nlg.sh
```

```
   --target_prompt_embedding_path "${MODEL_NAME}/${TARGET_TASK_NAME}_seed={$seed}" \
   --multi_task_names "mnli" "qnli" "qqp" "sst2" "superglue-record" "squad" \
   --prompt_embedding_path "${MODEL_NAME}/mnli_seed={$seed}" "${MODEL_NAME}/qnli_seed={$seed}" "${MODEL_NAME}/qqp_seed={$seed}" "${MODEL_NAME}/sst2_seed={$seed}" "${MODEL_NAME}/superglue-record_seed={$seed}" "${MODEL_NAME}/squad_seed={$seed}" \
```
Notely, Task Prompt Vector Learning must run firstly on sources and target task. Secondly, You need modify prompt_embedding_path and target_prompt_embedding_path for source tasks and target tasks, respectily. 

## Baselines
We provide SPoT.sh in /scripts, with which you can easily run SPoT baseline on the target task. you also need run [Task Prompt Vector Learning](#task-prompt-vector-learning) to obtain soft prompt firstly.
```
bash /scripts/SPoT.sh
```
