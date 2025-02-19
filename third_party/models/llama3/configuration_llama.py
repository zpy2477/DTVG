""" Llama model configuration """
from transformers.models.llama import LlamaConfig


class LlamaConfig(LlamaConfig):
    def __init__(self,
                 train_task_adapters=False,
                 prefix_tuning=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.train_task_adapters = train_task_adapters
        self.prefix_tuning = prefix_tuning
