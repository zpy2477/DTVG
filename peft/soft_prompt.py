""" soft prompt module"""
import json
from tkinter import NO
from torch import nn 
import os
import torch
import numpy as np # type: ignore
import glob

class SoftPrompt(nn.Module):
    def __init__(self, config, init_weight = None,task = None , need_train = False):

        """appends learned embedding to inputs
        Args:
            config : soft_prompt config
        """
        super().__init__()
        self.task = task
        self.config = config
        self.prefix_dim = config.prefix_dim
        self.d_model = config.d_model
        self.prefix_random_range = config.prefix_random_range
        self.device = config.device
        self.need_train = need_train
        
        ## same init prompt
        init_weight_self = init_weight.detach().clone() # type: ignore
        if self.need_train :
            self.soft_prompt = nn.parameter.Parameter(init_weight_self.to(self.device)) # type: ignore
        else :
            self.soft_prompt = init_weight_self.to(self.device)
        self.init_weight = init_weight.detach().clone().to(self.device) # type: ignore

    def get_soft_prompt(self):
        return self.soft_prompt

    def get_soft_prompt_task_vector(self):
        return self.soft_prompt - self.init_weight

    def save_soft_prompt(self, output_dir: str, task = None, soft_prompt = None, init_weight = None):
        
        task = task if task != None else self.task
        soft_prompt = soft_prompt if soft_prompt != None else self.soft_prompt
        init_weight = init_weight if init_weight != None else self.init_weight
        
        
        output_dir = os.path.join(output_dir,f"{task}_soft_prompt_set")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        soft_prompt_path = os.path.join(output_dir, task+".pt") # type: ignore
        soft_prompt_init_path = os.path.join(output_dir, task+"_init.pt") # type: ignore
        soft_prompt_task_vector_path = os.path.join(output_dir, task+"_task_vector.pt") # type: ignore

        
        torch.save(soft_prompt, soft_prompt_path)
        torch.save(init_weight, soft_prompt_init_path)
        torch.save(soft_prompt - init_weight, soft_prompt_task_vector_path)


    def load_soft_prompt_task_vector(self,path,task="final"):
        
        
        checkpoint_num = -1
        if "checkpoint" not in path:
            max_step_checkpoint = None
            for checkpoint_dir in glob.glob(os.path.join(path, "checkpoint-*")):
                max_step_checkpoint_num = int(checkpoint_dir.split("/")[-1].split('-')[-1])
                if np.greater(max_step_checkpoint_num,checkpoint_num):
                    checkpoint_num = max_step_checkpoint_num
                    max_step_checkpoint = checkpoint_dir
            if max_step_checkpoint != None:
                with open(os.path.join(max_step_checkpoint,"trainer_state.json"),mode='r') as f: # type: ignore
                    path = json.load(f)["best_model_checkpoint"] # type: ignore
                print(f"From max_step={max_step_checkpoint} load best checkpoint={path}")
        self.load_soft_prompt(path=path,task=task)
        self.load_soft_prompt_init(path=path,task=task)

    def load_soft_prompt_init(self, path, task="final"):
        task = task if task != None else self.task
        path = os.path.join(path,f"{task}_soft_prompt_set")
        soft_prompt_init_path = os.path.join(path, task+"_init.pt") # type: ignore
        print(f"load soft prompt init in {soft_prompt_init_path}")
        self.init_weight = torch.load(soft_prompt_init_path, map_location=torch.device(self.device),).detach().clone()
        
    def load_soft_prompt(self, path, task="final"):
        task = task if task != None else self.task
        path = os.path.join(path,f"{task}_soft_prompt_set")
        soft_prompt_path = os.path.join(path, task+".pt") # type: ignore
        print(f"load soft prompt in {soft_prompt_path}")
        if self.need_train :
            self.soft_prompt = nn.parameter.Parameter(torch.load(soft_prompt_path, map_location=torch.device(self.device)))
        else :
            self.soft_prompt = torch.load(soft_prompt_path, map_location=torch.device(self.device),).detach().clone()
