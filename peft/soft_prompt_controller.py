import enum
import imp
import json
from mimetypes import init
import select
from tkinter import NO
from tkinter.tix import Tree
from turtle import forward
import torch
import numpy as np # type: ignore
from torch import nn
from torch.nn import functional as F
from sklearn.metrics.pairwise import cosine_similarity
import os
from .soft_prompt import SoftPrompt
class SoftPromptController(nn.Module):
    def __init__(self, config, model_shared = None):
        super().__init__()
        self.config = config
        self.prefix_dim = config.prefix_dim
        self.d_model = config.d_model
        self.prefix_random_range = config.prefix_random_range   
        self.device = config.device
        self.soft_prompts = nn.ModuleDict(dict())
        self.ablation = config.ablation
        self.init_group = None
        self.main_task = -1
        print(self.config.ablation)

        if model_shared != None:
            self.shared = model_shared.state_dict()["weight"].clone().detach()
        if self.config.init_prefix_from_vocab:
            indices = np.random.permutation(range(5000))[:self.prefix_dim]
            self.init_weight = self.shared[indices].to(self.device)

        else:
            self.init_weight =  torch.FloatTensor(self.prefix_dim, self.d_model).uniform_(-self.prefix_random_range, self.prefix_random_range).to(self.device)
        self.construct_target_soft_prompt()
        
        if config.multi_task:
            self.len_multi_task  = len(self.config.prompt_embedding_path)
            target_task_prompt = self.soft_prompts["0"].get_soft_prompt()
            self.construct_source_tasks_soft_prompt()
            ### SPoT
            sim_list = []
            id_list = []
            if self.config.ablation == 10:
                for source_task_name in self.source_task_names:
                    id = self.task_id[source_task_name]
                    source_soft_prompt = self.soft_prompts[id].get_soft_prompt()
                    print(source_task_name,"\n",source_soft_prompt)
                    sim_list.append(self.SPoT_Sim(target_task_prompt,source_soft_prompt).item())
                    id_list.append(id)
                # print(sim_list)
                most_sim_task = np.argmax(sim_list)
                most_sim_task_id = id_list[most_sim_task]
                most_sim_task_name = self.id_task[most_sim_task_id]
                most_sim_task_id = int(most_sim_task_id)-1
                print(sim_list)
                print(most_sim_task_id)
                print(most_sim_task_name)
                ##load
                self.soft_prompts["0"].load_soft_prompt_task_vector(self.source_prompt_embedding_path[most_sim_task_id],most_sim_task_name)
                self.config.multi_task = False
                return
            ### ones
            self.source_soft_prompt_TV_norm2 = nn.parameter.Parameter(torch.ones((self.len_multi_task,self.prefix_dim,)).to(self.device))
            self.target_soft_prompt_TV_norm2 = nn.parameter.Parameter(torch.ones((self.prefix_dim,)).to(self.device))
                
    def construct_target_soft_prompt(self):
        ##model target task 
        self.target_task = self.config.task_name[0]
        self.target_prompt_embedding_path = self.config.target_prompt_embedding_path
        self.load_target_task_name = self.config.load_target_task_name if self.config.load_target_task_name!=None else self.target_task
        self.soft_prompts["0"] = SoftPrompt(self.config, self.init_weight, self.target_task, 
                                            need_train = False if self.config.ablation >=6 and self.config.ablation !=10 else True)
        
        if self.target_prompt_embedding_path != None:
            self.soft_prompts["0"].load_soft_prompt_task_vector(self.target_prompt_embedding_path,self.load_target_task_name)
        print(f"OK Load {self.load_target_task_name} task vector")

    def construct_source_tasks_soft_prompt(self):

        self.source_task_names = self.config.multi_task_names
        self.source_prompt_embedding_path = self.config.prompt_embedding_path

        ##model task2id  & set self target task name = 0
        self.task_id = {}
        self.id_task = {}
        for id,task in enumerate(self.source_task_names):
            self.task_id[task] = str(id+1)
            self.id_task[str(id+1)] = task
            
        ##model source tasks
        for i,source_task_name in enumerate(self.source_task_names):
            id = self.task_id[source_task_name]
            self.soft_prompts[id] = SoftPrompt(config=self.config, init_weight=self.init_weight, task=source_task_name,need_train=True if self.config.ablation >=6 else False)
            if self.source_prompt_embedding_path[i] != None:
                self.soft_prompts[id].load_soft_prompt_task_vector(self.source_prompt_embedding_path[i],source_task_name)

    def SPoT_Sim(self, V1, V2):
        task_embedding_V1 = V1.mean(dim=0)
        task_embedding_V2 = V2.mean(dim=0)
        cos_sim = F.cosine_similarity(task_embedding_V1, task_embedding_V2, dim=0)
        return cos_sim

    def sim(self, T1, T2):
        # return np.matmul(V1,V2.T).diagonal().mean()
        task_vector_1 = T1.mean(dim=0)
        task_vector_2 = T2.mean(dim=0)
        return (task_vector_1*task_vector_2).mean(dim=0)


    def get_forward_prompt(self,
                           target_soft_prompt=None,
                           target_soft_prompt_TV=None,
                           source_task_prompts=None,
                           source_task_prompt_TVs=None,
                           source_task_names=None,
                           input_embedding=None
                           ):
        
        #PMDG
        if self.config.ablation == 0:
            ## step 1
            task_similarity_list = []
            postive_source_task_num = 0
            threshold = 0
            target_soft_prompt_TV = (target_soft_prompt_TV.T * self.target_soft_prompt_TV_norm2.detach()).T # type: ignore
            for i,task_prompt_TVs in enumerate(source_task_prompt_TVs): # type: ignore
                task_prompt_TVs = (task_prompt_TVs.T * self.source_soft_prompt_TV_norm2[i].detach()).T
                # task_similarity = np.matmul(target_soft_prompt_TV,task_prompt_TVs.T).diagonal().mean()
                task_similarity = self.sim(target_soft_prompt_TV,task_prompt_TVs)
                task_similarity_list.append(task_similarity)
                if task_similarity >= threshold:
                    postive_source_task_num = postive_source_task_num + 1

            pi = [i for i in range(self.len_multi_task)]
            pi = sorted(pi,key=lambda x : task_similarity_list[pi[x]],reverse=True) # type: ignore


            KCS = [0]
            rank_list = pi[0:postive_source_task_num]
            for decend_order_id in range(1,len(rank_list)):
                intra_group_sim = []
                
                for id_i in rank_list[0:decend_order_id+1]:
                    source_task_prompt_TVs_i = (source_task_prompt_TVs[id_i].T * self.source_soft_prompt_TV_norm2[id_i].detach()).T # type: ignore
                    for id_j in rank_list[0:decend_order_id+1]:
                        source_task_prompt_TVs_j = (source_task_prompt_TVs[id_j].T * self.source_soft_prompt_TV_norm2[id_j].detach()).T # type: ignore
                        
                        if id_i != id_j:
                            task_similarity = self.sim(source_task_prompt_TVs_i,source_task_prompt_TVs_j)
                            #intra_group_sim.append(np.matmul(source_task_prompt_TVs_i,source_task_prompt_TVs_j.T).diagonal().mean())
                            intra_group_sim.append(task_similarity)       
                            
                KCS.append(torch.tensor(intra_group_sim).mean(dim=0))
            best_task_group = torch.argmax(torch.tensor(KCS)) + 1
            
                
            selected_task_group_id = rank_list[0:best_task_group]
        
            if self.config.is_logger and self.training:
                self.logger_spot_sim(source_task_names,task_similarity_list,pi,rank_list,KCS,best_task_group,selected_task_group_id)
                #self.logger_scaling_term(source_task_names,self.target_soft_prompt_TV_norm2,self.source_soft_prompt_TV_norm2)
            
            ## step 3
            postive_source_task_TV = None
            for id in selected_task_group_id:
                source_task_prompt_TV_intra_group = torch.tensor(source_task_prompt_TVs[id]).to(self.device).detach() # type: ignore
                source_task_prompt_TV_intra_group = (source_task_prompt_TV_intra_group.T * self.source_soft_prompt_TV_norm2[id]).T
                postive_source_task_TV =  source_task_prompt_TV_intra_group if postive_source_task_TV == None else \
                                        postive_source_task_TV + source_task_prompt_TV_intra_group
                
            forward_target_soft_prompt_TV = self.soft_prompts["0"].get_soft_prompt_task_vector()
            forward_target_soft_prompt_TV = (forward_target_soft_prompt_TV.T * self.target_soft_prompt_TV_norm2).T
            
            learned_embeds = self.init_weight.detach() + (forward_target_soft_prompt_TV + postive_source_task_TV) \
                                if postive_source_task_TV != None  else self.init_weight + forward_target_soft_prompt_TV
           
        #only postive
        elif self.config.ablation == 1:
            ## step 1
            task_similarity_list = []
            postive_source_task_num = 0
            threshold = 0
            target_soft_prompt_TV = (target_soft_prompt_TV.T * self.target_soft_prompt_TV_norm2.detach().cpu().numpy()).T # type: ignore
            for i,task_prompt_TVs in enumerate(source_task_prompt_TVs): # type: ignore
                task_prompt_TVs = (task_prompt_TVs.T * self.source_soft_prompt_TV_norm2[i].detach().cpu().numpy()).T
                # task_similarity = np.matmul(target_soft_prompt_TV,task_prompt_TVs.T).diagonal().mean()
                task_similarity = self.sim(target_soft_prompt_TV,task_prompt_TVs)
                task_similarity_list.append(task_similarity)
                if task_similarity >= threshold:
                    postive_source_task_num = postive_source_task_num + 1

            task_similarity_list = np.array(task_similarity_list)
            pi = [i for i in range(self.len_multi_task)]
            pi = sorted(pi,key=lambda x : task_similarity_list[pi[x]],reverse=True) # type: ignore

            
            ## step 2 only postive 
            rank_list = pi[0:postive_source_task_num]
            selected_task_group_id = rank_list
            
            ## step 3
            postive_source_task_TV = None
            for id in selected_task_group_id:
                source_task_prompt_TV_intra_group = torch.tensor(source_task_prompt_TVs[id]).to(self.device).detach() # type: ignore
                source_task_prompt_TV_intra_group = (source_task_prompt_TV_intra_group.T * self.source_soft_prompt_TV_norm2[id]).T
                postive_source_task_TV =  source_task_prompt_TV_intra_group if postive_source_task_TV == None else \
                                        postive_source_task_TV + source_task_prompt_TV_intra_group
                
            forward_target_soft_prompt_TV = self.soft_prompts["0"].get_soft_prompt_task_vector()
            forward_target_soft_prompt_TV = (forward_target_soft_prompt_TV.T * self.target_soft_prompt_TV_norm2).T
            
            learned_embeds = self.init_weight.detach() + (forward_target_soft_prompt_TV + postive_source_task_TV) \
                                if postive_source_task_TV != None  else self.init_weight + forward_target_soft_prompt_TV
        
        #only max_KCS                        
        elif self.config.ablation == 2:
            ## step 1
            task_similarity_list = []
            rank_list = []
            threshold = 0
            detach_source_soft_prompt_TVs = [(source_task_prompt_TVs[i].T * self.source_soft_prompt_TV_norm2[i].detach().cpu().numpy()).T for i in range(self.len_multi_task)] # type: ignore
            
            all_permutation = 2**self.len_multi_task
            KCS = [0]
            
            for permutation in range(all_permutation):
                select = []
                temp_all_permutation = permutation
                for select_task_id in range(self.len_multi_task):
                    if temp_all_permutation%2==0:
                        select.append(select_task_id)
                    temp_all_permutation = int(temp_all_permutation / 2)
                    
                    ## every select sim
                    temp_KCS = 0
                    intra_group_sim = []
                    for id_i in select:
                        source_task_prompt_TVs_i = detach_source_soft_prompt_TVs[id_i] # type: ignore
                        for id_j in select:
                            source_task_prompt_TVs_j = detach_source_soft_prompt_TVs[id_j] # type: ignore
                            
                            if id_i != id_j:
                                task_similarity = self.sim(source_task_prompt_TVs_i,source_task_prompt_TVs_j)
                                #intra_group_sim.append(np.matmul(source_task_prompt_TVs_i,source_task_prompt_TVs_j.T).diagonal().mean())
                                intra_group_sim.append(task_similarity)  
                         
                    temp_KCS = np.array(intra_group_sim).mean() if len(intra_group_sim)>=1 else 0
                    
                if np.greater(temp_KCS,np.max(KCS)):
                    rank_list = select
                KCS.append(temp_KCS)
            
            
                
            selected_task_group_id = rank_list
        
            
            ## step 3
            postive_source_task_TV = None
            for id in selected_task_group_id:
                source_task_prompt_TV_intra_group = torch.tensor(source_task_prompt_TVs[id]).to(self.device).detach() # type: ignore
                source_task_prompt_TV_intra_group = (source_task_prompt_TV_intra_group.T * self.source_soft_prompt_TV_norm2[id]).T
                postive_source_task_TV =  source_task_prompt_TV_intra_group if postive_source_task_TV == None else \
                                        postive_source_task_TV + source_task_prompt_TV_intra_group
                
            forward_target_soft_prompt_TV = self.soft_prompts["0"].get_soft_prompt_task_vector()
            forward_target_soft_prompt_TV = (forward_target_soft_prompt_TV.T * self.target_soft_prompt_TV_norm2).T
            
            learned_embeds = self.init_weight.detach() + (forward_target_soft_prompt_TV + postive_source_task_TV) \
                                if postive_source_task_TV != None  else self.init_weight + forward_target_soft_prompt_TV
        
        #only targe and scaling term
        elif self.config.ablation == 3:

            forward_target_soft_prompt_TV = self.soft_prompts["0"].get_soft_prompt_task_vector()
            forward_target_soft_prompt_TV = (forward_target_soft_prompt_TV.T * self.target_soft_prompt_TV_norm2).T
            
            learned_embeds = self.init_weight.detach() + forward_target_soft_prompt_TV 
           
        #Without Grouping
        elif self.config.ablation == 4: 
            ## step 1
            
                
            selected_task_group_id = [i for i in range(self.len_multi_task)] # type: ignore
        
            
            ## step 3
            postive_source_task_TV = None
            for id in selected_task_group_id:
                source_task_prompt_TV_intra_group = torch.tensor(source_task_prompt_TVs[id]).to(self.device).detach() # type: ignore
                source_task_prompt_TV_intra_group = (source_task_prompt_TV_intra_group.T * self.source_soft_prompt_TV_norm2[id]).T
                postive_source_task_TV =  source_task_prompt_TV_intra_group if postive_source_task_TV == None else \
                                        postive_source_task_TV + source_task_prompt_TV_intra_group
                
            forward_target_soft_prompt_TV = self.soft_prompts["0"].get_soft_prompt_task_vector()
            forward_target_soft_prompt_TV = (forward_target_soft_prompt_TV.T * self.target_soft_prompt_TV_norm2).T
            
            learned_embeds = self.init_weight.detach() + (forward_target_soft_prompt_TV + postive_source_task_TV)
           
        #Fix Group
        elif self.config.ablation == 5:
            
            if self.init_group == None:
                ## step 1
                task_similarity_list = []
                postive_source_task_num = 0
                threshold = 0
                target_soft_prompt_TV = (target_soft_prompt_TV.T * self.target_soft_prompt_TV_norm2.detach().cpu().numpy()).T # type: ignore
                for i,task_prompt_TVs in enumerate(source_task_prompt_TVs): # type: ignore
                    task_prompt_TVs = (task_prompt_TVs.T * self.source_soft_prompt_TV_norm2[i].detach().cpu().numpy()).T
                    # task_similarity = np.matmul(target_soft_prompt_TV,task_prompt_TVs.T).diagonal().mean()
                    task_similarity = self.sim(target_soft_prompt_TV,task_prompt_TVs)
                    task_similarity_list.append(task_similarity)
                    if task_similarity >= threshold:
                        postive_source_task_num = postive_source_task_num + 1

                task_similarity_list = np.array(task_similarity_list)
                pi = [i for i in range(self.len_multi_task)]
                pi = sorted(pi,key=lambda x : task_similarity_list[pi[x]],reverse=True) # type: ignore

                
                ## step 2
                KCS = [0]
                rank_list = pi[0:postive_source_task_num]
                for decend_order_id in range(1,len(rank_list)):
                    intra_group_sim = []
                    
                    for id_i in rank_list[0:decend_order_id+1]:
                        source_task_prompt_TVs_i = (source_task_prompt_TVs[id_i].T * self.source_soft_prompt_TV_norm2[id_i].detach().cpu().numpy()).T # type: ignore
                        for id_j in rank_list[0:decend_order_id+1]:
                            source_task_prompt_TVs_j = (source_task_prompt_TVs[id_j].T * self.source_soft_prompt_TV_norm2[id_j].detach().cpu().numpy()).T # type: ignore
                            
                            if id_i != id_j:
                                task_similarity = self.sim(source_task_prompt_TVs_i,source_task_prompt_TVs_j)
                                #intra_group_sim.append(np.matmul(source_task_prompt_TVs_i,source_task_prompt_TVs_j.T).diagonal().mean())
                                intra_group_sim.append(task_similarity)       
                                
                    KCS.append(np.array(intra_group_sim).mean())
                KCS = np.array(KCS)
                best_task_group = np.argmax(KCS) + 1
                
                    
                selected_task_group_id = rank_list[0:best_task_group]
                self.init_group = selected_task_group_id
                
            if self.config.is_logger and self.training:
                self.logger_spot_sim(source_task_names,task_similarity_list,pi,rank_list,KCS,best_task_group,self.init_group)
                #self.logger_scaling_term(source_task_names,self.target_soft_prompt_TV_norm2,self.source_soft_prompt_TV_norm2)
            
            
            ## step 3
            postive_source_task_TV = None
            for id in self.init_group: # type: ignore
                source_task_prompt_TV_intra_group = torch.tensor(source_task_prompt_TVs[id]).to(self.device).detach() # type: ignore
                source_task_prompt_TV_intra_group = (source_task_prompt_TV_intra_group.T * self.source_soft_prompt_TV_norm2[id]).T
                postive_source_task_TV =  source_task_prompt_TV_intra_group if postive_source_task_TV == None else \
                                        postive_source_task_TV + source_task_prompt_TV_intra_group
                
            forward_target_soft_prompt_TV = self.soft_prompts["0"].get_soft_prompt_task_vector()
            forward_target_soft_prompt_TV = (forward_target_soft_prompt_TV.T * self.target_soft_prompt_TV_norm2).T
            
            learned_embeds = self.init_weight.detach() + (forward_target_soft_prompt_TV + postive_source_task_TV) \
                                if postive_source_task_TV != None  else self.init_weight + forward_target_soft_prompt_TV
        

        ##SPoT Dynamic
        elif self.config.ablation >= 6:
            if self.main_task == -1:
                task_similarity_list = [self.SPoT_Sim(target_soft_prompt,source_soft_prompt) for source_soft_prompt in source_task_prompts] # type: ignore
                self.source_task_prompts = source_task_prompts
                self.main_task = np.argmax(task_similarity_list) 
            elif self.config.ablation == 6:
                index = []
                task_similarity_list = []
                for i in range(self.len_multi_task):
                    index.append(i)
                    task_similarity_list.append(self.SPoT_Sim(source_task_prompts[self.main_task],self.source_task_prompts[i])) # type: ignore
                index = sorted(index,key=lambda x : task_similarity_list[x],reverse=True) # type: ignore
                #self.main_task = index[0]
                self.logger_Dymaic_task(source_task_names,task_similarity_list,source_task_names[index],index) # type: ignore
            task_id = str(self.main_task+1)
            learned_embeds = self.soft_prompts[task_id].get_soft_prompt()
           

        return learned_embeds


    def logger_Dymaic_task(self,source_task_names,task_similarity_list,rank_source_task_names,index):
        
        loss_output_dir = f"/home/LAB/zhangpy/miracle/PMDG/Dynamic_group/train_loss_dynamic_{self.target_task}_{self.config.ablation}_15.json"
        
        if not os.path.exists(loss_output_dir):
            with open(loss_output_dir, "w", encoding="utf-8") as json_file:
                pass 
                
        with open(file=loss_output_dir, mode = 'r' ) as f:
            if f.readable() and f.read().strip():
                f.seek(0)  # Reset file pointer to the beginning
                trian_loss_dynamic_group = json.load(f)
            else:
                trian_loss_dynamic_group = {} 

            trian_loss_dynamic_group["self_task"] = self.target_task
            trian_loss_dynamic_group["base_task"] = source_task_names.tolist()
            for name in ["index","task_similarity_list","rank_source_task_names"]:
                if name not in trian_loss_dynamic_group:
                    trian_loss_dynamic_group[name] = []

            trian_loss_dynamic_group["index"].append(np.array(index).tolist())
            trian_loss_dynamic_group["task_similarity_list"].append(np.array(task_similarity_list).tolist())
            trian_loss_dynamic_group["rank_source_task_names"].append(np.array(rank_source_task_names).tolist())
        
        with open(loss_output_dir, 'w') as file:
            json.dump(trian_loss_dynamic_group, file, indent=4)  # indent=4 用于格式化输出

    def logger_scaling_term(self,source_task_names,source_soft_prompt_TV_norm2,target_soft_prompt_TV_norm2s):
        source = source_soft_prompt_TV_norm2.tolist()
        target = [target_soft_prompt_TV_norm2.tolist() for target_soft_prompt_TV_norm2 in  target_soft_prompt_TV_norm2s] # type: ignore
        
        output_dir = "/home/LAB/zhangpy/miracle/PMDG/Dynamic_group/train_loss_group_relu.json"
        with open(file=output_dir, mode = 'r' ) as f:
            if f.readable() and f.read().strip():
                f.seek(0)  # Reset file pointer to the beginning
                logger = json.load(f)
            else:
                logger = {}  # 或者根据需要返回一个默认值
        for name in ["source_soft_prompt_TV_norm2","target_soft_prompt_TV_norm2s"]:
                if name not in logger:
                    logger[name] = []
        logger["self_task"] = self.target_task
        logger["base_task"] = source_task_names.tolist()
        logger["source_soft_prompt_TV_norm2"].append(source)
        logger["target_soft_prompt_TV_norm2s"].append(target)
        with open(output_dir, 'w') as file:
            json.dump(logger, file, indent=4)  # indent=4 用于格式化输出

        

    def logger_spot_sim(self,source_task_names,task_similarity_list,pi,rank_list,KCS,best_task_group,selected_task_group_id):
        
        loss_output_dir = f"/home/LAB/zhangpy/miracle/PMDG/Dynamic_group/train_loss_group_{self.target_task}.json"
        
        # 确保文件存在（模拟空文件的场景）
        if not os.path.exists(loss_output_dir):
            with open(loss_output_dir, "w", encoding="utf-8") as json_file:
                pass  # 创建一个空文件
                
        with open(file=loss_output_dir, mode = 'r' ) as f:
            if f.readable() and f.read().strip():
                f.seek(0)  # Reset file pointer to the beginning
                trian_loss_dynamic_group = json.load(f)
            else:
                trian_loss_dynamic_group = {}  # 或者根据需要返回一个默认值

            trian_loss_dynamic_group["self_task"] = self.target_task
            trian_loss_dynamic_group["base_task"] = source_task_names.tolist()
            for name in ["rank_task","sim_for_ranked","pos_source_task_num","sim_group_consistant","num_select_final_group","selet_name"]:
                if name not in trian_loss_dynamic_group:
                    trian_loss_dynamic_group[name] = []


            trian_loss_dynamic_group["rank_task"].append(source_task_names[pi].tolist())
            trian_loss_dynamic_group["sim_for_ranked"].append(task_similarity_list[pi].tolist())
            trian_loss_dynamic_group["pos_source_task_num"].append(len(rank_list))
            trian_loss_dynamic_group["sim_group_consistant"].append(KCS.tolist())
            trian_loss_dynamic_group["num_select_final_group"].append(int(best_task_group))
            trian_loss_dynamic_group["selet_name"].append(source_task_names[selected_task_group_id].tolist())
        
        with open(loss_output_dir, 'w') as file:
            json.dump(trian_loss_dynamic_group, file, indent=4)  # indent=4 用于格式化输出
        
        
    def prepare_forward_prompt(self,input_embedding=None):
        target_task_soft_prompt =  self.soft_prompts["0"].get_soft_prompt().detach()
        target_task_soft_prompt_TV = self.soft_prompts["0"].get_soft_prompt_task_vector().detach()
        
        source_task_soft_prompts = []
        source_task_soft_prompt_TVs = []
        source_task_names = []
        for source_task_name in self.source_task_names:
            source_task_id = self.task_id[source_task_name]
            source_task_soft_prompt_TV = self.soft_prompts[source_task_id].get_soft_prompt_task_vector().detach()
            source_task_soft_prompt_TVs.append(source_task_soft_prompt_TV) # type: ignore
            source_task_soft_prompts.append(self.soft_prompts[source_task_id].get_soft_prompt().detach())
            source_task_names.append(source_task_name)
            
        source_task_names = np.array(source_task_names)
        

        forward_prompt = self.get_forward_prompt(
                                                 target_soft_prompt=target_task_soft_prompt,
                                                 target_soft_prompt_TV=target_task_soft_prompt_TV,
                                                 source_task_prompts=source_task_soft_prompts,
                                                 source_task_prompt_TVs=source_task_soft_prompt_TVs,
                                                 source_task_names=source_task_names,
                                                 input_embedding=input_embedding)
        return forward_prompt

    def update_soft_prompt_final_to_evaluate(self, path, task = 'final'):
        self.soft_prompts["0"].load_soft_prompt_task_vector(path = path, task=task)
        # setting for evaluate
        self.config.multi_task = False
    
    def forward(self, input_embedding, task=None):
        batch_size = input_embedding.size(0)
        if self.config.multi_task:
            learned_embeds = self.prepare_forward_prompt(input_embedding)
        else :
            learned_embeds = self.soft_prompts["0"].get_soft_prompt()
        return  torch.cat([learned_embeds.repeat(batch_size, 1, 1), input_embedding],dim=1)  # type: ignore

    def extent_attention_mask(self, attention_mask):
        batch_size = attention_mask.size(0)
        extent_attention_mask = torch.ones([batch_size,self.config.prefix_dim]).to(attention_mask.device)
        return torch.cat([extent_attention_mask,attention_mask],dim=1).contiguous()
    
    def save_soft_prompt(self, output_dir = None):
        self.soft_prompts["0"].save_soft_prompt(output_dir,
                                                task = self.target_task) # type: ignore
        
        if self.config.multi_task:
            for task_name in self.source_task_names:
                source_task_id = self.task_id[task_name]
                self.soft_prompts[source_task_id].save_soft_prompt(output_dir,
                                                                   task = task_name) # type: ignore
                
            ## save multi task prompt
            print("SAVE MULTITASK")
            multi_task_prompt = self.prepare_forward_prompt()       
            self.soft_prompts["0"].save_soft_prompt(output_dir, 
                                                    task = "final", 
                                                    soft_prompt=multi_task_prompt, 
                                                    init_weight = self.init_weight) # type: ignore
            print("Save mix mulitask prompt")
            

    def sigmoid(self,vector):
        vector = np.array(vector)
        return 1/(1+np.exp(-vector))

    def get_forward_prompt_sigmoid(self,
                           target_soft_prompt=None,
                           target_soft_prompt_TV=None,
                           source_task_prompts=None,
                           source_task_prompt_TVs=None,
                           source_task_names=None,
                           input_embedding=None,
                           ):
    
        ## step 1
        task_similarity_list = []
        postive_source_task_num = 0
        threshold = 0
        target_soft_prompt_TV_norm2 = self.target_soft_prompt_TV_norm2.detach().cpu().numpy()
        target_soft_prompt_TV_norm2 = self.sigmoid(target_soft_prompt_TV_norm2)
        
        target_soft_prompt_TV = (target_soft_prompt_TV.T * target_soft_prompt_TV_norm2).T # type: ignore
        for i,task_prompt_TVs in enumerate(source_task_prompt_TVs): # type: ignore
            source_soft_prompt_TV_norm2 = self.source_soft_prompt_TV_norm2[i].detach().cpu().numpy()
            source_soft_prompt_TV_norm2 = self.sigmoid(source_soft_prompt_TV_norm2)
            
            task_prompt_TVs = (task_prompt_TVs.T * source_soft_prompt_TV_norm2).T
            task_similarity = np.matmul(target_soft_prompt_TV,task_prompt_TVs.T).diagonal().mean()
            task_similarity_list.append(task_similarity)
            if task_similarity >= threshold:
                postive_source_task_num = postive_source_task_num + 1

        task_similarity_list = np.array(task_similarity_list)
        pi = [i for i in range(self.len_multi_task)]
        pi = sorted(pi,key=lambda x : task_similarity_list[pi[x]],reverse=True) # type: ignore

        ## step 2
        KCS = [0]
        rank_list = pi[0:postive_source_task_num]
        for decend_order_id in range(1,len(rank_list)):
            intra_group_sim = []
            for id_i in rank_list[0:decend_order_id+1]:
                source_soft_prompt_TV_norm2_i = self.source_soft_prompt_TV_norm2[id_i].detach().cpu().numpy()
                source_soft_prompt_TV_norm2_i = self.sigmoid(source_soft_prompt_TV_norm2_i) # type: ignore
                
                source_task_prompt_TVs_i = (source_task_prompt_TVs[id_i].T * source_soft_prompt_TV_norm2_i).T # type: ignore
                for id_j in rank_list[0:decend_order_id+1]:
                    source_soft_prompt_TV_norm2_j = self.source_soft_prompt_TV_norm2[id_j].detach().cpu().numpy()
                    source_soft_prompt_TV_norm2_j =  self.sigmoid(source_soft_prompt_TV_norm2_j) # type: ignore
                    
                    source_task_prompt_TVs_j = (source_task_prompt_TVs[id_j].T * source_soft_prompt_TV_norm2_j).T # type: ignore
                    
                    if id_i != id_j:
                        intra_group_sim.append(np.matmul(source_task_prompt_TVs_i,source_task_prompt_TVs_j.T).diagonal().mean())
                        
            KCS.append(np.array(intra_group_sim).mean())
        KCS = np.array(KCS)
        best_task_group = np.argmax(KCS) + 1
        
            
        selected_task_group_id = rank_list[0:best_task_group]

        if self.config.is_logger and self.training:
            source_soft_prompt_TV_norm2s = []
            for source_soft_prompt_TV_norm2 in self.source_soft_prompt_TV_norm2:
                source_soft_prompt_TV_norm2_i = self.sigmoid(source_soft_prompt_TV_norm2.detach().cpu().numpy())
                source_soft_prompt_TV_norm2s.append(source_soft_prompt_TV_norm2_i)
            source_soft_prompt_TV_norm2s = np.array(source_soft_prompt_TV_norm2s)
            self.logger_scaling_term(source_task_names,target_soft_prompt_TV_norm2,source_soft_prompt_TV_norm2s)
        ## step 3
        
        postive_source_task_TV = None
        for id in selected_task_group_id:
            source_task_prompt_TV_intra_group = torch.tensor(source_task_prompt_TVs[id]).to(self.device).detach() # type: ignore
            source_task_prompt_TV_intra_group = (source_task_prompt_TV_intra_group.T * torch.sigmoid(self.source_soft_prompt_TV_norm2[id]) ).T
            postive_source_task_TV =  source_task_prompt_TV_intra_group if postive_source_task_TV == None else \
                                      postive_source_task_TV + source_task_prompt_TV_intra_group
            
        forward_target_soft_prompt_TV = self.soft_prompts["0"].get_soft_prompt_task_vector()
        forward_target_soft_prompt_TV = (forward_target_soft_prompt_TV.T * torch.sigmoid(self.target_soft_prompt_TV_norm2)).T
        
        learned_embeds = self.init_weight.detach() + (forward_target_soft_prompt_TV + postive_source_task_TV) \
                            if postive_source_task_TV != None  else self.init_weight + forward_target_soft_prompt_TV
        

        
        return learned_embeds
    
    
    def get_forward_prompt_relu(self,
                           target_soft_prompt=None,
                           target_soft_prompt_TV=None,
                           source_task_prompts=None,
                           source_task_prompt_TVs=None,
                           source_task_names=None,
                           input_embedding=None,
                           ):
    
        ## step 1
        task_similarity_list = []
        postive_source_task_num = 0
        threshold = 0
        target_soft_prompt_TV_norm2 = self.target_soft_prompt_TV_norm2.detach().cpu().numpy()
        target_soft_prompt_TV_norm2 = np.maximum(0,target_soft_prompt_TV_norm2) # type: ignore
        
        target_soft_prompt_TV = (target_soft_prompt_TV.T * target_soft_prompt_TV_norm2).T # type: ignore
        for i,task_prompt_TVs in enumerate(source_task_prompt_TVs): # type: ignore
            source_soft_prompt_TV_norm2 = self.source_soft_prompt_TV_norm2[i].detach().cpu().numpy()
            source_soft_prompt_TV_norm2 = np.maximum(0,source_soft_prompt_TV_norm2) # type: ignore
            
            task_prompt_TVs = (task_prompt_TVs.T * source_soft_prompt_TV_norm2).T
            task_similarity = np.matmul(target_soft_prompt_TV,task_prompt_TVs.T).diagonal().mean()
            task_similarity_list.append(task_similarity)
            if task_similarity >= threshold:
                postive_source_task_num = postive_source_task_num + 1

        task_similarity_list = np.array(task_similarity_list)
        pi = [i for i in range(self.len_multi_task)]
        pi = sorted(pi,key=lambda x : task_similarity_list[pi[x]],reverse=True) # type: ignore

        ## step 2
        KCS = [0]
        rank_list = pi[0:postive_source_task_num]
        for decend_order_id in range(1,len(rank_list)):
            intra_group_sim = []
            for id_i in rank_list[0:decend_order_id+1]:
                source_soft_prompt_TV_norm2_i = self.source_soft_prompt_TV_norm2[id_i].detach().cpu().numpy()
                source_soft_prompt_TV_norm2_i = np.maximum(0,source_soft_prompt_TV_norm2_i) # type: ignore
                
                source_task_prompt_TVs_i = (source_task_prompt_TVs[id_i].T * source_soft_prompt_TV_norm2_i).T # type: ignore
                for id_j in rank_list[0:decend_order_id+1]:
                    source_soft_prompt_TV_norm2_j = self.source_soft_prompt_TV_norm2[id_j].detach().cpu().numpy()
                    source_soft_prompt_TV_norm2_j = np.maximum(0,source_soft_prompt_TV_norm2_j) # type: ignore
                    
                    source_task_prompt_TVs_j = (source_task_prompt_TVs[id_j].T * source_soft_prompt_TV_norm2_j).T # type: ignore
                    
                    if id_i != id_j:
                        intra_group_sim.append(np.matmul(source_task_prompt_TVs_i,source_task_prompt_TVs_j.T).diagonal().mean())
                        
            KCS.append(np.array(intra_group_sim).mean())
        KCS = np.array(KCS)
        best_task_group = np.argmax(KCS) + 1
        
            
        selected_task_group_id = rank_list[0:best_task_group]

        if self.config.is_logger and self.training:
            source_soft_prompt_TV_norm2s = []
            for source_soft_prompt_TV_norm2 in self.source_soft_prompt_TV_norm2:
                source_soft_prompt_TV_norm2_i = np.maximum(0,source_soft_prompt_TV_norm2.detach().cpu().numpy()) # type: ignore
                source_soft_prompt_TV_norm2s.append(source_soft_prompt_TV_norm2_i)
            source_soft_prompt_TV_norm2s = np.array(source_soft_prompt_TV_norm2s)
            self.logger_scaling_term(source_task_names,target_soft_prompt_TV_norm2,source_soft_prompt_TV_norm2s)
        ## step 3
        
        postive_source_task_TV = None
        for id in selected_task_group_id:
            source_task_prompt_TV_intra_group = torch.tensor(source_task_prompt_TVs[id]).to(self.device).detach() # type: ignore
            source_task_prompt_TV_intra_group = (source_task_prompt_TV_intra_group.T * torch.relu(self.source_soft_prompt_TV_norm2[id]) ).T
            postive_source_task_TV =  source_task_prompt_TV_intra_group if postive_source_task_TV == None else \
                                      postive_source_task_TV + source_task_prompt_TV_intra_group
            
        forward_target_soft_prompt_TV = self.soft_prompts["0"].get_soft_prompt_task_vector()
        forward_target_soft_prompt_TV = (forward_target_soft_prompt_TV.T * torch.relu(self.target_soft_prompt_TV_norm2)).T
        
        learned_embeds = self.init_weight.detach() + (forward_target_soft_prompt_TV + postive_source_task_TV) \
                            if postive_source_task_TV != None  else self.init_weight + forward_target_soft_prompt_TV
        

        
        return learned_embeds