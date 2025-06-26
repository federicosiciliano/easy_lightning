#!/usr/bin/env python
# coding: utf-8

# # Preparation stuff

# ## Connect to Drive

# In[1]:


connect_to_drive = False


# In[2]:


#Run command and authorize by popup --> other window
if connect_to_drive:
    from google.colab import drive
    drive.mount('/content/gdrive', force_remount=True)


# ## Install packages

# In[3]:


if connect_to_drive:
    #Install FS code
    get_ipython().system('pip install  --upgrade --no-deps --force-reinstall git+https://github.com/federicosiciliano/easy_lightning.git@fedsic')

    get_ipython().system('pip install pytorch_lightning')


# ## IMPORTS

# In[4]:


#Put all imports here
import numpy as np
import matplotlib.pyplot as plt
#from copy import deepcopy
#import pickle
import os
import sys
#import cv2
import torch


# ## Define paths

# In[5]:


#every path should start from the project folder:
project_folder = "../"
if connect_to_drive:
    project_folder = "/content/gdrive/Shareddrives/<SharedDriveName>" #Name of SharedDrive folder
    #project_folder = "/content/gdrive/MyDrive/<MyDriveName>" #Name of MyDrive folder

#Config folder should contain hyperparameters configurations
cfg_folder = os.path.join(project_folder,"cfg")

#Data folder should contain raw and preprocessed data
data_folder = os.path.join(project_folder,"data")
raw_data_folder = os.path.join(data_folder,"raw")
processed_data_folder = os.path.join(data_folder,"processed")

#Source folder should contain all the (essential) source code
source_folder = os.path.join(project_folder,"src")

#The out folder should contain all outputs: models, results, plots, etc.
out_folder = os.path.join(project_folder,"out")
img_folder = os.path.join(out_folder,"img")


# ## Import own code

# In[6]:


#To import from src:

#attach the source folder to the start of sys.path
sys.path.insert(0, project_folder)

#import from src directory

import easy_exp, easy_rec, easy_torch #easy_data


# # MAIN

# ## Train

# ### Data

# In[7]:


cfg = easy_exp.cfg.load_configuration("config_rec")

for _ in cfg.sweep("model.loss"):
    # In[8]:


    cfg["data_params"]["data_folder"] = raw_data_folder


    # In[9]:


    #cfg["data_params"]["test_sizes"] = [cfg["data_params.dataset_params.out_seq_len.val"],cfg["data_params.dataset_params.out_seq_len.test"]]

    data, maps = easy_rec.data_generation_utils.preprocess_dataset(**cfg["data_params"])

    #TODO: save maps


    # In[10]:


    datasets = easy_rec.rec_torch.prepare_rec_datasets(data,**cfg["data_params"]["dataset_params"])


    # In[11]:


    cfg["data_params"]["collator_params"]["num_items"] = np.max(list(maps["sid"].values()))


    # In[12]:

    cfg["model"]["rec_model"]["num_items"] = np.max(list(maps["sid"].values()))
    cfg["model"]["rec_model"]["num_users"] = np.max(list(maps["uid"].values()))
    cfg["model"]["rec_model"]["lookback"] = cfg["data_params"]["collator_params"]["lookback"]


    # In[26]:


    exp_found, experiment_id = easy_exp.exp.get_set_experiment_id(cfg,nosave_removed=True) #WATCH OUT FOR NOSAVE REMOVED
    print("Experiment already found:", exp_found, "----> The experiment id is:", experiment_id)


    # In[16]:


    if exp_found:


        # In[17]:


        import json
        if exp_found:
            #load json file from out/exp
            orig_exp_cfg = json.load(open(os.path.join(out_folder,"exp",cfg["__exp__"]["name"],f"{experiment_id}.json"),"r"))


        # In[18]:


        #Load exps list
        exp_list = json.load(open(os.path.join(out_folder,"exp",f'{cfg["__exp__"]["name"]}_exp_list.json'),"r"))


        # In[19]:


        #Find difference between cfg and orig_exp_cfg
        different_keys = []
        for k in cfg:
            if k not in orig_exp_cfg:
                print("New key in cfg:",k)
                different_keys.append(k)
            elif orig_exp_cfg[k] != cfg[k]:
                print("Different value in key:",k,"cfg:",cfg[k],"orig_exp_cfg:",orig_exp_cfg[k])
                different_keys.append(k)
        #If only __exp__ is different, then it is fine
        if len(different_keys) != 1 or different_keys[0] != "__exp__":
            raise RuntimeError("Difference in cfg and orig_exp_cfg")


        # In[20]:


        sep_cfg, exp_cfg = easy_exp.exp.separate_exp_cfg(cfg)


        # In[21]:


        import copy
        jsonified_cfg = easy_exp.exp.jsonify_cfg(copy.deepcopy(sep_cfg))
        old_hash = easy_exp.exp.hash_config(orig_exp_cfg)
        new_hash = easy_exp.exp.hash_config(jsonified_cfg)
        if old_hash != new_hash: raise RuntimeError("Hashes are different")
        #Here nosave keys are not removed yet


        # In[22]:


        if exp_list[new_hash] != experiment_id: raise RuntimeError("Exp IDs are different")


        # In[34]:


        #Remove hash from exp_list
        exp_list.pop(new_hash, None)
        # Save exp_list again
        with open(os.path.join(out_folder,"exp",f'{cfg["__exp__"]["name"]}_exp_list.json'), 'w') as f:
            json.dump(exp_list, f)


        # In[35]:


        exp_found, new_experiment_id = easy_exp.exp.get_set_experiment_id(cfg)


        # In[36]:


        cfg["__exp__"]["experiment_id"] = experiment_id


        # In[37]:


        easy_exp.exp.save_experiment(cfg)

