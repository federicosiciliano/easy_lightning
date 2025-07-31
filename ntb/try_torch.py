#!/usr/bin/env python
# coding: utf-8

# # Testing Easy Torch

# In[1]:


import sys


# In[2]:


project_folder = "../"
sys.path.insert(0, project_folder)
print(sys.path) # View the path and verify


# In[3]:


import easy_data, easy_exp, easy_torch


# ## Configuration loading

# In[4]:


cfg = easy_exp.cfg.load_configuration("config_nn")


# In[5]:


cfg


# In[6]:


cfg["data"]


# In[7]:


data, _ = easy_data.data.load_data(**cfg["data"])


# In[8]:


loaders = easy_torch.preparation.prepare_data_loaders(data, **cfg["model"]["loader_params"])


# In[9]:


cfg["model"]["in_channels"] = data["train_x"].shape[1]
cfg["model"]["out_features"] = data["train_y"].shape[1]


# In[10]:


main_module = easy_torch.model.get_torchvision_model(**cfg["model"])


# In[11]:


exp_found, experiment_id = easy_exp.exp.get_experiment_id(cfg)


# In[12]:


print(exp_found, experiment_id)


# In[13]:


# Set experiment_id in trainer_params
trainer_params = easy_torch.preparation.prepare_experiment_id(cfg["model"]["trainer_params"], experiment_id)


# In[14]:


trainer_params["callbacks"] = easy_torch.preparation.prepare_callbacks(trainer_params)
trainer_params["logger"] = easy_torch.preparation.prepare_logger(trainer_params)
trainer = easy_torch.preparation.prepare_trainer(**trainer_params)

# callbacks = easy_torch.preparation.prepare_callbacks(cfg["model"]["trainer_params"])
# logger = easy_torch.preparation.prepare_logger(cfg["model"]["trainer_params"])
# already_defined = ["callbacks","logger"]
# trainer = easy_torch.preparation.prepare_trainer(**{k:cfg["model"]["trainer_params"][k] for k in cfg["model"]["trainer_params"] if k not in already_defined}, callbacks=callbacks, logger=logger)


# In[15]:


loss = easy_torch.preparation.prepare_loss(cfg["model"]["loss"])


# In[16]:


optimizer = easy_torch.preparation.prepare_optimizer(**cfg["model"]["optimizer"])


# In[17]:


model = easy_torch.process.create_model(main_module, loss=loss, optimizer=optimizer)


# In[18]:


easy_torch.process.train_model(trainer, model, loaders)


# In[19]:


easy_torch.process.test_model(trainer, model, loaders)


# In[20]:


easy_exp.exp.save_experiment(cfg)


# In[ ]:




