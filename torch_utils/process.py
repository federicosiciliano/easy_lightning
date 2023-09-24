import pandas as pd
import os
import torch
import pytorch_lightning as pl
from .model import BaseNN

def create_model(main_module, loss, optimizer, metrics={}, log_params={}, seed=42):
    pl.seed_everything(seed, workers=True) #for weight initialization
    model = BaseNN(main_module, loss, optimizer, metrics, log_params)
    return model

def train_model(trainer, model, loaders, train_key="train", val_key="val"):
    #pl.seed_everything(42, workers=True) #useless?
    if val_key is not None:
        if isinstance(val_key, str): val_dataloaders = loaders[val_key]
        elif isinstance(val_key, list): val_dataloaders = {key: loaders[key] for key in val_key}
        else: raise NotImplementedError
    else: val_dataloaders = None
    trainer.fit(model, loaders[train_key], val_dataloaders)
    
def test_model(trainer, model, loaders, loaders_key = "test"):
    #pl.seed_everything(42, workers=True) #probably useless?
    
    trainer.test(model, loaders[loaders_key])

def shutdown_dataloaders_workers():
    if torch.distributed.is_initialized():
        print("\n\nTORCH DISTR INIT\n\n")
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()

def load_model(model_cfg, path):
    model = BaseNN.load_from_checkpoint(path, **model_cfg)
    return model

def load_logs(name, exp_id, project_folder="../"):
    file_path = os.path.join(project_folder, "out", "log", name, exp_id, "lightning_logs", "version_0", "metrics.csv")

    #load CSV with pandas
    logs = pd.read_csv(file_path)

    return logs