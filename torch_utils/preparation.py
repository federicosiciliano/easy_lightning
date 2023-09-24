import multiprocessing

import torch
import pytorch_lightning as pl
import torchmetrics

from copy import deepcopy

from .model import BaseNN
from . import metrics as custom_metrics

def prepare_data_loaders(data, loader_params, split_keys = {"train": ["train_x", "train_y"], "val": ["val_x", "val_y"], "test": ["test_x", "test_y"]}):                         
    default_loader_params = {"num_workers": multiprocessing.cpu_count(), "pin_memory": True, "persistent_workers": True, "drop_last": {"train": False, "val": False, "test": False}} #"all": False, 
    loader_params = dict(list(default_loader_params.items()) + list(loader_params.items()))

    loaders = {}
    for split_name, data_keys in split_keys.items():
        split_loader_params = deepcopy(loader_params)
        # select specific parameters for this split
        for key,value in split_loader_params.items():
            if isinstance(value, dict):
                if split_name in value.keys():
                    split_loader_params[key] = value[split_name]
                # elif "all" in value.keys():
                #     split_loader_params[key] = value["all"]                        
        
        #get data
        split_data = []
        for data_key in data_keys:
            split_data.append(torch.Tensor(data[data_key]))

        # # if batch_size is float --> convert to int
        # if isinstance(split_loader_params["batch_size"], float):
        #     split_loader_params["batch_size"] = int(len(data[split_x])*split_loader_params["batch_size"])

        td = torch.utils.data.TensorDataset(*split_data)
        # Would need LongTensor if data[split_y] is not one-hot encoded

        loaders[split_name] = torch.utils.data.DataLoader(td, **split_loader_params)
    return loaders


def prepare_experiment_id(original_trainer_params, experiment_id):
    trainer_params = deepcopy(original_trainer_params)
    if "callbacks" in trainer_params:
        for callback_dict in trainer_params["callbacks"]:
            if isinstance(callback_dict, dict):
                for callback_name,callback_params in callback_dict.items():
                    if callback_name=="ModelCheckpoint":
                        callback_params["dirpath"] += experiment_id + "/"
                    else:
                        print(f"Warning: {callback_name} not recognized for adding experiment_id")
                        pass

    if "logger" in trainer_params:
        trainer_params["logger"]["params"]["save_dir"] += experiment_id + "/"

    return trainer_params

def prepare_callbacks(trainer_params):
    pl.seed_everything(42, workers=True) #probably useless

    callbacks = []
    if "callbacks" in trainer_params:
        for callback_dict in trainer_params["callbacks"]:
            if isinstance(callback_dict, dict):
                for callback_name,callback_params in callback_dict.items():
                    callbacks.append(getattr(pl.callbacks, callback_name)(**callback_params))
                    # if callback_name=="ModelCheckpoint": #doesn't seem to work; should make ModelCheckpoint overwrite old "best" file
                    #     if os.path.isdir(callbacks[-1].dirpath):
                    #         callbacks[-1].STARTING_VERSION = -1
            else:
                callbacks.append(callback_dict)
    
    return callbacks
    #new_trainer_params = copy.deepcopy(trainer_params)
    #new_trainer_params["callbacks"] = callbacks
    #return new_trainer_params

def prepare_logger(trainer_params):
    pl.seed_everything(42, workers=True) #probably useless
    logger = None
    if "logger" in trainer_params:
        logger = getattr(pl.loggers, trainer_params["logger"]["name"])(**trainer_params["logger"]["params"])
    return logger

def prepare_trainer(seed=42, **kwargs):
    pl.seed_everything(seed, workers=True) #useless?

    default_trainer_params = {"enable_checkpointing": False, "accelerator": "auto", "devices": "auto"}
    trainer_params = dict(list(default_trainer_params.items()) + list(kwargs.items()))

    trainer = pl.Trainer(**trainer_params)

    return trainer

def prepare_loss(loss):
    if isinstance(loss, str):
        loss_name = loss
        loss_params = {}
    elif isinstance(loss, dict):
        loss_name = list(loss.keys())[0] #accepts only one loss #TODO
        loss_params = loss[loss_name]
    #elif isinstance(loss, list): #should weight them, one parameter should be the weight
    else:
        raise NotImplementedError
    
    return getattr(torch.nn, loss_name)(**loss_params)

def prepare_metrics(metrics_info):
    metrics = {}
    for metric_name in metrics_info:
        if isinstance(metrics_info, list): metric_vals = {}
        elif isinstance(metrics_info, dict): metric_vals = metrics_info[metric_name]
        else: raise NotImplementedError
        
        #if torchmetrics has metric_name
        if hasattr(torchmetrics,metric_name):
            metrics_package = torchmetrics
        elif hasattr(custom_metrics,metric_name):
            metrics_package = custom_metrics
        else:
            raise NotImplementedError
        metrics[metric_name] = getattr(metrics_package,metric_name)(**metric_vals)
    metrics = torch.nn.ModuleDict(metrics)
    return metrics


#Prototype for log different for metric / loss
# def prepare_loss(loss_info):
#     if isinstance(loss_info, str):
#         iterate_on = {loss_info:{}}
#     elif isinstance(loss_info, list):
#         iterate_on = {metric_name:{} for metric_name in loss_info}
#     elif isinstance(loss_info, dict):
#         iterate_on = loss_info
#     else:
#         raise NotImplementedError

#     loss = {}
#     for loss_name, loss_params in iterate_on.items():
#         #Separate log_params from loss_params
#         loss_log_params = loss_params.pop("log_params", {})

#         loss_weight = loss_params.pop("weight", 1.0)

#         loss[loss_name] = {"loss":getattr(torch.nn,loss_name)(**loss_params), "log_params":loss_log_params, "weight":loss_weight}

#     return loss

# def prepare_metrics(metrics_info):
#     if isinstance(metrics_info, str):
#         iterate_on = {metrics_info:{}}
#     elif isinstance(metrics_info, list):
#         iterate_on = {metric_name:{} for metric_name in metrics_info}
#     elif isinstance(metrics_info, dict):
#         iterate_on = metrics_info
#     else:
#         raise NotImplementedError

#     metrics = {}
#     for metric_name, metric_params in iterate_on.items():
#         #Separate log_params from metric_params
#         metric_log_params = metric_params.pop("log_params", {})

#         metrics[metric_name] = {"metric":getattr(torchmetrics,metric_name)(**metric_params), "log_params":metric_log_params}
#     return metrics

def prepare_optimizer(name, params):
    return lambda model_params: getattr(torch.optim,name)(model_params,**params)

def prepare_model(model_cfg):
    pl.seed_everything(model_cfg["seed"], workers=True) #for weight initialization
    model = BaseNN(**model_cfg)
    return model

#To solve OSError: [Errno 24] Too many open files?
    #sharing_strategy = "file_system"
# def set_worker_sharing_strategy(worker_id: int) -> None:
#     torch.multiprocessing.set_sharing_strategy(sharing_strategy)
#torch.multiprocessing.set_sharing_strategy(sharing_strategy)


# def add_exp_info_to_ModelCheckpoint(callbacks_dict, add_to_dirpath):
#     #print(callbacks_dict)
#     new_list = copy.deepcopy(callbacks_dict)

#     for MC_index, dc in enumerate(new_list):
#         if any([x=="ModelCheckpoint" for x in new_list]):
#             break

#     new_list[MC_index]["ModelCheckpoint"]["dirpath"] += str(add_to_dirpath)
#     #print(new_list)
#     return new_list

# def express_neuron_per_layers(cfg_model_cfg, model_cfg):
#     #probably not efficient since expressing all possible combinations
#     num_neurons = model_cfg["num_neurons"]
#     num_layers = model_cfg["num_layers"]

#     neurons_per_layer = []

#     for layer in num_layers:
#         neurons_per_layer += list(it.product(num_neurons,repeat=layer))

#     for cfg in [cfg_model_cfg, model_cfg]:
#         cfg.pop('num_neurons', None)
#         cfg.pop('num_layers', None)

#         cfg["neurons_per_layer"] = neurons_per_layer