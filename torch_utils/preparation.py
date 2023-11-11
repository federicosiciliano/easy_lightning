# Import necessary libraries
import multiprocessing
import torch
import pytorch_lightning as pl
import torchmetrics
from copy import deepcopy
from torch.utils.data import DataLoader, TensorDataset
# Import modules and functions from local files
from .model import BaseNN
from . import metrics as custom_metrics
from . import losses as custom_losses  # Ensure your custom losses are imported

# Function to prepare data loaders
def prepare_data_loaders(data, loader_params, split_keys={"train": ["train_x", "train_y"], "val": ["val_x", "val_y"], "test": ["test_x", "test_y"]}):                         
    # Default loader parameters
    default_loader_params = {
        "num_workers": multiprocessing.cpu_count(),
        "pin_memory": True,
        "persistent_workers": True,
        "drop_last": {"train": False, "val": False, "test": False}
    }
    # Combine default and custom loader parameters
    loader_params = dict(list(default_loader_params.items()) + list(loader_params.items()))

    loaders = {}
    for split_name, data_keys in split_keys.items():
        split_loader_params = deepcopy(loader_params)
        # Select specific parameters for this split
        for key, value in split_loader_params.items():
            if isinstance(value, dict):
                if split_name in value.keys():
                    split_loader_params[key] = value[split_name]
        
        # Get data and create the TensorDataset
        features = torch.tensor(data[data_keys[0]], dtype=torch.float32)
        targets = torch.tensor(data[data_keys[1]], dtype=torch.float32)
        # Create an indices tensor
        indices = torch.arange(start=0, end=len(features), dtype=torch.long)
        td = TensorDataset(features, targets, indices)

        # Create the DataLoader
        loaders[split_name] = DataLoader(td, **split_loader_params)
    return loaders


# Function to prepare trainer parameters with experiment ID
def prepare_experiment_id(original_trainer_params, experiment_id):
    # Create a deep copy of the original trainer parameters
    trainer_params = deepcopy(original_trainer_params)

    # Check if "callbacks" is in trainer_params
    if "callbacks" in trainer_params:
        for callback_dict in trainer_params["callbacks"]:
            if isinstance(callback_dict, dict):
                for callback_name, callback_params in callback_dict.items():
                    if callback_name == "ModelCheckpoint":
                        # Update the "dirpath" to include the experiment_id
                        callback_params["dirpath"] += experiment_id + "/"
                    else:
                        # Print a warning message for unrecognized callback names
                        print(f"Warning: {callback_name} not recognized for adding experiment_id")
                        pass

    # Check if "logger" is in trainer_params
    if "logger" in trainer_params:
        # Update the "save_dir" in logger parameters to include the experiment_id
        trainer_params["logger"]["params"]["save_dir"] += experiment_id + "/"

    return trainer_params

# Function to prepare callbacks
def prepare_callbacks(trainer_params):
    # Seed the random number generator, although this is probably unnecessary
    pl.seed_everything(42, workers=True)  # Probably useless

    # Initialize an empty list for callbacks
    callbacks = []

    # Check if "callbacks" is in trainer_params
    if "callbacks" in trainer_params:
        for callback_dict in trainer_params["callbacks"]:
            if isinstance(callback_dict, dict):
                for callback_name, callback_params in callback_dict.items():
                    # Create callback instances based on callback names and parameters
                    callbacks.append(getattr(pl.callbacks, callback_name)(**callback_params))
                    # The following lines are commented out because they seem to be related to a specific issue
                    # if callback_name == "ModelCheckpoint":
                    #     if os.path.isdir(callbacks[-1].dirpath):
                    #         callbacks[-1].STARTING_VERSION = -1
            else:
                # If the callback is not a dictionary, add it directly to the callbacks list
                callbacks.append(callback_dict)
    
    return callbacks
    # The following lines are commented out because they seem to be related to a specific issue
    # new_trainer_params = copy.deepcopy(trainer_params)
    # new_trainer_params["callbacks"] = callbacks
    # return new_trainer_params


# Function to prepare a logger based on trainer parameters
def prepare_logger(trainer_params):
    pl.seed_everything(42, workers=True)  # Probably useless
    logger = None
    if "logger" in trainer_params:
        # Get the logger class based on its name and initialize it with parameters
        logger = getattr(pl.loggers, trainer_params["logger"]["name"])(**trainer_params["logger"]["params"])
    return logger

# Function to prepare a PyTorch Lightning Trainer instance
def prepare_trainer(seed=42, **kwargs):
    pl.seed_everything(seed, workers=True)  # Useless?

    # Default trainer parameters
    default_trainer_params = {"enable_checkpointing": False, "accelerator": "auto", "devices": "auto"}

    # Combine default parameters with user-provided kwargs
    trainer_params = dict(list(default_trainer_params.items()) + list(kwargs.items()))

    # Create a Trainer instance with the specified parameters
    trainer = pl.Trainer(**trainer_params)

    return trainer

# Function to prepare a loss function
def prepare_loss(loss):
    if isinstance(loss, str):
        # If 'loss' is a string, assume it's the name of a loss function
        loss_name = loss
        loss_params = {}
    elif isinstance(loss, dict):
        # If 'loss' is a dictionary, assume it contains loss name and parameters
        loss_name = list(loss.keys())[0]  # Accepts only one loss #TODO
        loss_params = loss[loss_name]
    else:
        raise NotImplementedError

    # Check if the loss_name exists in torch.nn or custom_losses
    if hasattr(torch.nn, loss_name):
        loss_module = torch.nn
    elif hasattr(custom_losses, loss_name):
        loss_module = custom_losses
    else:
        raise NotImplementedError("The loss function is not found in torch.nn or custom_losses.")

    # Create the loss function using the name and parameters
    return getattr(loss_module, loss_name)(**loss_params)


def prepare_metrics(metrics_info):
    # Initialize an empty dictionary to store metrics
    metrics = {}
    
    for metric_name in metrics_info:
        if isinstance(metrics_info, list): 
            metric_vals = {}  # Initialize an empty dictionary for metric parameters
        elif isinstance(metrics_info, dict): 
            metric_vals = metrics_info[metric_name]  # Get metric parameters from the provided dictionary
        else: 
            raise NotImplementedError  # Raise an error for unsupported input types
        
        # Check if the metric_name exists in torchmetrics or custom_metrics
        if hasattr(torchmetrics, metric_name):
            metrics_package = torchmetrics
        elif hasattr(custom_metrics, metric_name):
            metrics_package = custom_metrics
        else:
            raise NotImplementedError  # Raise an error if the metric_name is not found in any package
        
        # Create a metric object using getattr and store it in the metrics dictionary
        metrics[metric_name] = getattr(metrics_package, metric_name)(**metric_vals)
    
    # Convert the metrics dictionary to a ModuleDict for easy handling
    metrics = torch.nn.ModuleDict(metrics)
    return metrics

def prepare_optimizer(name, params):
    # Return a lambda function that creates an optimizer based on the provided name and parameters
    return lambda model_params: getattr(torch.optim, name)(model_params, **params)

def prepare_model(model_cfg):
    # Seed the random number generator for weight initialization
    pl.seed_everything(model_cfg["seed"], workers=True)
    
    # Create a model instance based on the provided configuration
    model = BaseNN(**model_cfg)
    return model


"""
# Prototype for logging different configurations for metrics and losses
def prepare_loss(loss_info):
    '''
    Prepare a loss function or multiple loss functions with different configurations.

    Parameters:
    - loss_info: Single loss function name or a list of loss function names with configurations.

    Returns:
    - loss: Dictionary containing loss functions and their respective configurations.
    '''
    if isinstance(loss_info, str):
        iterate_on = {loss_info: {}}
    elif isinstance(loss_info, list):
        iterate_on = {metric_name: {} for metric_name in loss_info}
    elif isinstance(loss_info, dict):
        iterate_on = loss_info
    else:
        raise NotImplementedError

    loss = {}
    for loss_name, loss_params in iterate_on.items():
        # Separate log_params from loss_params
        loss_log_params = loss_params.pop("log_params", {})

        loss_weight = loss_params.pop("weight", 1.0)

        loss[loss_name] = {"loss": getattr(torch.nn, loss_name)(**loss_params), "log_params": loss_log_params, "weight": loss_weight}

    return loss

def prepare_metrics(metrics_info):
    '''
    Prepare evaluation metrics or multiple metrics with different configurations.

    Parameters:
    - metrics_info: Single metric name or a list of metric names with configurations.

    Returns:
    - metrics: Dictionary containing metrics and their respective configurations.
    '''
    if isinstance(metrics_info, str):
        iterate_on = {metrics_info: {}}
    elif isinstance(metrics_info, list):
        iterate_on = {metric_name: {} for metric_name in metrics_info}
    elif isinstance(metrics_info, dict):
        iterate_on = metrics_info
    else:
        raise NotImplementedError

    metrics = {}
    for metric_name, metric_params in iterate_on.items():
        # Separate log_params from metric_params
        metric_log_params = metric_params.pop("log_params", {})

        metrics[metric_name] = {"metric": getattr(torchmetrics, metric_name)(**metric_params), "log_params": metric_log_params}
"""


# To solve OSError: [Errno 24] --->  Too many open files?
# sharing_strategy = "file_system"
# def set_worker_sharing_strategy(worker_id: int) -> None:
#     torch.multiprocessing.set_sharing_strategy(sharing_strategy)
# torch.multiprocessing.set_sharing_strategy(sharing_strategy)

# Function to add experiment info to ModelCheckpoint
# def add_exp_info_to_ModelCheckpoint(callbacks_dict, add_to_dirpath):
#     # print(callbacks_dict)
#     new_list = copy.deepcopy(callbacks_dict)

#     for MC_index, dc in enumerate(new_list):
#         if any([x == "ModelCheckpoint" for x in new_list]):
#             break

#     new_list[MC_index]["ModelCheckpoint"]["dirpath"] += str(add_to_dirpath)
#     # print(new_list)
#     return new_list

# Function to express neurons per layers
# def express_neuron_per_layers(cfg_model_cfg, model_cfg):
#     # probably not efficient since expressing all possible combinations
#     num_neurons = model_cfg["num_neurons"]
#     num_layers = model_cfg["num_layers"]

#     neurons_per_layer = []

#     for layer in num_layers:
#         neurons_per_layer += list(it.product(num_neurons, repeat=layer))

#     for cfg in [cfg_model_cfg, model_cfg]:
#         cfg.pop('num_neurons', None)
#         cfg.pop('num_layers', None)

#         cfg["neurons_per_layer"] = neurons_per_layer
