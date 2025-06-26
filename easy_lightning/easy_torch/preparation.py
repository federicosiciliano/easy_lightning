# Import necessary libraries
import multiprocessing
import torch
import pytorch_lightning as pl
import torchmetrics
from copy import deepcopy
from torch.utils.data import DataLoader, TensorDataset
#import wandb
import os

from ray.train.lightning import prepare_trainer as prepare_ray_trainer
import ray.train.lightning as ray_lightning

# Import modules and functions from local files
from .model import BaseNN
from . import metrics as custom_metrics
from . import losses as custom_losses  # Ensure your custom losses are imported
from . import callbacks as custom_callbacks  # Ensure your custom losses are imported
from . import utils
from . import process


# Function to prepare data loaders
def prepare_data_loaders(data, split_keys={"train": ["train_x", "train_y"], "val": ["val_x", "val_y"], "test": ["test_x", "test_y"]}, dtypes = None, **loader_params):                             
    # Default loader parameters
    default_loader_params = {
        "num_workers": multiprocessing.cpu_count(),
        "pin_memory": True,
        "persistent_workers": True,
        "drop_last": {"train": False, "val": False, "test": False},
        "shuffle": {"train": True, "val": False, "test": False}
    }
    # Combine default and custom loader parameters
    loader_params = dict(list(default_loader_params.items()) + list(loader_params.items()))

    if dtypes is None or isinstance(dtypes, str) or isinstance(dtypes, torch.dtype):
        if isinstance(dtypes, str):
            dtypes = getattr(torch, dtypes)
        dtypes = {split_name: {data_key:dtypes for data_key in data_keys} for split_name, data_keys in split_keys.items()}
    elif isinstance(dtypes, dict):
        new_dtypes = {}
        for split_name, data_keys in split_keys.items():
            new_dtypes[split_name] = {}
            for data_key in data_keys:
                new_dtypes[split_name][data_key] = dtypes[data_key] if data_key in dtypes.keys() else None
                if isinstance(new_dtypes[split_name][data_key], str):
                    new_dtypes[split_name][data_key] = getattr(torch, new_dtypes[split_name][data_key])
        dtypes = new_dtypes
    else:
        raise NotImplementedError(f"Unsupported dtype: {dtypes}")

    loaders = {}
    for split_name, data_keys in split_keys.items():
        split_loader_params = deepcopy(loader_params)
        # Select specific parameters for this split
        for key, value in split_loader_params.items():
            if isinstance(value, dict):
                if split_name in value.keys():
                    split_loader_params[key] = value[split_name]
        
        # Get data and create the TensorDataset
        td = TensorDataset(*[torch.tensor(data[data_key], dtype=dtypes[split_name][data_key]) for data_key in data_keys])

        # Create the DataLoader
        loaders[split_name] = DataLoader(td, **split_loader_params)
    return loaders


# Function to prepare trainer parameters with experiment ID
def prepare_experiment_id(original_trainer_params, experiment_id, cfg=None):
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
        if trainer_params["logger"]["name"] == "WandbLogger":
            trainer_params["logger"]["params"]["id"] = experiment_id
            trainer_params["logger"]["params"]["name"] = experiment_id
            if cfg is not None:
                trainer_params["logger"]["params"]["config"] = cfg #TODO: Clean configuration
    return trainer_params

# Function to prepare callbacks
def prepare_callbacks(trainer_params, additional_module=None, seed=42):
    pl.seed_everything(seed) # Seed the random number generator

    # Initialize an empty list for callbacks
    callbacks = []

    # Check if "callbacks" is in trainer_params
    if "callbacks" in trainer_params:
        for callback_dict in trainer_params["callbacks"]:
            if isinstance(callback_dict, dict):
                for callback_name, callback_params in callback_dict.items():
                    # Create callback instances based on callback names and parameters
                    callbacks.append(get_single_callback(callback_name, callback_params, additional_module))
                    # The following lines are commented out because they seem to be related to a specific issue
                    # if callback_name == "ModelCheckpoint":
                    #     if os.path.isdir(callbacks[-1].dirpath):
                    #         callbacks[-1].STARTING_VERSION = -1
            else:
                # If the callback is not a dictionary, add it directly to the callbacks list
                callbacks.append(callback_dict)
    
    return callbacks

def remove_keys_from_dict(input_dict, keys_to_remove):
    """
    Recursively remove keys from a dictionary and all its sub-dictionaries.
    """
    if isinstance(input_dict, dict):
        for key in keys_to_remove:
            if key in input_dict:
                del input_dict[key]
        for value in input_dict.values():
            remove_keys_from_dict(value, keys_to_remove)
    return input_dict

# def log_wandb(trainer_params):
#     items_to_delete = ['__nosave__', 'emission_tracker', 'metrics',
#                        'data_folder', 'log_params', 'step_routing']
#     cfg = exp_utils.cfg.load_configuration()
#     exp_found, experiment_id = exp_utils.exp.get_set_experiment_id(cfg)
#     if not exp_found:
#         wandb.login(key=trainer_params["logger"]["key"])
#         if trainer_params["logger"]["entity"] is not None:
#             wandb.init(entity=trainer_params["logger"]["entity"],
#                     project=trainer_params["logger"]["project"],
#                     name = cfg['__exp__.name'] + "_" + experiment_id,
#                     id = experiment_id,
#                     config = remove_keys_from_dict(cfg, items_to_delete))
#         else:
#             wandb.init(project=trainer_params["logger"]["project"],
#                     name = cfg['__exp__.name'] + "_" + experiment_id,
#                     id = experiment_id,
#                     config = remove_keys_from_dict(cfg, items_to_delete))
    

# Function to prepare a logger based on trainer parameters
def prepare_logger(trainer_params, additional_module=None, seed=42):
    pl.seed_everything(seed) # Seed the random number generator
    logger = None
    if "logger" in trainer_params:
        # Get the logger class based on its name and initialize it with parameters
        if not os.path.exists(trainer_params["logger"]["params"]["save_dir"]):
            os.makedirs(trainer_params["logger"]["params"]["save_dir"])
        logger = get_function(trainer_params["logger"]["name"], additional_module, pl.loggers)(**trainer_params["logger"]["params"])
        #if isinstance(logger, pl.loggers.wandb.WandbLogger):
        #This is the case when the logger is wandb so we check for the entity and the the key
            #log_wandb(trainer_params)
        #TODO: Multiple loggers

    return logger

# Function to prepare strategy
def prepare_strategy(trainer_params, additional_module=None):
    # Have to check if strategy is in pytorch_lightning.strategies or additional_module, otherwise leave it as string (the trainer will handle it)

    # Check if "strategy" is in trainer_params
    strategy = "auto"
    if "strategy" in trainer_params:
        strategy_info = trainer_params["strategy"]
        if isinstance(strategy_info, str):
            strategy_name = strategy_info
            strategy_params = {}
        elif isinstance(strategy_info, dict):
            strategy_name = strategy_info["name"]
            strategy_params = strategy_info.get("params", {})
            
        function_module = get_correct_package(strategy_name, additional_module, pl.strategies, ray_lightning)

        if function_module is not None:
            strategy = getattr(function_module, strategy_name)(**strategy_params)
        else:
            strategy = strategy_name
    return strategy

def prepare_plugins(trainer_params, additional_module=None):
    # Check if "plugins" is in trainer_params
    plugins = [] # Initialize an empty list for plugins
    if "plugins" in trainer_params:
        for plugin_info in trainer_params["plugins"]:
            if isinstance(plugin_info, str):
                plugin_name = plugin_info
                plugin_params = {}
            elif isinstance(plugin_info, dict):
                plugin_name = plugin_info["name"]
                plugin_params = plugin_info.get("params", {})
            
            plugin = get_function(plugin_name, additional_module, pl.plugins, ray_lightning)(**plugin_params)

            plugins.append(plugin)

    return plugins

# Function to prepare a PyTorch Lightning Trainer instance
def prepare_trainer(seed=42, raytune=False, **trainer_kwargs):
    pl.seed_everything(seed) # Seed the random number generator

    # Default trainer parameters
    default_trainer_params = {"enable_checkpointing": False, "accelerator": "auto", "devices": "auto"}

    # Combine default parameters with user-provided kwargs
    trainer_params = dict(list(default_trainer_params.items()) + list(trainer_kwargs.items()))

    # Create a Trainer instance with the specified parameters
    trainer = pl.Trainer(**trainer_params)

    if raytune:
        trainer = prepare_ray_trainer(trainer)

    return trainer

# Function to prepare a loss function
def prepare_loss(loss_info, *additional_modules, seed=42):
    pl.seed_everything(seed) # Seed the random number generator
    if isinstance(loss_info, str):
        # If 'loss' is a string, assume it's the name of a loss function
        loss = get_single_loss(loss_info, {}, *additional_modules)
    elif isinstance(loss_info, dict):
        # If 'loss' is a dictionary, assume it contains loss name and parameters
        loss = {}
        for loss_name, loss_params in sorted(loss_info.items()):
            if loss_name != "__weight__":
                loss[loss_name] = get_single_loss(loss_params["name"], loss_params.get("params",{}), *additional_modules)
        loss = torch.nn.ModuleDict(loss)
        loss.__weight__ = loss_info.get("__weight__", torch.ones(len(loss)))
    else:
        raise NotImplementedError
    return loss

def get_single_loss(loss_name, loss_params, *additional_modules):
    return get_function(loss_name, *additional_modules, custom_losses, torch.nn)(**loss_params)

def get_single_callback(callback_name, callback_params, additional_module=None):
    return get_function(callback_name, additional_module, custom_callbacks, pl.callbacks, ray_lightning)(**callback_params)

def get_function(function_name, *modules):
    # Check if the function_name exists in additional_module or torch/torchmetrics
    function_module = get_correct_package(function_name, *modules)
    
    # Return the function using the name and parameters
    return getattr(function_module, function_name)

def prepare_metrics(metrics_info, *additional_modules, split_keys={"train":1,"val":2,"test":3}, seed=42):
    # TODO: repeat metric if same dataloader is used for multiple splits?

    # Initialize an empty dictionary to store metrics
    metrics = {}
    if isinstance(metrics_info, dict) and all([key in metrics_info for key in split_keys.keys()]):
        metrics_info_already_split = True
    else:
        metrics_info_already_split = False
    
    for split_name, num_dataloaders in split_keys.items():
        metrics[split_name] = []
        for dataloader_idx in range(num_dataloaders):
            metrics[split_name].append({})
            if metrics_info_already_split:
                metrics_info_to_use = metrics_info[split_name][dataloader_idx]
            else:
                metrics_info_to_use = metrics_info
    
            for metric_name in metrics_info_to_use:
                if isinstance(metrics_info_to_use, list): 
                    metric_vals = {}  # Initialize an empty dictionary for metric parameters
                elif isinstance(metrics_info_to_use, dict): 
                    metric_vals = metrics_info_to_use[metric_name]  # Get metric parameters from the provided dictionary
                else: 
                    raise NotImplementedError  # Raise an error for unsupported input types
                
                pl.seed_everything(seed) # Seed the random number generator

                # Check if metric_name is the special FakeMetricCollectionMetric
                metric_name, true_metric_name, metric_vals = handle_FakeMetricCollection(metric_name, metric_vals, *additional_modules)

                # Create a metric object using getattr and store it in the metrics dictionary
                metrics[split_name][-1][true_metric_name] = get_function(metric_name, *additional_modules, custom_metrics, torchmetrics)(**metric_vals)
            metrics[split_name][-1] = torch.nn.ModuleDict(metrics[split_name][-1])
        metrics[split_name] = torch.nn.ModuleList(metrics[split_name])
    
    # Convert the metrics dictionary to a ModuleDict for easy handling
    metrics = utils.RobustModuleDict(metrics)
    return metrics

def handle_FakeMetricCollection(metric_name, metric_params, *additional_modules):
    # Check if the metric name is "FakeMetricCollectionMetric"
    true_metric_name = metric_name
    if "FakeMetricCollection" in metric_name:
        metric_name,true_metric_name = metric_name.split(":") #TODO Check if best way to split
        # Get the actual class from the name
        metric_params = {**metric_params, "metric_class": get_function(true_metric_name, *additional_modules, custom_metrics, torchmetrics)} #to avoid overwriting the original metric_params
    return metric_name, true_metric_name, metric_params

def prepare_optimizer(name, params={}, seed=42):
    pl.seed_everything(seed) # Seed the random number generator
    # Return a lambda function that creates an optimizer based on the provided name and parameters
    return lambda model_params: getattr(torch.optim, name)(model_params, **params)

def prepare_model(model_cfg):
    # Seed the random number generator for weight initialization
    pl.seed_everything(model_cfg["seed"]) # Seed the random number generator
    
    # Create a model instance based on the provided configuration
    model = BaseNN(**model_cfg)
    return model

def prepare_emission_tracker(experiment_id, **tracker_kwargs):
    from codecarbon import EmissionsTracker
    # Update the "output_dir" in tracker parameters to include the experiment_id
    tracker_kwargs["output_dir"] = tracker_kwargs.get("output_dir", "../out/log/") + experiment_id + "/"
    tracker = EmissionsTracker(**tracker_kwargs)
    return tracker

def prepare_flops_profiler(model, experiment_id, **profiler_kwargs):
    from deepspeed.profiling.flops_profiler import FlopsProfiler
    output_dir = profiler_kwargs.pop("output_dir", "../out/log/")
    profiler = FlopsProfiler(model, **profiler_kwargs)
    profiler.output_dir = output_dir + experiment_id + "/"
    return profiler

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
#     new_list = copy.deepcopy(callbacks_dict)

#     for MC_index, dc in enumerate(new_list):
#         if any([x == "ModelCheckpoint" for x in new_list]):
#             break

#     new_list[MC_index]["ModelCheckpoint"]["dirpath"] += str(add_to_dirpath)
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

def get_correct_package(name, *modules, raise_error=True):
    # Check if name exists in any module, in order
    for module in modules:
        if hasattr(module, name):
            return module
    if raise_error:
        raise NotImplementedError(f"The function/class {name} is not found in [{', '.join([module.__name__ for module in modules])}]")
    else: #raise only a warning
        print(f"Warning: The function/class {name} is not found in [{', '.join([module.__name__ for module in modules])}]")

def complete_prepare_trainer(cfg, experiment_id, model_params=None, additional_module={}, raytune=False):
    if model_params is None:
        model_params = deepcopy(cfg["model"])

    trainer_params = prepare_experiment_id(model_params["trainer_params"], experiment_id)

    # Prepare callbacks and logger using the prepared trainer_params
    trainer_params["callbacks"] = prepare_callbacks(trainer_params, getattr(additional_module,"callbacks",None))
    trainer_params["logger"] = prepare_logger(trainer_params, getattr(additional_module,"loggers",None))
    trainer_params["strategy"] = prepare_strategy(trainer_params, getattr(additional_module,"strategies",None))
    trainer_params["plugins"] = prepare_plugins(trainer_params, getattr(additional_module,"plugins",None))

    # Prepare the trainer using the prepared trainer_params
    trainer = prepare_trainer(**trainer_params, raytune=raytune)

    return trainer

def complete_prepare_model(cfg, main_module, *additional_modules, model_params=None):
    model_params = deepcopy(cfg["model"])

    model_params["loss"] = prepare_loss(model_params["loss"], *[getattr(module,"losses",module) for module in additional_modules])

    # Prepare the optimizer using configuration from cfg
    model_params["optimizer"] = prepare_optimizer(**model_params["optimizer"])

    # Prepare the metrics using configuration from cfg
    model_params["metrics"] = prepare_metrics(model_params["metrics"], *[getattr(module,"metrics",module) for module in additional_modules])

    # Create the model using main_module, loss, and optimizer
    model = process.create_model(main_module, **model_params)

    return model

#TODO: check additional_modules functionality

# Deprecated
def prepare_profiler(trainer_params, additional_module=None, seed=42):
    pl.seed_everything(seed) # Seed the random number generator

    # Check if "profiler" is in trainer_params
    if "profiler" in trainer_params:
        if isinstance(trainer_params["profiler"], dict):
            # Create profiler instances based on profiler names and parameters
            profiler = get_single_callback(trainer_params["profiler"]["name"], trainer_params["profiler"]["params"], additional_module)
        trainer_params["profiler"] = profiler
    return trainer_params