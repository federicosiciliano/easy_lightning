from . import data_generation_utils, rec_torch
from copy import deepcopy
import numpy as np

def prepare_rec_data(cfg, data_params=None):
    """
        Prepares recommendation data for training, validation, and testing.

        Args:
            cfg (dict): Configuration dictionary containing at least a `"data_params"` key
                        with all parameters needed for dataset preprocessing.
            data_params (dict, optional): Override dictionary for data parameters.
                                        If None, `cfg["data_params"]` is deep-copied and used.

        Returns:
            tuple:
                - data (dict): Dictionary containing split data (e.g., train/valid/test sequences).
                - maps (dict): Dictionary with mapping info (e.g., original to dense user/item IDs).
    """
    if data_params is None:
        data_params = deepcopy(cfg["data_params"])

    data, maps = data_generation_utils.preprocess_dataset(**data_params)

    return data, maps


def prepare_rec_dataloaders(cfg, data, maps=None, data_params=None, collator_params=None, loader_params=None):
    """
        Prepares PyTorch DataLoaders for recommendation tasks based on the provided configuration and data.

        Args:
            cfg (dict): Configuration dictionary.
            data (dict): Preprocessed data dictionary (typically the output of `prepare_rec_data`),
                        containing train/validation/test user interaction sequences.
            maps (dict, optional): Mapping dictionary for user and item IDs.
            data_params (dict, optional): Override dictionary for data-related parameters.
                                        If not provided, `cfg["data_params"]` is used.
            collator_params (dict, optional): Parameters for constructing data collators
                                            (e.g., padding, negative sampling settings).
            loader_params (dict, optional): Parameters for the DataLoader (e.g., batch size, shuffle).

        Returns:
            dict: A dictionary of PyTorch DataLoaders.
    """
    if data_params is None:
        data_params = deepcopy(cfg["data_params"])

    datasets = rec_torch.prepare_rec_datasets(data,**data_params["dataset_params"])

    if collator_params is None:
        collator_params = deepcopy(cfg["data_params"]["collator_params"])
    if "num_items" not in collator_params:
        if maps is None:
            raise ValueError("Item mapping must be provided if num_items is not in collator_params")
        collator_params["num_items"] = data_generation_utils.get_max_number_of(maps, "sid")
    collators = rec_torch.prepare_rec_collators(**collator_params)

    if loader_params is None:
        loader_params = deepcopy(cfg["model"]["loader_params"])
    loaders = rec_torch.prepare_rec_data_loaders(datasets, **loader_params, collate_fn=collators)

    return loaders


def prepare_rec_model(cfg, maps=None, data_params=None, rec_model_params=None, additional_module=None):
    """
        Prepares and instantiates a recommendation model using the given configuration.

        Args:
            cfg (dict): Configuration dictionary. 
            maps (dict, optional): Dictionary containing ID mappings for users (`"uid"`) and
                                items (`"sid"`).
            data_params (dict, optional): Override dictionary for data parameters.
                                        If None, `cfg["data_params"]` is used.
            rec_model_params (dict, optional): Parameters used to build the model.
                                            If None, `cfg["model"]["rec_model"]` is used.

        Returns:
            torch.nn.Module: Instantiated recommendation model ready for training.

        Raises:
            ValueError: If `maps` is not provided and required keys (`num_items`, `num_users`)
                        are missing in `rec_model_params`.
    """
       
    if data_params is None:
        data_params = deepcopy(cfg["data_params"])
    
    if rec_model_params is None:
        rec_model_params = deepcopy(cfg["model"]["rec_model"])
    if "num_items" not in rec_model_params:
        if maps is None:
            raise ValueError("Item mapping must be provided if num_items is not in rec_model_params")
        rec_model_params["num_items"] = data_generation_utils.get_max_number_of(maps, "sid")
    if "num_users" not in rec_model_params:
        if maps is None:
            raise ValueError("User mapping must be provided if num_users is not in rec_model_params")
        rec_model_params["num_users"] = data_generation_utils.get_max_number_of(maps, "uid")
    if "lookback" not in rec_model_params:
        rec_model_params["lookback"] = data_params["collator_params"]["lookback"]

    main_module = rec_torch.create_rec_model(**rec_model_params, additional_module=additional_module)#, graph=easy_rec.data_generation_utils.get_graph_representation(data["train_sid"]))
    
    return main_module




#TODO FS: fix preparation, avoid redundancy with rec_torch