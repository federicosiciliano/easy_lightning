import os  # Import the 'os' module for operating system-related functions.

from .file import *  # Import all modules and functions from the 'file' module.
from .transform import one_hot_encode_data, merge_splits, split_data, scale_data  # Import specific functions from the 'transform' module.
from .uci import download_UCI  # Import the 'download_UCI' function from the 'uci' module.
from .torch import get_torchvision_data  # Import the 'get_torchvision_data' function from the 'torch' module.

def load_data(source="local", split_vars=None, merge_before_split=False, one_hot_encode=True, **kwargs):
    """
    Load and preprocess data for machine learning.

    :param source: Source of the data ('local', 'uci', 'torchvision', 'as_param', or a custom data loader).
    :param split_vars: Function to split variables within the data.
    :param merge_before_split: Boolean, whether to merge data before splitting.
    :param one_hot_encode: Boolean, whether to one-hot encode categorical data.
    :param kwargs: Additional keyword arguments for data loading and preprocessing.
    :return: Loaded and preprocessed data, and an optional data scaler.
    """
    # Call the 'select_dataloader' function to determine the appropriate data loader based on the 'source' parameter.
    dataloader = select_dataloader(source)

    # Call the selected data loader to obtain the 'data' object using any provided keyword arguments (**kwargs).
    data = dataloader(**kwargs)

    # If 'split_vars' is provided, call the 'split_vars' function to split the 'data' object.
    if split_vars is not None:
        data = split_vars(data)

    # If 'merge_before_split' is True, call the 'merge_splits' function to merge the data before splitting.
    if merge_before_split:
        merge_splits(**kwargs)

    # Call the 'split_data' function to split the 'data' object based on provided keyword arguments (**kwargs).
    data = split_data(data, **kwargs)

    # Call the 'scale_data' function to scale the 'data' object and obtain a scaled version of it.
    data, data_scaler = scale_data(data, **kwargs)

    # If 'one_hot_encode' is True, call the 'one_hot_encode_data' function to one-hot encode the 'data' object.
    if one_hot_encode:
        data = one_hot_encode_data(data, **kwargs)

    # If data_scaler is None, return only the 'data' object; otherwise, return both 'data' and 'data_scaler'.
    if data_scaler is None:
        return data
    return data, data_scaler

def select_dataloader(source):
    """
    Select the appropriate data loader function based on the 'source' parameter.

    :param source: Source of the data ('uci', 'torchvision', 'local', 'as_param', or a custom data loader).
    :return: Selected data loader function.
    """
    # Convert the source name to lowercase for case-insensitive comparison.
    loc = source.lower()

    if loc == "uci":
        dataloader = download_UCI  # Use the 'download_UCI' function for UCI data.
    elif loc == "torchvision":
        dataloader = get_torchvision_data  # Use the 'get_torchvision_data' function for torchvision data.
    elif loc == "local":
        dataloader = get_local_data  # Use the 'get_local_data' function for local data.
    elif loc == "as_param":
        dataloader = lambda *args, **kwargs: kwargs["data"]  # Use the data provided as a parameter.
    elif callable(loc):
        dataloader = loc  # Use a custom data loader function if 'source' is callable.
    else:
        raise NotImplementedError("DATA IMPORT NOT IMPLEMENTED FROM", loc)  # Raise an error for unsupported sources.

    return dataloader

def get_local_data(name, data_folder="../data/", loader_params={}, **kwargs):
    """
    Load local data based on file format.

    :param name: Name of the data file.
    :param data_folder: Folder path where data is located.
    :param loader_params: Parameters for data loading.
    :param kwargs: Additional keyword arguments for data loading.
    :return: Loaded data as a dictionary.
    """
    # Construct the full path to the data file.
    filename = os.path.join(data_folder, name)

    # Get the file extension to determine the file format.
    ext = filename.split(".")[-1]

    if ext == "npz":
        # If the file is in NPZ format, load it as a dictionary of arrays.
        dct = load_npz(filename, loader_params, **kwargs)
        return dct
    else:
        # For other file formats (e.g., CSV, NPY), load the data as an "x" key in a dictionary.
        app = load_csv(filename, loader_params, **kwargs) if ext == "csv" else load_numpy(filename, loader_params, **kwargs)
        return {"x": app}

    raise NotImplementedError  # Raise an error for unsupported file formats.
