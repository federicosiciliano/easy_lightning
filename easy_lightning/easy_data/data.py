import os  # Import the 'os' module for operating system-related functions.

from .file import *  # Import all modules and functions from the 'file' module.
from .utils import *  # Import all modules and functions from the 'utils' module.
from .transform import one_hot_encode_data, merge_splits, split_data, scale_data  # Import specific functions from the 'transform' module.
from .uci import download_UCI  # Import the 'download_UCI' function from the 'uci' module.
from .torch import get_torchvision_data  # Import the 'get_torchvision_data' function from the 'torch' module.

def load_data(source="local", custom_data_functions=[], merge_before_split=False, one_hot_encode=True, **kwargs):
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

    # If 'custom_data_functions' is provided, call each 'custom_data_functions' function to modify the 'data' object.
    for custom_function_dict in custom_data_functions:
        data = custom_function_dict["function"](data, **custom_function_dict["kwargs"])

    # If 'merge_before_split' is True, call the 'merge_splits' function to merge the data before splitting.
    #TODO? do not use boolean to decide but other parameter
    if merge_before_split:
        merge_splits(**kwargs)

    # Call the 'split_data' function to split the 'data' object based on provided keyword arguments (**kwargs).
    data = split_data(data, **kwargs)

    # Call the 'scale_data' function to scale the 'data' object and obtain a scaled version of it.
    data, data_scaler = scale_data(data, **kwargs)

    # If 'one_hot_encode' is True, call the 'one_hot_encode_data' function to one-hot encode the 'data' object.
    #TODO? do not use boolean to decide but other parameter
    if one_hot_encode:
        data = one_hot_encode_data(data, **kwargs)

    # If data_scaler is None, return only the 'data' object; otherwise, return both 'data' and 'data_scaler'.
    if data_scaler is None:
        return data
    return data, data_scaler

def simple_load_data(source="local", **kwargs):
    # Call the 'select_dataloader' function to determine the appropriate data loader based on the 'source' parameter.
    dataloader = select_dataloader(source)

    # Call the selected data loader to obtain the 'data' object using any provided keyword arguments (**kwargs).
    data = dataloader(**kwargs)

    return data

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

def get_local_data(name, data_folder="../data/", local_key="x", loader_params={}, **kwargs):
    """
    Load local data based on file format.

    :param name: Name of the data file.
    :param data_folder: Folder path where data is located.
    :param loader_params: Parameters for data loading.
    :param kwargs: Additional keyword arguments for data loading.
    :return: Loaded data as a dictionary.
    """

    # If name is a list, load all files in the same dict
    if isinstance(name, list):
        # Check if local_key is a list of the same length
        if not isinstance(local_key, list):
            local_key = [local_key]
        if len(local_key) != len(name): #repeat local_key
            if len(local_key) == 1:
                local_key = [local_key[0]+str(i) for i in range(len(name))]
            else:
                raise ValueError("local_key must be a list of the same length as name or a single value")
        
        out = {}
        for filename,key in zip(name,local_key):
            loader_params_to_use = loader_params[filename] if filename in loader_params else loader_params
            out.update(get_local_data(filename, data_folder, key, loader_params_to_use, **kwargs))
        return out
    
    # Construct the full path to the data file.
    path = os.path.join(data_folder, name)

    #if filename is a folder, get all files in folder
    if os.path.isdir(path):
        filename = os.listdir(path)[0]
        ext = filename.split(".")[-1] # Get the file extension to determine the file format.
        num_files = len(os.listdir(path))
        out = {} 
        for file_i,f in enumerate(sorted(os.listdir(path))):
            print("Loading file",f,"(",file_i+1,"/",num_files,")")
            data = get_single_local_file(os.path.join(path,f), loader_params, ext, key=local_key, **kwargs)

            # Merge the data from all files into a single dictionary.
            # Continuously to avoid memory issues
            for key, value in data.items():
                if key not in out:
                    out[key] = np.zeros((num_files, *value.shape))
                if out[key][file_i].shape != value.shape: #increase size of out to match new data
                    new_shape = tuple(np.maximum(out[key][file_i].shape, value.shape))
                    if new_shape != out[key][file_i].shape:
                        out[key].resize((num_files, *new_shape))
                        print("WARNING: resizing",key,"to",new_shape)
                    insertion_slices = tuple(slice(0, value.shape[idx]) for idx in range(len(value.shape)))
                    out[key][file_i][insertion_slices] = value
                else:
                    out[key][file_i] = value
    else:
        out = get_single_local_file(path, loader_params, key=local_key, **kwargs)

    return out

dict_extensions = ["npz"]
array_extensions = ["npy", "csv", "jpg", "png"]

def get_single_local_file(filename, loader_params={}, ext=None, key="x", **kwargs):
    """
    Load a single local file based on file format.
    """

    ext = filename.split(".")[-1] if ext is None else ext
    
    if ext in dict_extensions:
        # If the file is in NPZ format, load it as a dictionary of arrays.
        dct = load_npz(filename, loader_params, **kwargs)
        return dct
    elif ext in array_extensions: # For other file formats (e.g., CSV, NPY), load the data as an key in a dictionary.
        if ext == "csv":
            app = load_csv(filename, loader_params, **kwargs)
        elif ext == "npy":
            app = load_numpy(filename, loader_params, **kwargs)
        elif ext in {"jpg","png"}:
            app = load_image(filename, loader_params, **kwargs)
        return {key: app}
    else:
        raise NotImplementedError("DATA IMPORT NOT IMPLEMENTED FOR", ext)
