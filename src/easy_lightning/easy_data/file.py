import numpy as np  # Import the 'numpy' library as 'np'.
import pandas as pd  # Import the 'pandas' library as 'pd'.
import imageio  # Import the 'imageio' library.

def load_csv(filename, loader_params, **kwargs):
    """
    Load data from a CSV file.

    :param filename: Name of the CSV file to load.
    :param loader_params: Parameters for loading the data.
    :param load_using_pandas: Boolean, whether to load using Pandas (default is False).
    :param kwargs: Additional keyword arguments for data loading.
    :return: Loaded data as a NumPy array or Pandas DataFrame.
    """
    if loader_params.get("load_using_pandas", False):
        load_data = pd.read_csv(filename, **{k: v for k, v in loader_params.items() if k != "load_using_pandas"})
    else:
        load_data = np.genfromtxt(filename, **loader_params)  # Load data using NumPy.
    return load_data

def load_numpy(filename, loader_params, **kwargs):
    """
    Load data from a NumPy binary file (.npy).

    :param filename: Name of the NumPy binary file to load.
    :param loader_params: Parameters for loading the data.
    :param kwargs: Additional keyword arguments for data loading.
    :return: Loaded data as a NumPy array.
    """
    arr = np.load(filename, **loader_params)
    return arr

def load_npz(filename, loader_params, **kwargs):
    """
    Load data from a NumPy compressed archive file (.npz).

    :param filename: Name of the NumPy compressed archive file to load.
    :param loader_params: Parameters for loading the data.
    :param kwargs: Additional keyword arguments for data loading.
    :return: Loaded data as a dictionary.
    """
    arr = np.load(filename, **loader_params)
    return dict(arr.items())

#TODO: load_pickle

def load_image(filename, loader_params, **kwargs):
    """
    Load data from an image file.

    :param filename: Name of the image file to load.
    :param loader_params: Parameters for loading the data.
    :param kwargs: Additional keyword arguments for data loading.
    :return: Loaded data as a NumPy array.
    """
    arr = np.asarray(imageio.imread(filename, **loader_params))
    return arr

def load_arrays(filename, loader_params, **kwargs):
    """
    Load data from a NumPy binary file (.npy) and return it as a dictionary.

    :param filename: Name of the NumPy binary file to load.
    :param loader_params: Parameters for loading the data.
    :param kwargs: Additional keyword arguments for data loading.
    :return: Loaded data as a dictionary.
    """
    arr = load_numpy(filename, loader_params)
    arr.allow_pickle = True  # Set allow_pickle to True to support loading of pickled objects.
    return dict(arr.items())

def save_arrays(filename, arr):
    """
    Save data as a NumPy compressed archive file (.npz).

    :param filename: Name of the output NumPy compressed archive file.
    :param arr: Dictionary containing data to be saved.
    """
    np.savez(filename, **arr)