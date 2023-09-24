import numpy as np
import pandas as pd

def load_csv(filename, loader_params, load_using_pandas=False, **kwargs):
    if load_using_pandas:
        load_data = pd.read_csv(filename, **loader_params)#.to_numpy() #or .to_numpy()
    else:
        load_data = np.genfromtxt(filename, **loader_params)
    return load_data

def load_numpy(filename, loader_params, **kwargs):
    arr = np.load(filename, **loader_params)
    return arr

def load_npz(filename, loader_params, **kwargs):
    arr = np.load(filename, **loader_params)
    #arr = {key:value for key,value in app.items()}
    return dict(arr.items())

#def load_pickle

def load_arrays(filename, loader_params, **kwargs):
    arr = load_numpy(filename, loader_params)
    arr.allow_pickle = True #why this?
    return dict(arr.items())

def save_arrays(filename, arr):
    np.savez(filename, **arr)