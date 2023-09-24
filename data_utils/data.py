import os

from .file import *
from .transform import one_hot_encode_data, merge_splits, split_data, scale_data
from .uci import download_UCI
from .torch import get_torchvision_data

def load_data(source="local", split_vars = None, merge_before_split=False, one_hot_encode=True, **kwargs):
	dataloader = select_dataloader(source)

	#using kwargs["data"] instead of data, cause data may be inside kwargs in case from=="as_param" may avoid duplication?
	data = dataloader(**kwargs)

	if split_vars is not None: data = split_vars(data)

	if merge_before_split: merge_splits(**kwargs)
	
	data = split_data(data, **kwargs)

	data, data_scaler = scale_data(data, **kwargs)

	if one_hot_encode: data = one_hot_encode_data(data, **kwargs)

	if data_scaler is None: return data
	return data, data_scaler

def select_dataloader(source):
	###Select data loading
	loc = source.lower()
	if loc == "uci":
		dataloader = download_UCI
	# elif kwargs["from"].lower() == "tfds":
	# 	dataloader = get_tfds_data
	elif loc == "torchvision":
		dataloader = get_torchvision_data
	elif loc == "local":
		dataloader = get_local_data
	elif loc == "as_param":
		dataloader = lambda *args, **kwargs: kwargs["data"]
	elif callable(loc):
		dataloader = loc
	else:
		raise NotImplementedError("DATA IMPORT NOT IMPLEMENTED FROM", loc)
	return dataloader

def get_local_data(name, data_folder = "../data/", loader_params = {}, **kwargs):
	filename = os.path.join(data_folder,name)
	ext = filename.split(".")[-1]
	if ext=="npz": #already a dict
		dct = load_arrays(filename, loader_params, **kwargs)
		return dct
	# elif ext in ["pkl","pickle"]: #already a dict
	# 	dct = load_pickle(filename)
	# 	return dct
	else:
		if ext=="csv":
			app = load_csv(filename, loader_params, **kwargs)
		elif ext=="npy":
			app = load_numpy(filename, loader_params, **kwargs)
		elif ext=="npz":
			app = load_npz(filename, loader_params, **kwargs)
		return {"x": app}
	raise NotImplementedError