import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler


def merge_splits(data, merge_keys = {"x": ["train_x", "val_x", "test_x"], "y": ["train_y", "val_y", "test_y"]}, concat_function = np.concatenate, del_after_merge = True, **kwargs):
	for merged_var,split_vars in merge_keys.items():
		app = []
		for key in split_vars:
			if key in data:
				app.append(data[key])
				if del_after_merge: del data[key]
		data[merged_var] = concat_function(app, axis=0) #or torch?

def split_data(data, split_keys = {"x": ["train_x", "val_x", "test_x"], "y": ["train_y", "val_y", "test_y"]}, test_sizes = 0.2, train_sizes = None, split_random_state=21094, split_shuffle = True, split_stratify = None, del_after_split = False, **kwargs):
	num_splits = len(split_keys[list(split_keys.keys())[0]])-1
	if isinstance(test_sizes,int) or isinstance(test_sizes,float): test_sizes = [test_sizes]*num_splits
	if train_sizes is None or isinstance(train_sizes,int) or isinstance(train_sizes,float): train_sizes = [train_sizes]*num_splits
	if split_random_state is None or isinstance(split_random_state,int): split_random_state = [split_random_state]*num_splits
	if isinstance(split_shuffle,bool): split_shuffle = [split_shuffle]*num_splits
	if not isinstance(split_stratify,tuple) or not isinstance(split_stratify,list): split_stratify = [split_stratify]*num_splits
	#can be other types? numpy array?

	accum_vars = []
	for merged_var,split_vars in split_keys.items():
		accum_vars.append(split_vars[0])
		data[accum_vars[-1]] = data[merged_var]

	for merged_vars, test_size, train_size, random_state, shuffle, stratify in zip(list(zip(*split_keys.values()))[1:], test_sizes, train_sizes, split_random_state, split_shuffle, split_stratify):
			if not all([x in data for x in merged_vars]): #do not overwrite existing data (if all existing). Should put condition?
				app = train_test_split(*[data[accum_var] for accum_var in accum_vars], test_size = test_size, train_size=train_size, random_state = random_state, shuffle=shuffle, stratify=stratify)

				cont = 0
				for accum_key, new_key in zip(accum_vars, merged_vars):
					data[accum_key] = app[cont]
					data[new_key] = app[cont+1]
					cont += 2
			else:
				print("Split already existing")

	if del_after_split:
		for merged_var in split_keys.keys():
			del data[merged_var]
	return data

def scale_data(data, scaling_method = None, scaling_params = {}, scaling_fit_key = "train_x", scaling_keys = ["train_x", "val_x", "test_x"], scaling_fit_params = {}, scaling_transform_params = [{},{},{}], **kwargs):
	#scaling_fit_key could be removed and selected as scaling_keys[0]
	#this would be less general, as someone may want, in rare cases, to fit on a variable but not transform it
	
	data_scaler = None
	if scaling_method is not None:
		data_scaler = getattr(preprocessing,scaling_method)(**scaling_params)
		data_scaler.fit(data[scaling_fit_key],**scaling_fit_params)
		for key,params in zip(scaling_keys,scaling_transform_params):
			data[key] = data_scaler.transform(data[key],**params)
	return data, data_scaler

def one_hot_encode_data(data, encode_keys = {"y", "train_y", "val_y", "test_y"}, encode_fit_key = "train_y", onehotencoder_params= {"sparse": False}, **kwargs):
	# if encode_fit_key not in data: raise error?
	
	encode_keys = encode_keys.intersection(data.keys()).difference({encode_fit_key}) #or raise error?

	#if (len(train_y.shape)==1) or (train_y.shape==1) or (len(np.unique(train_y))>2) or (train_y.dtype not in ["int","float"]):
	enc = OneHotEncoder(**onehotencoder_params)

	#fit on one key
	data[encode_fit_key] = one_hot_encode_matrix(enc, data[encode_fit_key], "fit_transform")
	for key in encode_keys: #transform other keys
		data[key] = one_hot_encode_matrix(enc, data[key], "transform")

	return data

def one_hot_encode_matrix(enc, arr, method="transform"):
	orig_shape = arr.shape
	new_arr = np.array(getattr(enc,method)(arr.flatten()[:,None])) #to have (N,1) shape
	new_arr = new_arr.reshape(*orig_shape,np.prod(new_arr.shape)//np.prod(orig_shape)) #to get (*original shape,one_hot_encode_dim)
	return new_arr