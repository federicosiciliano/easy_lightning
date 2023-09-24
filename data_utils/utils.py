import numpy as np

def separate_single_column(data, separate_keys = {"x": ["y","x"]}, column_ids = {"x":[-1]}, del_after_separate=False): #use_pandas = False,
	for merged_var,separate_vars in separate_keys.items():
		for key,col in zip(separate_vars[:-1],column_ids[merged_var]):
			data[key] = data[merged_var][:,col] #.iloc
		data[separate_vars[-1]] = data[merged_var][:,np.array([i for i in range(data[merged_var].shape[1]) if i not in column_ids])] #.iloc
		if del_after_separate: del data[merged_var]
	return data