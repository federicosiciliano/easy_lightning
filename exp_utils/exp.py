import os, random, string, sys
import numpy as np
import json
import hashlib

from .cfg import ConfigObject

from .var import *

#SHOULD REMOVE ALL returns?
#Now all functions are modifying the inputs

def separate_exp_cfg(cfg):
	exp_cfg = cfg.pop(experiment_universal_key) #remove exp config
	return cfg, exp_cfg

def combine_exp_cfg(cfg, exp_cfg):
	cfg[experiment_universal_key] = exp_cfg
	return cfg

def remove_nosave_keys(cfg, exp_cfg):
	for key in exp_cfg[experiment_nosave_key]:
		exp_cfg[experiment_nosave_key][key] = cfg.pop(key)
	return cfg, exp_cfg

def restore_nosave_keys(cfg, exp_cfg):
	for key,value in exp_cfg[experiment_nosave_key].items():
		cfg[key] = value
		exp_cfg[experiment_nosave_key][key] = None #should set value to None now?
	return cfg, exp_cfg

def hash_config(cfg):
	return hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest() #hash config

def generate_random_id(key_len = 16, key_prefix = "", **kwargs):
    #string.ascii_letters + string.digits = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
	#Please note: no control over generation, so any vulgarity that may arise is purely coincidental.
	return key_prefix+''.join(random.choices(string.ascii_letters + string.digits, k=key_len))

def get_exp_folder(exp_cfg):
	exp_folder = os.path.join(get_out_folder("exp",**exp_cfg), exp_cfg["name"])
	if_not_folder_create(exp_folder)
	return exp_folder

def get_out_folder(out_type, project_folder="../", **kwargs):
	out_folder = os.path.join(project_folder,'out',out_type)
	if_not_folder_create(out_folder)
	return out_folder
	
def if_not_folder_create(folder_path):
	if not os.path.isdir(folder_path):
		print(folder_path,"not found --> creating")
		os.makedirs(folder_path)

def get_exp_list(exp_cfg):
	return get_exp_folder(exp_cfg)+"_exp_list.json" #change "_exp_list" name?

def get_exp_file(exp_cfg,exp_id):
	return os.path.join(get_exp_folder(exp_cfg), exp_id+".json")

def get_all_exp_list(exp_list_file):
	if os.path.isfile(exp_list_file):
		with open(exp_list_file, "r") as f:
			all_exps = json.load(f)
	else:
		all_exps = {}
	return all_exps

def get_experiment_id(cfg, exp_cfg=None, nosave_removed=False, reset_random_seed=True):
	exp_cfg_was_None = False
	if exp_cfg is None:
		exp_cfg_was_None = True
		cfg, exp_cfg = separate_exp_cfg(cfg)

	exp_list_file = get_exp_list(exp_cfg)
	#<NAME>_exp_list.jsonl should be a dict with
	# key = MD5 hashing of a configuration
	# value = experiment_id
	# if multiple experiments have the same hashing, the value is a list
	all_exps = get_all_exp_list(exp_list_file)

	if not nosave_removed:
		cfg, exp_cfg = remove_nosave_keys(cfg, exp_cfg)

	#This could be made better using binary search in the file, if kept sorted, instead of loading the whole dict
	cfg_hash = get_set_hashing(cfg,exp_cfg)

	exp_id = all_exps.get(cfg_hash,None)
	#print(exp_id)
	if exp_id is None: #hash not found
		exp_found = False
	else: #hash found
		if isinstance(exp_id,str): #only one exp_id 
			exp_found, exp_id = check_json(cfg, exp_cfg, [exp_id])
		elif isinstance(exp_id,list): #same hashing for multiple experiments 
			exp_found, exp_id = check_json(cfg, exp_cfg, exp_id)
		else:
			raise TypeError("READING; exp id is type ",type(exp_id))
	
	if not exp_found:
		#check if file with same id is in the folder
		if reset_random_seed:
			random.seed(None)
		exp_id = generate_random_id(**exp_cfg)
		while os.path.isfile(get_exp_file(exp_cfg,exp_id)):
			exp_id = generate_random_id(**exp_cfg)

	if not nosave_removed:
		cfg, exp_cfg = restore_nosave_keys(cfg, exp_cfg)
		
	if exp_cfg_was_None:
		combine_exp_cfg(cfg, exp_cfg)

	return exp_found, exp_id

def set_experiment_id(exp_cfg, exp_id):
	exp_cfg["experiment_id"] = exp_id

def get_set_experiment_id(cfg, exp_cfg=None, nosave_removed=False): #parameter to overwrite experiment_id?
	exp_cfg_was_None = False
	if exp_cfg is None:
		exp_cfg_was_None = True
		cfg, exp_cfg = separate_exp_cfg(cfg)

	exp_found, exp_id = get_experiment_id(cfg, exp_cfg, nosave_removed) #if "experiment_id" not in exp_cfg else True,exp_cfg["experiment_id"]
	set_experiment_id(exp_cfg, exp_id)

	if exp_cfg_was_None:
		combine_exp_cfg(cfg, exp_cfg)
	return exp_found, exp_id

def load_single_json(exp_file):
	if os.path.isfile(exp_file):
		with open(exp_file, "r") as f:
			cfg = ConfigObject(json.load(f))
	else: raise FileNotFoundError("Experiment "+exp_file+" doesn't exist")
	#if not exist, recreate?
	return cfg

def check_json(cfg, exp_cfg, exp_ids):
	for exp_id in exp_ids:
		exp_file = get_exp_file(exp_cfg,exp_id)
		new_cfg = load_single_json(exp_file)
		if cfg == new_cfg: #experiment found
			return True, exp_id
	return False, None

def get_experiments(project_folder, name, sub_cfg = None, check_type = None,  **kwargs):
	exp_folder = os.path.join(get_out_folder("exp",project_folder), name)
	all_experiments = {}
	for exp_filename in os.listdir(exp_folder):
		exp_id = exp_filename.split(".")[0]
		cfg = load_single_json(os.path.join(exp_folder,exp_filename))
		if check_type is None: #no check
			cond = True
		elif "match" in check_type.lower():
			cond = check_if_cfg_matching(cfg,**sub_cfg)
		elif "contain" in check_type.lower():
			cond = check_if_cfg_contained(cfg,sub_cfg)
		else: raise ValueError("Check type "+check_type+" doesn't exist")
		if cond: all_experiments[exp_id] = cfg
	return all_experiments

# def get_all_experiments(project_folder, name, **kwargs):
# 	return get_experiments(project_folder, name)

def check_if_cfg_matching(cfg, **kwargs):
	for key, value in kwargs.items():
		if cfg[key] != value:
			return False
	return True

def check_if_cfg_contained(cfg, sub_cfg):
	for key, value in sub_cfg.items():
		if isinstance(value,dict):
			all_good = check_if_cfg_contained(cfg[key], value) #This can raise error if key not in cfg
		else:
			all_good = cfg[key] == value #This can raise error if key not in cfg
		if not all_good: return False
	return True

def get_set_hashing(cfg,exp_cfg):
	exp_cfg["hash"] = hash_config(cfg)
	return exp_cfg["hash"]

def save_experiment(cfg, exp_cfg=None, compute_exp_id=False):
	#cfg = jsonify(copy.deepcopy(cfg) #TODO

	exp_cfg_was_None = False
	if exp_cfg is None:
		exp_cfg_was_None = True
		cfg, exp_cfg = separate_exp_cfg(cfg)

	cfg, exp_cfg = remove_nosave_keys(cfg, exp_cfg)

	experiment_id_was_missing = "experiment_id" not in exp_cfg
	if compute_exp_id or experiment_id_was_missing:
		get_set_experiment_id(cfg, exp_cfg, nosave_removed = True)

	save_hashing(cfg, exp_cfg)

	save_config(cfg, exp_cfg)

	cfg, exp_cfg = restore_nosave_keys(cfg, exp_cfg)
	
	#put option? = what to do if replacing an existing experiment

	# remove exp info from cfg
	if experiment_id_was_missing:
		exp_cfg.pop("experiment_id")
		exp_cfg.pop("hash")

	if exp_cfg_was_None:
		combine_exp_cfg(cfg, exp_cfg)

def save_hashing(cfg, exp_cfg):
	exp_list_file = get_exp_list(exp_cfg)

	#This part is needed if append to the file
	# if not os.path.isfile(exp_list_file):
	# 	print("EXPERIMENT FILE NOT FOUND: INITIALIZE IT")
	# 	open(exp_list_file, 'w').close()

	exp_id = exp_cfg["experiment_id"]

	all_exps = get_all_exp_list(exp_list_file)
	cfg_hash = get_set_hashing(cfg, exp_cfg)
	if cfg_hash in all_exps:
		prev_exp_id = all_exps[cfg_hash]
		if isinstance(prev_exp_id,str): #only one exp_id 
			if exp_id != prev_exp_id: all_exps[cfg_hash] = [prev_exp_id,exp_id]
		elif isinstance(prev_exp_id,list): #same hashing for multiple experiments 
			if exp_id not in prev_exp_id: all_exps[cfg_hash] = [*prev_exp_id,exp_id]
		else:
			raise TypeError("WRITING; exp id is type ",type(exp_id))
	else:
		all_exps[cfg_hash] = exp_id
	
	with open(exp_list_file,'w') as f:
		json.dump(all_exps,f)

def save_config(cfg, exp_cfg):
	exp_file = get_exp_file(exp_cfg, exp_cfg["experiment_id"])
	with open(exp_file,'w') as f:
		json.dump(cfg,f,sort_keys=True,indent=4)
	return cfg

# def jsonify(obj):
# 	if isinstance(obj, dict):
# 		for key,value in obj.items():
# 			obj[key] = jsonify(value)
# 	elif isinstance(obj,list):
# 		for i,value in enumerate(obj):
# 			obj[i] = jsonify(value)
# 	elif isinstance(obj,tuple):
# 		obj = list(obj)
# 		obj = jsonify(obj)
# 	elif type(obj).__module__=="numpy":
# 		obj = jsonify_numpys(obj)
	
# 	return obj

# def jsonify_numpys(value):
# 	if isinstance(value,np.ndarray):
# 		new_value = value.tolist()
# 	else:
# 		if hasattr(value,"item"):
# 			new_value = value.item()
# 		else:
# 			new_value = value
# 	return new_value


