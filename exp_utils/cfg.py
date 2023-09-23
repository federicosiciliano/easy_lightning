import argparse
#import numpy as np
import os
import re
import yaml
from copy import deepcopy

from .var import * #experiment_universal_key, yaml_additional_char, yaml_global_key

def load_configuration(config_name=None,config_path=None):
	args = parse_arguments() #parse arguments, especially 
	sub_config_name,sub_config_path = get_cfg_info(args)
	if config_name is None: config_name=sub_config_name
	if config_path is None: config_path=sub_config_path

	cfg = ConfigObject({})
	cfg = set_default_exp_key(cfg)

	cfg = load_yaml(config_name,config_path,cfg=cfg)

	cfg = handle_globals(cfg) #merge globals to cfg

	cfg = set_nosave(cfg) #move nosave to __exp__

	cfg = handle_relatives(cfg, cfg) #get references to other keys

	cfg = change_nosave_to_dict(cfg)

	#when to do this?
	# for k,v in args.items():
	# 	cfg.set_composite_key(k,v)

	return cfg

def parse_arguments():
	cfg_parser = argparse.ArgumentParser()

	cfg_parser.add_argument("--cfg_path", default = "../cfg")
	cfg_parser.add_argument("--cfg", default = "config")

	args, unknown_args = cfg_parser.parse_known_args()

	return vars(args)

def get_cfg_info(args):
	config_name = args.pop('cfg')#, None)
	config_path = args.pop('cfg_path')#, None)
	return config_name,config_path

def load_yaml(config_name, config_path, cfg={}):
	with open(os.path.join(config_path,config_name+".yaml"), 'r') as f:
		cfg = merge_dicts(cfg, yaml.safe_load(f), preference=1)		

	cfg = handle_special_keys(cfg, config_path)

	return cfg

def handle_special_keys(cfg, config_path):
	#import stuff
	import_stuff(cfg)

	level_parser = argparse.ArgumentParser()
	for key,value in cfg.copy().items(): #why copy?
		if key[0]==yaml_argparse_char: #argparse argument
			handle_parse_args(cfg, key, level_parser)
	
	#if isinstance(cfg,dict):
	for key,value in cfg.copy().items(): #why copy?
		if key[0]==yaml_additional_char:
			handle_additions(cfg, key, value, config_path)
		elif isinstance(value,list):
			cfg[key] = handle_special_keys_for_lists(value, config_path)
		elif isinstance(value,dict):
			cfg[key] = handle_special_keys(value, config_path)

	for key,value in cfg.copy().items(): #why copy?
		if key[0]==yaml_nosave_char:
			handle_nosave(cfg, key, value)

	for key,value in cfg.copy().items(): #why copy?
		if isinstance(value,dict):
			cfg = raise_globals(cfg, cfg[key])
			if key!=yaml_global_key:
				cfg = raise_nosave(cfg, cfg[key], key)
		if isinstance(value,list) and len(value)>0:
			#cfg = raise_globals(cfg, cfg[key]) #TO IMPLEMENT
			if isinstance(value[-1],dict):
				cfg = raise_nosave(cfg, cfg[key][-1], key)
				if cfg[key][-1] == {}:
					del cfg[key][-1]

	if len(cfg) == 1 and yaml_skip_key in cfg:
		cfg = cfg[yaml_skip_key]

	return cfg

def handle_special_keys_for_lists(cfg_list, config_path):
	for i, sub_value in enumerate(cfg_list.copy()):  #why copy?
		if isinstance(sub_value,dict):
			cfg_list[i] = handle_special_keys(sub_value, config_path)
		elif isinstance(sub_value,list):
			cfg_list[i] = handle_special_keys_for_lists(sub_value, config_path)

	no_save_dict = {}
	for i, sub_value in enumerate(cfg_list.copy()):  #why copy?
		no_save_dict = raise_nosave_for_lists(no_save_dict, cfg_list[i], str(i))

	if len(no_save_dict)>0:
		cfg_list.append(no_save_dict)
				
	return cfg_list

def import_stuff(cfg):
	if experiment_universal_key in cfg and yaml_imports_key in cfg[experiment_universal_key]:
		for import_dict in cfg[experiment_universal_key][yaml_imports_key]:
			#transform in dict
			if isinstance(import_dict,str):
				import_dict = {"name": import_dict}
			elif isinstance(import_dict,dict):
				pass
			else:
				raise NotImplementedError
			
			#prepare import_as
			if " as " in import_dict["name"]:
				import_dict["name"],import_as = import_dict["name"].split(" as ")
			elif "fromlist" in import_dict:
				import_as = import_dict.pop("as",import_dict["fromlist"])
			else:
				import_as = import_dict["name"]
			
			#import
			app = __import__(**import_dict)
			
			#parse import_as
			if isinstance(import_as,str):
				globals()[import_as] = app
			else:
				for imp_name,method_name in zip(import_as,import_dict["fromlist"]):
					globals()[imp_name] = getattr(app,method_name)

def clean_key(key,special_char):
	if special_char==yaml_argparse_char:
		key = key[2:] if key[1]==yaml_argparse_char else key[1:]
	else:
		key = key[1:] #.replace(yaml_additional_char,"").strip()
	return key

def remove_nosave(key):
	if key[0]==yaml_nosave_char:
		key = key[1:]
	return key

def handle_parse_args(cfg, key, level_parser):
	#!!!TO IMPLEMENT!!!
	parse_dict = cfg.pop(key)
	
	key = clean_key(key,yaml_argparse_char)
	real_key = remove_nosave(key)

	eval_fun = eval(parse_dict.get("eval","lambda x: x"))
	# should instead assing eval to type in parse_dict and pass to argparse?

	value = eval_fun(parse_dict["value"]) #should check if value exists?

	cfg[real_key] = value

	# parse dots in names
	
	#special keys: value, eval
	# level_parser.add_argument(real_key, **value)
	# args_value = value.pop('value', None) #value.pop("default")
	
	# del cfg[key]
	# key = key.replace(yaml_additional_char,"").strip()

def handle_additions(cfg, key, value, config_path):
	if not isinstance(value, list): #ignore list of additions to sweep
		del cfg[key]
		key = clean_key(key,yaml_additional_char)
		real_key = remove_nosave(key)
		
		#optional,key = check_optional(key)

		#if isinstance(value, list):
		#	value = value[0] #####!!!!! CHANGE TO HANDLE LISTS!!!!!!#####
		#if list? Load and merge all configurations?
		#try:

		#get additional path
		if value[0] == "/":
			app = value.split("/")
			additional_path = os.path.join(config_path,*app[:-1])
			value = app[-1]
		else:
			additional_path = os.path.join(config_path,real_key)

		additional_cfg = load_yaml(value,additional_path,cfg={}) #need to specify cfg, otherwise scope problem!

		cfg = raise_globals(cfg, additional_cfg)

		#cfg = raise_nosave(cfg, additional_cfg, key)

		cfg = raise_keys(cfg, additional_cfg)

		if len(additional_cfg)>0:
				cfg[key] = additional_cfg
		#except FileNotFoundError:
			# if not optional:
			# 	raise FileNotFoundError("SPECIALIZED CFG NOT FOUND:"+os.path.join(config_path,key,value))
			# else:
		#	raise FileNotFoundError("SPECIALIZED CFG NOT FOUND, BUT OPTIONAL:",os.path.join(config_path,key,value))

#raise yaml_global_key dict from new_cfg to cfg
def raise_globals(cfg, new_cfg):
	if yaml_global_key in new_cfg:
		cfg[yaml_global_key] = merge_dicts(cfg.get(yaml_global_key,{}),new_cfg[yaml_global_key])
		new_cfg.pop(yaml_global_key,None)
	return cfg

def raise_nosave(cfg, new_cfg, key):
	if experiment_nosave_key in new_cfg:
		cfg[experiment_nosave_key] = cfg.get(experiment_nosave_key,[]) + [key+"."+x for x in new_cfg[experiment_nosave_key]]
		new_cfg.pop(experiment_nosave_key,None)
	return cfg

def raise_nosave_for_lists(cfg, new_cfg, i):
	if isinstance(new_cfg,list):
		app = new_cfg[-1] if isinstance(new_cfg[-1],dict) else {}
		del new_cfg[-1]
		new_cfg = app
	if isinstance(new_cfg,dict):
		if experiment_nosave_key in new_cfg:
			cfg[experiment_nosave_key] = cfg.get(experiment_nosave_key,[]) + [str(i)+"."+x for x in new_cfg[experiment_nosave_key]]
			new_cfg.pop(experiment_nosave_key,None)
	return cfg
	
def raise_keys(cfg, new_cfg):
	to_pop = set()
	for key,value in new_cfg.items():
		if key[0] == yaml_raise_char:
			new_key = clean_key(key,yaml_raise_char)
			cfg[new_key] = value
			to_pop.add(key)
	for key in to_pop:
		new_cfg.pop(key,None)
	return cfg

def handle_nosave(cfg, key, value):
	del cfg[key]
	key = key[1:]
	cfg[key] = value

	cfg[experiment_nosave_key] = [*cfg.get(experiment_nosave_key,[]),key]

def handle_globals(cfg):
	if yaml_global_key in cfg:
		cfg = merge_dicts(cfg,cfg[yaml_global_key])
		cfg.pop(yaml_global_key,None)
	return cfg

def set_nosave(cfg):
	if experiment_nosave_key in cfg:
		cfg[experiment_universal_key][experiment_nosave_key] = cfg[experiment_nosave_key]
		cfg.pop(experiment_nosave_key,None)
	return cfg

def handle_relatives(obj, global_cfg):
	if isinstance(obj, dict):
		for key,value in obj.items():
			obj[key] = handle_relatives(value, global_cfg)
	elif isinstance(obj, list):
		for i, elem in enumerate(obj):
			obj[i] = handle_relatives(elem, global_cfg)
	elif isinstance(obj, str):
		if yaml_reference_char in obj:
			return handle_reference(global_cfg, obj)
	return obj


#TODO: should raise error if key not found
def handle_reference(cfg, obj, char=yaml_reference_char):
	matches = [match for match in re.finditer(re.escape(char)+r"\{(.*?)\}",obj)]
	if len(matches) == 1:
		match = matches[0]
		start_idx, end_idx = match.span()
		if end_idx-start_idx == len(obj):
			return cfg[match.group(1)]
	
	new_string = ""
	start_idx, end_idx = 0,-1
	for match in matches:
		span_start, span_end = match.span()
		new_string += obj[start_idx:span_start]
		new_string += str(cfg[match.group(1)])
		start_idx = span_end
	new_string += obj[start_idx:]
	return new_string
	

# def check_optional(key):
# 	key_split = key.split(" ")
# 	try:
# 		optional_id = key_split.index("optional")
# 		return True," ".join(key_split[:optional_id] + key_split[optional_id+1:])
# 	except ValueError:
# 		return False, key


def merge_dicts(a, b, path = [], preference = None, merge_lists=False):
	#preference = 0 --> prefer a
	#preference = 1 --> prefer b
	#preference = None (default) --> raise conflict
	for key in b:
		if key in a:
			if isinstance(a[key], dict) and isinstance(b[key], dict):
				merge_dicts(a[key], b[key], path + [str(key)], preference=preference, merge_lists=merge_lists)
			elif a[key] == b[key]:
				pass # same leaf value
			else:
				if preference is None:
					if (merge_lists or key=="__nosave__") and isinstance(a[key],list) and isinstance(b[key],list):
						a[key] += b[key]
					else:
						raise Exception('Conflict at %s' % '.'.join(path + [str(key)]))
				elif preference==0:
					pass
				elif preference==1:
					a[key] = b[key]
				else:
					raise ValueError('Preference value not in {None,1,2}')
		else:
			a[key] = b[key]
	return a


#TODO? Add inverse_get: get all keys except those in a list/set
class ConfigObject(dict):
	__getattr__ = dict.__getitem__ #can't use super() here --> RuntimeError: super(): no arguments
	__setattr__ = dict.__setitem__
	__delattr__ = dict.__delitem__
	# Now it's possible to access dictionary keys as attributes (i.e. dict.key)

	def __init__(self, cfg):
		super().__init__(cfg)
		#If needed, every nested dict can become a ConfigObject
		# for k,v in dct.items():
		# 	if isinstance(v,dict):
		# 		v = ConfigObject(v)
		#need also to check lists?

	def __getitem__(self, relative_key): #get_composite_key
		value = self
		for key in relative_key.split("."):
			if isinstance(value,dict):
				value = value.get(key,{}) #should raise error instead of default?
			else:
				value = value[int(key)]
		return value

	def __setitem__(self, relative_key, set_value): #set_composite_key
		no_save = False
		if relative_key[0] == yaml_nosave_char:
			relative_key = relative_key[1:]
			no_save=True
		keys = relative_key.split(".")
		value = self
		for key in keys[:-1]:
			if isinstance(value,dict):
				value = value.setdefault(key, {})
			else:
				value = value[int(key)]
		dict.__setitem__(value,keys[-1],set_value)
		if no_save:
			self[experiment_universal_key][experiment_nosave_key][relative_key] = None

	def pop(self, relative_key, default_value=None): #get_composite_key
		keys = relative_key.split(".")
		value = self
		for key in keys[:-1]:
			if isinstance(value,dict):
				value = value.get(key,{}) #should raise error instead of default?
			else:
				value = value[int(key)]
		return_value = value.get(keys[-1],default_value)
		dict.__delitem__(value,keys[-1])
		return return_value

	def sweep(self, relative_key):
		values = self[relative_key]#.copy()
		for value in values:
			self[relative_key] = value
			yield value
		self[relative_key] = values

	def sweep_additions(self, relative_key, config_path="../cfg"):
		addition_key = f"+{relative_key}"
		orig_cfg = deepcopy(self)
		for value in self.sweep(addition_key):
			handle_additions(self, addition_key, value, config_path)
			yield value
			self = orig_cfg
		self = orig_cfg

	#TODO
	# def __delitem__(self, relative_key): #del_composite_key
	# 	value = self
	# 	for key in relative_key.split("."):
	# 		value = value.get(key,{}) #should raise error instead of default?
	# 	return value

def set_default_exp_key(cfg):
	default_exp_dict = {"name": "experiment_name", #name of the experiment
		     			"project_folder": "../", #project folder, used to locate folders, optional, default = "../"
						"key_len": 16, #Length of experiment key, optional, default = 16
						"key_prefix": "", #Prefix for experiment key, optional, default = ""
						experiment_nosave_key: []
						}
	cfg[experiment_universal_key] = merge_dicts(cfg.get(experiment_universal_key,{}), default_exp_dict, preference=0)
	
	#imports could be dropped --> TODO?

	return cfg

def change_nosave_to_dict(cfg):
	nosave_dict = {}
	for key in cfg[experiment_universal_key][experiment_nosave_key]:
		nosave_dict[key] = None
	cfg[experiment_universal_key][experiment_nosave_key] = nosave_dict
	return cfg