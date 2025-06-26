import argparse  # Import the argparse module for parsing command-line arguments
import os  # Import the os module for interacting with the operating system
import re  # Import the re module for regular expressions
import yaml  # Import the yaml module for YAML file handling
from copy import deepcopy  # Import the deepcopy function from the copy module
from itertools import zip_longest

# Import some variables from another module
from .var import *  # Import specific variables from the 'var' module

def load_configuration(config_name=None, config_path=None):
    """
    Loads a configuration from a YAML file and performs various operations on it.

    :param config_name: The name of the configuration file (without the extension).
    :param config_path: The path to the directory containing the configuration file.
    :return: The loaded and processed configuration.
    """
    
    # Parse command-line arguments, especially cfg and cfg_path
    args = parse_arguments()

    # Get sub-config name and path
    sub_config_name, sub_config_path = get_cfg_info(args)

    # Set config_name and config_path if not provided
    if config_name is None:
        config_name = sub_config_name
    if config_path is None:
        config_path = sub_config_path

    # Create an empty configuration object
    cfg = ConfigObject({})

    # Set default experiment keys
    cfg = set_default_exp_key(cfg)

    # Load the YAML configuration file
    cfg = load_yaml(config_name, config_path, cfg=cfg)

    # Merge global variables into the configuration
    cfg = handle_globals(cfg)

    # Move 'nosave' keys to '__exp__'
    cfg = set_nosave(cfg)
    # Move 'sweep' keys to '__exp__'
    cfg = set_sweep(cfg)

    # Get references to other keys
    cfg = handle_relatives(cfg, cfg)

    # Change 'nosave' keys to dictionaries
    cfg = change_nosave_to_dict(cfg)
    # Change 'sweep' keys to default value and store sweep parameters in __sweep__
    cfg = change_sweep_to_default(cfg)

    # Uncomment and complete this section if needed
    # for k, v in args.items():
    #     cfg.set_composite_key(k, v)

    return cfg

def parse_arguments():
    """
    Parse command-line arguments using argparse.

    :return: A dictionary containing the parsed arguments.
    """

    # Create an ArgumentParser object for parsing command-line arguments
    cfg_parser = argparse.ArgumentParser()

    # Add command-line arguments for cfg_path and cfg with default values
    cfg_parser.add_argument("--cfg_path", default="../cfg")
    cfg_parser.add_argument("--cfg", default="config")

    # Parse the command-line arguments and return them as a dictionary
    args, unknown_args = cfg_parser.parse_known_args()

    return vars(args)


def get_cfg_info(args):
    """
    Extract the configuration name and path from the parsed arguments.

    :param args: A dictionary containing parsed arguments.
    :return: The configuration name and path.
    """

    # Extract the 'cfg' and 'cfg_path' values from the 'args' dictionary
    config_name = args.pop('cfg')
    config_path = args.pop('cfg_path')

    # Return the extracted configuration name and path
    return config_name, config_path


def load_yaml(config_name, config_path, cfg={}):
    """
    Load a YAML configuration file and merge it into the provided configuration object.

    :param config_name: The name of the configuration file (without the extension).
    :param config_path: The path to the directory containing the configuration file.
    :param cfg: The configuration object to merge with.
    :return: The merged configuration object.
    """

    # Open and read the YAML configuration file
    with open(os.path.join(config_path, config_name + ".yaml"), 'r') as f:
        # Load the YAML data from the file and merge it into the 'cfg' object
        cfg = merge_dicts(cfg, yaml.safe_load(f), preference=1)

    # Handle special keys in the configuration
    cfg = handle_special_keys_for_dicts(cfg, config_path)

    # Return the merged configuration object
    return cfg


def handle_special_keys_for_dicts(cfg, config_path):
    """
    Handle special keys in the configuration.

    :param cfg: The configuration object to process.
    :param config_path: The path to the directory containing the configuration file.
    :return: The processed configuration object.
    """

    # Import necessary functions and variables
    import_stuff(cfg)

    # Create an ArgumentParser for parsing argparse arguments
    level_parser = argparse.ArgumentParser()

    # Iterate over a copy of the keys and values in the configuration
    # Using a copy to avoid modifying the dictionary while iterating
    for key, value in cfg.copy().items():
        if key[0] == yaml_argparse_char:  # Check if it's an argparse argument
            handle_parse_args(cfg, key, level_parser)

    # Iterate over a copy of the keys and values in the configuration
    for key, value in cfg.copy().items():
        if key[0] == yaml_additional_char:  # Check if it's an additional key
            handle_additions(cfg, key, value, config_path)
        elif isinstance(value, list):
            cfg[key] = handle_special_keys_for_lists(value, config_path)
        elif isinstance(value, dict):# and key != experiment_universal_key:
            cfg[key] = handle_special_keys_for_dicts(value, config_path)

    # Iterate over a copy of the keys and values in the configuration
    for key, value in cfg.copy().items():
        if key[0] == yaml_nosave_char:  # Check if it's a nosave key
            handle_nosave(cfg, key, value)

    # Iterate over a copy of the keys and values in the configuration
    for key, value in cfg.copy().items():
        if key[0] == yaml_sweep_char:  # Check if it's a sweep key
            handle_sweep(cfg, key, value)

    # Iterate over a copy of the keys and values in the configuration
    for key, value in cfg.copy().items():
        if isinstance(value, dict):# and key != experiment_universal_key:
            cfg = raise_globals(cfg, cfg[key])
            if key != yaml_global_key:
                cfg = raise_nosave(cfg, cfg[key], key)
                cfg = raise_sweep(cfg, cfg[key], key)

    # Iterate over a copy of the keys and values in the configuration to get nosave and sweep if in a list
    for key, value in cfg.copy().items():
        if isinstance(value, list) and len(value) > 0:
            if isinstance(value[-1], dict):
                cfg = raise_nosave(cfg, cfg[key][-1], key)
                cfg = raise_sweep(cfg, cfg[key][-1], key)
                if cfg[key][-1] == {}:
                    del cfg[key][-1]

    # Remove a key if it has only one item, and that key is 'yaml_skip_key'
    if len(cfg) == 1 and yaml_skip_key in cfg:
        cfg = cfg[yaml_skip_key]

    return cfg

def handle_special_keys_for_lists(cfg_list, config_path):
    """
    Handle special keys within a list in the configuration.

    :param cfg_list: The list to process.
    :param config_path: The path to the directory containing the configuration file.
    :return: The processed list.
    """

    # Iterate over the list with a copy of its values
    for i, sub_value in enumerate(cfg_list.copy()):
        if isinstance(sub_value, dict):
            # Recursively handle special keys for dictionaries within the list
            cfg_list[i] = handle_special_keys_for_dicts(sub_value, config_path)
        elif isinstance(sub_value, list):
            # Recursively handle special keys for sublists within the list
            cfg_list[i] = handle_special_keys_for_lists(sub_value, config_path)

    # Create a dictionary to hold nosave and sweep items from the list
    nosave_sweep_dict = {}
    
    # Iterate over the list with a copy of its values
    for i, sub_value in enumerate(cfg_list.copy()):
        # Raise nosave items to the top level and use their indices as keys
        nosave_sweep_dict = raise_nosave_for_lists(nosave_sweep_dict, cfg_list[i], str(i))
        nosave_sweep_dict = raise_sweep_for_lists(nosave_sweep_dict, cfg_list[i], str(i))

    # If there are nosave or sweep items, append the nosave_sweep_dict to the list
    if len(nosave_sweep_dict) > 0:
        cfg_list.append(nosave_sweep_dict)

    return cfg_list

def import_stuff(cfg):
    """
    Import modules and objects specified in the configuration.

    :param cfg: The configuration object containing import instructions.
    """
    
    # Check if there are import instructions in the configuration
    if experiment_universal_key in cfg and yaml_imports_key in cfg[experiment_universal_key]:
        
        # Iterate over the import instructions
        for import_dict in cfg[experiment_universal_key][yaml_imports_key]:
            
            # Transform the import instruction into a dictionary if it's a string
            if isinstance(import_dict, str):
                import_dict = {"name": import_dict}
            elif isinstance(import_dict, dict):
                pass
            else:
                raise NotImplementedError  # Handle unsupported import instruction
            
            # Prepare the import_as name
            if " as " in import_dict["name"]:
                import_dict["name"], import_as = import_dict["name"].split(" as ")
            elif "fromlist" in import_dict:
                import_as = import_dict.pop("as", import_dict["fromlist"])
            else:
                import_as = import_dict["name"]
            
            # Perform the import
            app = __import__(**import_dict)
            
            # Parse import_as and assign it to the global namespace
            if isinstance(import_as, str):
                globals()[import_as] = app
            else:
                # Handle multiple imports with their associated names
                for imp_name, method_name in zip(import_as, import_dict["fromlist"]):
                    globals()[imp_name] = getattr(app, method_name)


def clean_key(key, special_char):
    """
    Clean a configuration key by removing special characters.

    :param key: The key to clean.
    :param special_char: The special character to remove.
    :return: The cleaned key.
    """

    # Check if the special character is the yaml_argparse_char
    if special_char == yaml_argparse_char:
        # Remove one or two special characters from the beginning of the key
        key = key[2:] if key[1] == yaml_argparse_char else key[1:]
    else:
        # Remove one special character from the beginning of the key
        key = key[1:]

    return key

def remove_nosave(key):
    """
    Remove the 'nosave' special character from a configuration key.

    :param key: The key to process.
    :return: The key with the 'nosave' character removed.
    """

    # Check if the key starts with the yaml_nosave_char
    if key[0] == yaml_nosave_char:
        # Remove the yaml_nosave_char from the beginning of the key
        key = key[1:]

    return key

def remove_sweep(key):
    """
    Remove the 'sweep' special character from a configuration key.

    :param key: The key to process.
    :return: The key with the 'sweep' character removed.
    """

    # Check if the key starts with the yaml_sweep_char
    if key[0] == yaml_sweep_char:
        # Remove the yaml_sweep_char from the beginning of the key
        key = key[1:]

    return key


def handle_parse_args(cfg, key, level_parser):
    """
    Handle parsing of argparse-style configuration keys.

    :param cfg: The configuration object to modify.
    :param key: The configuration key to process.
    :param level_parser: The ArgumentParser for the current level of parsing.
    """

    # Extract the parse_dict from the configuration
    parse_dict = cfg.pop(key)

    # Clean the key by removing special characters
    key = clean_key(key, yaml_argparse_char)

    # Remove the 'nosave' special character from the key
    real_key = remove_nosave(key)

    # Remove the 'sweep' special character from the key
    real_key = remove_sweep(key)

    # Get the eval function from the parse_dict, default to lambda x: x if not specified
    eval_fun = eval(parse_dict.get("eval", "lambda x: x"))

    # Evaluate the value using the eval function
    value = eval_fun(parse_dict["value"])

    # Assign the evaluated value to the cleaned key in the configuration
    cfg[real_key] = value

    # You can uncomment and complete this section if needed:
    # Handle special keys like 'value' and 'eval' for ArgumentParser
    # level_parser.add_argument(real_key, **value)
    # args_value = value.pop('value', None)  # value.pop("default")
    # del cfg[key]
    # key = key.replace(yaml_additional_char, "").strip()


def handle_additions(cfg, key, value, config_path):
    """
    Handle additional configurations specified in the configuration.

    :param cfg: The configuration object to modify.
    :param key: The configuration key to process.
    :param value: The value associated with the key.
    :param config_path: The path to the directory containing the configuration file.
    """

    # Check if the value is not a list (ignore lists of additions to sweep)
    if not isinstance(value, list):
        
        # Delete the original key from the configuration
        del cfg[key]
        
        # Clean the key by removing special characters
        key = clean_key(key, yaml_additional_char)
        
        # Remove the 'nosave' special character from the key
        real_key = remove_nosave(key)
        # Remove the 'sweep' special character from the key
        real_key = remove_sweep(key)
        
        # Determine the additional configuration path
        if value[0] == "/":
            app = value.split("/")
            additional_path = os.path.join(config_path, *app[:-1])
            value = app[-1]
        else:
            additional_path = os.path.join(config_path, real_key)

        # Load the additional configuration and merge it into the current configuration
        additional_cfg = load_yaml(value, additional_path, cfg={})

        # Raise global variables from the additional configuration
        cfg = raise_globals(cfg, additional_cfg)

        # Raise keys from the additional configuration
        cfg = raise_keys(cfg, additional_cfg)

        # If the additional configuration is not empty, add it to the current configuration
        if len(additional_cfg) > 0:
            cfg[key] = additional_cfg

        # Handle exceptions if the specialized configuration is not found (optional vs. not optional)
        # try:
        #     ...
        # except FileNotFoundError:
        #     if not optional:
        #         raise FileNotFoundError("SPECIALIZED CFG NOT FOUND:" + os.path.join(config_path, key, value))
        #     else:
        #         raise FileNotFoundError("SPECIALIZED CFG NOT FOUND, BUT OPTIONAL:", os.path.join(config_path, key, value))


def raise_globals(cfg, new_cfg):
    """
    Raise global keys from new_cfg to cfg.

    :param cfg: The target configuration object to merge global keys into.
    :param new_cfg: The source configuration object to extract global keys from.
    :return: The updated target configuration object.
    """
    
    if yaml_global_key in new_cfg:
        # Merge the yaml_global_key dictionary from new_cfg into cfg
        cfg[yaml_global_key] = merge_dicts(cfg.get(yaml_global_key, {}), new_cfg[yaml_global_key])
        
        # Remove the yaml_global_key from new_cfg
        new_cfg.pop(yaml_global_key, None)
    
    return cfg

def move_nosave(cfg, new_cfg, key, new_key):
    """
    Move nosave keys from new_cfg to cfg.

    :param cfg: The target configuration object to merge nosave keys into.
    :param new_cfg: The source configuration object to extract nosave keys from.
    :param key: The key used to prefix the moved nosave keys.
    :return: The updated target configuration object.
    """
    
    if experiment_nosave_key in new_cfg:
        # Add new_key to the nosave keys from new_cfg
        cfg[experiment_nosave_key] = cfg.get(experiment_nosave_key, []) + [new_key]
        # Remove key from the nosave keys from new_cfg
        new_cfg[experiment_nosave_key].remove(key)
    
    return cfg

def move_sweep(cfg, new_cfg, key, new_key):
    """
    Move sweep keys from new_cfg to cfg.

    :param cfg: The target configuration object to merge sweep keys into.
    :param new_cfg: The source configuration object to extract sweep keys from.
    :param key: The key used to prefix the moved sweep keys.
    :return: The updated target configuration object.
    """
    
    if experiment_sweep_key in new_cfg:
        # Add new_key to the sweep keys from new_cfg
        cfg[experiment_sweep_key] = cfg.get(experiment_sweep_key, []) + [new_key]
        # Remove key from the sweep keys from new_cfg
        new_cfg[experiment_sweep_key].remove(key)
    
    return cfg

def raise_nosave(cfg, new_cfg, key):
    """
    Raise nosave keys from new_cfg to cfg.

    :param cfg: The target configuration object to merge nosave keys into.
    :param new_cfg: The source configuration object to extract nosave keys from.
    :param key: The key used to prefix the raised nosave keys.
    :return: The updated target configuration object.
    """
    
    if experiment_nosave_key in new_cfg:
        # Prefix the nosave keys from new_cfg with the provided key
        cfg[experiment_nosave_key] = cfg.get(experiment_nosave_key, []) + [key + "." + x for x in new_cfg[experiment_nosave_key]]
        
        # Remove the experiment_nosave_key from new_cfg
        new_cfg.pop(experiment_nosave_key, None)
    
    return cfg

def raise_sweep(cfg, new_cfg, key):
    """
    Raise sweep keys from new_cfg to cfg.

    :param cfg: The target configuration object to merge sweep keys into.
    :param new_cfg: The source configuration object to extract sweep keys from.
    :param key: The key used to prefix the raised sweep keys.
    :return: The updated target configuration object.
    """
    
    if experiment_sweep_key in new_cfg:
        # Prefix the sweep keys from new_cfg with the provided key
        cfg[experiment_sweep_key] = cfg.get(experiment_sweep_key, []) + [key + "." + x for x in new_cfg[experiment_sweep_key]]
        
        # Remove the experiment_sweep_key from new_cfg
        new_cfg.pop(experiment_sweep_key, None)
    
    return cfg


def raise_nosave_for_lists(cfg, new_cfg, i):
    """
    Raise nosave keys from new_cfg to cfg when dealing with lists.

    :param cfg: The target configuration object to merge nosave keys into.
    :param new_cfg: The source configuration object to extract nosave keys from.
    :param i: The index used to prefix the raised nosave keys.
    :return: The updated target configuration object.
    """
    
    # If new_cfg is a list, check the last element to see if it's a dictionary
    if isinstance(new_cfg, list):
        new_cfg = new_cfg[-1] if isinstance(new_cfg[-1], dict) else {}
        #del new_cfg[-1]
    
    # If new_cfg is a dictionary and contains experiment_nosave_key
    if isinstance(new_cfg, dict) and experiment_nosave_key in new_cfg:
        # Prefix the nosave keys from new_cfg with the provided index
        cfg[experiment_nosave_key] = cfg.get(experiment_nosave_key, []) + [str(i) + "." + x for x in new_cfg[experiment_nosave_key]]
        
        # Remove the experiment_nosave_key from new_cfg
        new_cfg.pop(experiment_nosave_key, None)
        if new_cfg == {}:
            del new_cfg
    
    return cfg

def raise_sweep_for_lists(cfg, new_cfg, i):
    """
    Raise sweep keys from new_cfg to cfg when dealing with lists.

    :param cfg: The target configuration object to merge sweep keys into.
    :param new_cfg: The source configuration object to extract sweep keys from.
    :param i: The index used to prefix the raised sweep keys.
    :return: The updated target configuration object.
    """
    
    # If new_cfg is a list, check the last element to see if it's a dictionary
    if isinstance(new_cfg, list):
        new_cfg = new_cfg[-1] if isinstance(new_cfg[-1], dict) else {}
        #del new_cfg[-1]
    
    # If new_cfg is a dictionary and contains experiment_sweep_key
    if isinstance(new_cfg, dict) and experiment_sweep_key in new_cfg:
        # Prefix the sweep keys from new_cfg with the provided index
        cfg[experiment_sweep_key] = cfg.get(experiment_sweep_key, []) + [str(i) + "." + x for x in new_cfg[experiment_sweep_key]]
        
        # Remove the experiment_sweep_key from new_cfg
        new_cfg.pop(experiment_sweep_key, None)
        if new_cfg == {}:
            del new_cfg
    
    return cfg


def raise_keys(cfg, new_cfg):
    """
    Raise keys from new_cfg to cfg.

    :param cfg: The target configuration object to merge keys into.
    :param new_cfg: The source configuration object to extract keys from.
    :return: The updated target configuration object.
    """
    
    to_pop = set()
    
    # Iterate over keys and values in new_cfg
    for key, value in new_cfg.items():
        # Check if the key starts with yaml_raise_char
        if key[0] == yaml_raise_char:
            # Clean the key by removing yaml_raise_char
            new_key = clean_key(key, yaml_raise_char)
            # Assign the value to the cleaned key in cfg
            # If the key is already present, merge the values
            cfg = merge_dicts(cfg,{new_key: value}, preference=1)
            
            # Add the key to the set of keys to remove from new_cfg
            to_pop.add(key)

            # Check if key was in nosave keys, and raise it too
            if key in new_cfg.get(experiment_nosave_key,[]):
                move_nosave(cfg, new_cfg, key, new_key)

            # Check if key was in sweep keys, and raise it too
            if key in new_cfg.get(experiment_sweep_key,[]):
                move_sweep(cfg, new_cfg, key, new_key)
    
    # Remove the keys from new_cfg
    for key in to_pop:
        new_cfg.pop(key, None)
    
    return cfg


def handle_nosave(cfg, key, value):
    """
    Handle nosave keys by moving them from cfg to a special nosave list.

    :param cfg: The configuration object to modify.
    :param key: The nosave key to handle.
    :param value: The value associated with the nosave key.
    """
    
    # Remove the original nosave key from cfg
    del cfg[key]
    
    # Remove the 'nosave' special character from the key
    key = key[1:]

    #TODO: should take into account if the key contains other special characters (e.g. sweep)
    #Fast fix: not working cause disappears from sweep
    # if key[0] == yaml_sweep_char:
    #     key = key[1:]
    
    # Assign the value to the cleaned key in cfg
    cfg[key] = value

    
    # Add the key to the experiment_nosave_key list in cfg
    cfg[experiment_nosave_key] = [*cfg.get(experiment_nosave_key, []), key]

def handle_sweep(cfg, key, value):
    """
    Handle sweep keys by moving them from cfg to a special sweep list.

    :param cfg: The configuration object to modify.
    :param key: The sweep key to handle.
    :param value: The value associated with the sweep key.
    """
    
    # Remove the original sweep key from cfg
    del cfg[key]
    
    # Remove the 'sweep' special character from the key
    key = key[1:]
    
    # Assign the value to the cleaned key in cfg
    cfg[key] = value
    
    # Add the key to the experiment_sweep_key list in cfg
    cfg[experiment_sweep_key] = [*cfg.get(experiment_sweep_key, []), key]


def handle_globals(cfg):
    """
    Handle global keys in the configuration.

    :param cfg: The configuration object to process.
    :return: The updated configuration object.
    """
    
    # Check if yaml_global_key is present in the configuration
    if yaml_global_key in cfg:
        # Merge the global keys into the main configuration
        cfg = merge_dicts(cfg, cfg[yaml_global_key])
        # Remove the yaml_global_key from the configuration
        cfg.pop(yaml_global_key, None)
    
    return cfg


def set_nosave(cfg):
    """
    Move nosave keys to the universal configuration section.

    :param cfg: The configuration object to process.
    :return: The updated configuration object.
    """
    
    # Check if experiment_nosave_key is present in the configuration
    if experiment_nosave_key in cfg:
        # Move the nosave keys to the universal configuration section
        cfg[experiment_universal_key][experiment_nosave_key] = cfg[experiment_nosave_key]
        # Remove the experiment_nosave_key from the configuration
        cfg.pop(experiment_nosave_key, None)
    
    return cfg

def set_sweep(cfg):
    """
    Move sweep keys to the universal configuration section.

    :param cfg: The configuration object to process.
    :return: The updated configuration object.
    """
    
    # Check if experiment_sweep_key is present in the configuration
    if experiment_sweep_key in cfg:
        # Move the sweep keys to the universal configuration section
        cfg[experiment_universal_key][experiment_sweep_key] = cfg[experiment_sweep_key]
        # Remove the experiment_sweep_key from the configuration
        cfg.pop(experiment_sweep_key, None)
    
    return cfg


def handle_relatives(obj, global_cfg):
    """
    Handle relative references in the configuration.

    :param obj: The configuration object to process.
    :param global_cfg: The global configuration object.
    :return: The updated configuration object with relative references resolved.
    """
    
    if isinstance(obj, dict):
        # If obj is a dictionary, iterate over its key-value pairs
        for key, value in obj.items():
            # Recursively process each value in the dictionary
            obj[key] = handle_relatives(value, global_cfg)
    elif isinstance(obj, list):
        # If obj is a list, iterate over its elements
        for i, elem in enumerate(obj):
            # Recursively process each element in the list
            obj[i] = handle_relatives(elem, global_cfg)
    elif isinstance(obj, str):
        # If obj is a string and contains yaml_reference_char, handle the reference
        if yaml_reference_char in obj:
            return handle_reference(global_cfg, obj)
    
    return obj

def handle_reference(cfg, obj, char=yaml_reference_char):
    """
    Handle references in the configuration.

    :param cfg: The configuration object to use for resolving references.
    :param obj: The string containing the reference to handle.
    :param char: The character used to denote references (default is yaml_reference_char).
    :return: The string with resolved references.
    """
    
    # Find all matches of references in the string
    matches = [match for match in re.finditer(re.escape(char) + r"\{(.*?)\}", obj)]
    
    # If there is only one match and it spans the entire string, return the referenced value
    if len(matches) == 1:
        match = matches[0]
        start_idx, end_idx = match.span()
        if end_idx - start_idx == len(obj):
            return cfg[match.group(1)]
    
    # Initialize a new string to build the result
    new_string = ""
    start_idx, end_idx = 0, -1
    
    # Iterate over matches and replace them with referenced values
    for match in matches:
        span_start, span_end = match.span()
        new_string += obj[start_idx:span_start]
        new_string += str(cfg[match.group(1)])
        start_idx = span_end
    
    # Append any remaining part of the original string
    new_string += obj[start_idx:]
    
    return new_string

# def check_optional(key):
#     """
#     Check if the word "optional" is present in a key and remove it if found.

#     :param key: The key to check.
#     :return: A tuple containing a boolean indicating if "optional" was found and the modified key.
#     """
#     key_split = key.split(" ")
#     try:
#         # Find the index of "optional" in the split key
#         optional_id = key_split.index("optional")
#         # Remove "optional" from the split key and rejoin the remaining parts
#         return True, " ".join(key_split[:optional_id] + key_split[optional_id + 1:])
#     except ValueError:
#         # If "optional" is not found, return False and the original key
#         return False, key

def merge_dicts(a, b, path=[], preference=None, merge_lists=False):
    """
    Recursively merge two dictionaries, a and b.

    :param a: The first dictionary.
    :param b: The second dictionary.
    :param path: List representing the path in the dictionary.
    :param preference: An integer indicating preference for one dictionary over the other (0 for a, 1 for b).
                       If None, conflicts raise an exception.
    :param merge_lists: A boolean indicating whether to merge lists or raise a conflict.
    :return: The merged dictionary.
    """
    
    # Iterate over keys in b
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                # Recursively merge dictionaries for nested keys
                merge_dicts(a[key], b[key], path + [str(key)], preference=preference, merge_lists=merge_lists)
            elif a[key] == b[key]:
                # Same leaf value, no conflict
                pass
            else:
                if preference is None:
                    # Raise an exception for conflicts when preference is not specified
                    if (merge_lists or key in [experiment_nosave_key,experiment_sweep_key]) and isinstance(a[key], list) and isinstance(b[key], list):
                        # Merge lists if merge_lists is True, and if key is in [experiment_nosave_key,experiment_sweep_key] and both are lists
                        a[key] += b[key]
                    else:
                        raise Exception('Conflict at %s' % '.'.join(path + [str(key)]))
                elif preference == 0:
                    # Preference for a, keep a's value
                    pass
                elif preference == 1:
                    # Preference for b, update a's value with b's value
                    a[key] = b[key]
                else:
                    # Invalid preference value
                    raise ValueError('Preference value not in {None, 0, 1}')
        else:
            # Key is in b but not in a, add it to a
            a[key] = b[key]
    
    return a


class ConfigObject(dict):
    """
    Custom dictionary subclass that allows dictionary keys to be accessed as attributes.

    Example:
    cfg = ConfigObject({"key1": "value1", "key2": {"nested_key": "nested_value"}})
    print(cfg.key1)  # Accessing dictionary key as an attribute

    :param cfg: The initial configuration dictionary.
    """

    # __getattr__ = dict.__getitem__  # Allow accessing dictionary keys as attributes
    # __setattr__ = dict.__setitem__  # Allow setting dictionary keys as attributes
    # __delattr__ = dict.__delitem__  # Allow deleting dictionary keys as attributes

    def __init__(self, cfg):
        """
        Initialize the ConfigObject with a configuration dictionary.

        :param cfg: The initial configuration dictionary.
        """
        super().__init__(cfg)

        # If needed, every nested dict can become a ConfigObject
        # for k, v in dct.items():
        #     if isinstance(v, dict):
        #         v = ConfigObject(v)
        # Need to check lists as well?
    
    def __getitem__(self, relative_key):
        """
        Get a value from the configuration using a relative key.

        This method allows you to access nested values using dot notation.

        Example:
        cfg = ConfigObject({"key1": {"nested_key": "nested_value"}})
        value = cfg["key1.nested_key"]

        :param relative_key: The relative key to access values.
        :return: The value associated with the relative key.
        """
        #TODO: error if key doesn't exist
        value = self  # Start with the entire configuration as the initial value
        
        # Split the relative key by dot to access nested values
        for key in relative_key.split("."):
            if isinstance(value, dict):
                # If the current value is a dictionary, get the value associated with the key
                value = value.get(key, {})  # You can specify a default value here instead of an empty dictionary
            else:
                # If the current value is not a dictionary, assume it's a list and use the key as an index
                value = value[int(key)]
        
        return value
    

    def __setitem__(self, relative_key, set_value):
        """
        Set a value in the configuration using a relative key.

        This method allows you to set nested values using dot notation.

        Example:
        cfg = ConfigObject({})
        cfg["key1.nested_key"] = "nested_value"

        :param relative_key: The relative key to access and set values.
        :param set_value: The value to set.
        """

        nosave = False  # Flag to indicate whether to mark the key as nosave
        # Check if the relative key starts with yaml_nosave_char
        if relative_key[0] == yaml_nosave_char:
            relative_key = relative_key[1:]  # Remove the nosave character
            nosave = True

        sweep = False  # Flag to indicate whether to mark the key as sweep
        # Check if the relative key starts with yaml_sweep_char
        if relative_key[0] == yaml_sweep_char:
            relative_key = relative_key[1:]  # Remove the sweep character
            sweep = True

        keys = relative_key.split(".")  # Split the relative key by dot

        value = self  # Start with the entire configuration as the initial value

        # Traverse the keys, creating nested dictionaries as needed
        for key in keys[:-1]:
            if isinstance(value, dict):
                # If the current value is a dictionary, set it as the value associated with the key
                value = value.setdefault(key, {})
            else:
                # If the current value is not a dictionary, assume it's a list and use the key as an index
                value = value[int(key)]

        #TODO: doesn't work if sweep/save in higher dict
        if nosave:
            # Mark the key as nosave in the universal configuration
            self[experiment_universal_key][experiment_nosave_key][relative_key] = None

        if sweep:
            # Mark the key as sweep in the universal configuration
            self[experiment_universal_key][experiment_sweep_key][relative_key] = set_value
            set_value = set_value["default"] # Set the set_value to the default value

        # Set the final key in the configuration to the specified value
        dict.__setitem__(value, keys[-1], set_value)

        
    def pop(self, relative_key, default_value=None):
        """
        Remove and return a value from the configuration using a relative key.

        This method allows you to remove values using dot notation and return the removed value.

        Example:
        cfg = ConfigObject({"key1": {"nested_key": "nested_value"}})
        removed_value = cfg.pop("key1.nested_key")

        :param relative_key: The relative key to access and remove values.
        :param default_value: The value to return if the key is not found (default is None).
        :return: The removed value or the default value if the key is not found.
        """
        
        keys = relative_key.split(".")  # Split the relative key by dot
        value = self  # Start with the entire configuration as the initial value
        
        # Traverse the keys to locate the value to remove
        for key in keys[:-1]:
            if isinstance(value, dict):
                # If the current value is a dictionary, get the value associated with the key
                value = value.get(key, {})  # You can specify a default value here instead of an empty dictionary
            else:
                # If the current value is not a dictionary, assume it's a list and use the key as an index
                value = value[int(key)]
        
        # Remove the final key from the configuration and return its value
        return_value = value.get(keys[-1], default_value)
        dict.__delitem__(value, keys[-1])
        
        return return_value

    def sweep(self, *sweep_info):
        """
        ATTENTION: this may behave strangely with zip and multiple keys, cause zip is lazy and doesn't fully iterate through all the list, but as soon as one is exhausted, it stops. 
        This means that one key is going to have its values restored, while the others are going to be left with the last value.
        To avoid this, use itertools.zip_longest instead of zip.
        """
        sweep_dict = self["__exp__"]["__sweep__"]["parameters"]

        if isinstance(sweep_info, str):
            sweep_info = [sweep_info]

        # Iterate all keys and yield all values at the same time, one from each key
        yield from self._sweep_loop(sweep_dict, sweep_info)
    
    def _sweep_loop(self, sweep_dict, sweep_info, value_to_yield=[]):
        if len(sweep_info) == 0:
            yield value_to_yield
        else:
            relative_key = sweep_info[0]
            for value in self._sweep_single_or_multiple(sweep_dict, relative_key):
                value_to_yield.append(value)
                yield from self._sweep_loop(sweep_dict, sweep_info[1:], value_to_yield)
                value_to_yield.pop()

    def _sweep_single_or_multiple(self, sweep_dict, relative_key):
        if isinstance(relative_key, (list, tuple)):
            yield from self._sweep_multiple_keys(sweep_dict, *relative_key)
        else:
            yield from self._sweep_single_key(sweep_dict, relative_key)
        
    #TODO: bug: if the key value is modified, it will not change in next iteration
    #probably need to set self[relative_key] = value after the yield 
    def _sweep_single_key(self, sweep_dict, relative_key):
        if relative_key in sweep_dict:
            sweep_values = sweep_dict[relative_key]["values"] #TODO: more complex method if sweep is based on distribution
        else:
            sweep_values = self[relative_key]  # Get the list-like values from the configuration
        original_value = self[relative_key]
        for value in sweep_values:
            self[relative_key] = value  # Set the current value to the relative key
            yield value  # Yield the current value
        self[relative_key] = original_value  # Restore the original values in the configuration
        
    def _sweep_multiple_keys(self, sweep_dict, *relative_keys):
        """
        :param relative_keys: The relative keys to access and sweep through.
        :yield: Each value associated with the relative keys.
        """
        
        for values in zip_longest(*[self._sweep_single_key(sweep_dict, relative_key) for relative_key in relative_keys]):
            yield values

    # def sweep_multiple(self, *relative_keys):
    #     """
    #     Iterate through and yield values of multiple list-like keys in the configuration.

    #     This method allows you to sweep through multiple list-like keys, such as when multiple values are stored under the same key.

    #     Example:
    #     cfg = ConfigObject({"key1": [1, 2, 3], "key2": [4, 5, 6]})
    #     for value1, value2 in cfg.sweep_multiple(["key1", "key2"]):
    #         print(value1, value2)

    #     :param relative_keys: The relative keys to access and sweep through.
    #     :yield: Each value associated with the relative keys.
    #     """
        
    #     for values in zip_longest(*[self.sweep(relative_key) for relative_key in relative_keys]):
    #         yield values

    # def sweep_safe(self, relative_key):
    #     """
    #     Iterate through and yield values of a list-like key in the configuration.
    #     If not list-like, yield the value as is.

    #     :param relative_key: The relative key to access and sweep through.
    #     :yield: Each value associated with the relative key.
    #     """

    #     if isinstance(self[relative_key], (list, tuple)):
    #         yield from self.sweep(relative_key)
    #     else:
    #         yield self[relative_key]

    # def sweep_additions(self, relative_key, config_path="../cfg"):
    #     """
    #     Iterate through and yield values of a list-like key with additional configurations.

    #     This method allows you to sweep through a list-like key, such as when multiple values are stored under the same key,
    #     and for each value, load and merge additional configurations from specified paths.

    #     Example:
    #     cfg = ConfigObject({"key1": [1, 2, 3]})
    #     for value in cfg.sweep_additions("key1"):
    #         print(value)

    #     :param relative_key: The relative key to access and sweep through.
    #     :param config_path: The path to the directory containing additional configurations.
    #     :yield: Each value associated with the relative key.
    #     """
        
    #     addition_key = f"+{relative_key}"  # Construct the key for additional configurations
    #     orig_cfg = deepcopy(self)  # Create a deep copy of the original configuration
        
    #     # Iterate through values of the list-like key
    #     for value in self.sweep(addition_key):
    #         # Handle additions for each value and yield the value
    #         handle_additions(self, addition_key, value, config_path)
    #         yield value  # Yield the current value
    #         self = orig_cfg  # Restore the original configuration
        
    #     self = orig_cfg  # Restore the original configuration after sweeping


    def __delitem__(self, relative_key):
        """
        Delete a value from the configuration using a relative key.

        This method allows you to delete values using dot notation.

        Example:
        cfg = ConfigObject({"key1": {"nested_key": "nested_value"}})
        del cfg["key1.nested_key"]

        :param relative_key: The relative key to access and delete values.
        """
        
        value = self  # Start with the entire configuration as the initial value
        
        keys = relative_key.split(".")  # Split the relative key by dot
        
        # Traverse the keys to locate the value to delete
        for key in keys[:-1]:
            if isinstance(value, dict):
                # If the current value is a dictionary, get the value associated with the key
                value = value.get(key, {})  # You can specify a default value here instead of an empty dictionary
            else:
                # If the current value is not a dictionary, assume it's a list and use the key as an index
                value = value[int(key)]
        
        # Delete the final key from the configuration
        dict.__delitem__(value, keys[-1])

    def update(self, other_dict):
        for key,value in other_dict.items():
            self.__setitem__(key,value)


def set_default_exp_key(cfg):
    """
    Set default experiment keys in the configuration.

    This function adds default experiment keys to the universal configuration if they are not already present.

    :param cfg: The configuration to which default keys should be added.
    :return: The updated configuration with default experiment keys.
    """
    
    # Define a dictionary with default experiment keys
    default_exp_dict = {
        "name": "experiment_name",       # Name of the experiment
        "project_folder": "../",         # Project folder, used to locate folders, optional, default = "../"
        "key_len": 16,                  # Length of experiment key, optional, default = 16
        "key_prefix": "",               # Prefix for experiment key, optional, default = ""
        experiment_nosave_key: [],       # List of keys marked as nosave
        experiment_sweep_key: []       # List of keys marked as sweep
    }
    
    # Merge the default experiment keys into the universal configuration, preferring existing keys
    cfg[experiment_universal_key] = merge_dicts(cfg.get(experiment_universal_key, {}), default_exp_dict, preference=0)
    
    # Imports could be dropped - TODO?
    
    return cfg


def change_nosave_to_dict(cfg):
    """
    Change the nosave keys in the configuration to a dictionary format.

    This function converts the nosave keys in the configuration from a list format to a dictionary format.

    :param cfg: The configuration to be updated.
    :return: The updated configuration with nosave keys in dictionary format.
    """
    
    # Create an empty dictionary to hold the nosave keys
    nosave_dict = {}
    
    # Iterate through the nosave keys and add them to the dictionary with None as values
    for key in cfg[experiment_universal_key][experiment_nosave_key]:
        nosave_dict[key] = None
    
    # Replace the nosave keys list in the configuration with the nosave dictionary
    cfg[experiment_universal_key][experiment_nosave_key] = nosave_dict
    
    return cfg

def change_sweep_to_default(cfg):
    """
    Set the sweep values in the configuration to their default value.

    This function converts the sweep keys in the configuration from a list format to a dictionary format.

    :param cfg: The configuration to be updated.
    :return: The updated configuration with sweep keys in dictionary format.
    """
    
    # Create an empty dictionary to hold the sweep keys
    sweep_dict = {"parameters":{}}
    
    # Iterate through the sweep keys and add them to the dictionary with all values
    for key in cfg[experiment_universal_key][experiment_sweep_key]:
        sweep_dict["parameters"][key] = cfg[key]
        cfg[key] = cfg[key]["default"] #replace the value with the default value

    if f"{experiment_sweep_key}add" in cfg[experiment_universal_key]: 
        sweep_dict.update(cfg[experiment_universal_key][f"{experiment_sweep_key}add"])
        cfg[experiment_universal_key].pop(f"{experiment_sweep_key}add", None)

    # Replace the sweep keys list in the configuration with the sweep dictionary
    cfg[experiment_universal_key][experiment_sweep_key] = sweep_dict
    
    return cfg

