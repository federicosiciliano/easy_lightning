import os, random, string
import numpy as np
import json
import hashlib
import copy

from .cfg import ConfigObject

from .var import *

# All function now modify the input even as side effect

# This function separates the experiment configuration from the main configuration.
def separate_exp_cfg(cfg):
    """
    Separates the experiment configuration from the main configuration.

    This function extracts the experiment configuration from the main configuration dictionary.

    :param cfg: The combined configuration dictionary.
    :return: A tuple containing the main configuration dictionary and the experiment configuration.
    """
    # exp_cfg = cfg.pop(experiment_universal_key)  # Remove exp config
    exp_cfg = cfg[experiment_universal_key]  # Remove exp config
    cfg = ConfigObject({key:value for key,value in cfg.items() if key != experiment_universal_key})
    return cfg, exp_cfg

# This function combines the experiment configuration with the main configuration.
def combine_exp_cfg(cfg, exp_cfg):
    """
    Combines the experiment configuration with the main configuration.

    This function merges the experiment configuration back into the main configuration dictionary.

    :param cfg: The main configuration dictionary.
    :param exp_cfg: The experiment configuration.
    :return: The combined configuration dictionary.
    """
    cfg[experiment_universal_key] = exp_cfg
    return cfg

def get_clean_cfg(cfg):
    """
    Get a clean copy of the configuration.

    :param cfg: The configuration to clean.
    :return: A clean copy of the configuration.
    """
    
    # Separate the experiment configuration from the main configuration
    cfg, exp_cfg = separate_exp_cfg(copy.deepcopy(cfg))

    # Remove keys that should not be saved in the experiment
    cfg, exp_cfg = remove_nosave_keys(cfg, exp_cfg)

    return cfg

# This function removes keys from the configuration that should not be saved.
def remove_nosave_keys(cfg, exp_cfg):
    """
    Removes keys from the configuration that should not be saved.

    This function removes keys specified as "nosave" in the experiment configuration from the main configuration.

    :param cfg: The main configuration dictionary.
    :param exp_cfg: The experiment configuration.
    :return: A tuple containing the updated main configuration dictionary and the experiment configuration.
    """
    cfg = copy.deepcopy(cfg)
    for key in exp_cfg[experiment_nosave_key]:
        exp_cfg[experiment_nosave_key][key] = cfg.pop(key)
    return cfg, exp_cfg

# This function restores keys that were previously removed as "nosave."
def restore_nosave_keys(cfg, exp_cfg):
    """
    Restores keys that were previously removed as "nosave."

    This function adds back keys to the main configuration that were previously removed as "nosave."

    :param cfg: The main configuration dictionary.
    :param exp_cfg: The experiment configuration.
    :return: A tuple containing the updated main configuration dictionary and the experiment configuration.
    """
    for key, value in exp_cfg[experiment_nosave_key].items():
        cfg[key] = value
        exp_cfg[experiment_nosave_key][key] = None  # Set value to None now?
    return cfg, exp_cfg

# This function calculates a hash of the configuration.
def hash_config(cfg):
    """
    Calculates a hash of the configuration.

    This function computes an MD5 hash of the configuration dictionary.

    :param cfg: The configuration dictionary to be hashed.
    :return: The MD5 hash of the configuration.
    """
    return hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()

# This function generates a random ID.
def generate_random_id(key_len=16, key_prefix="", **kwargs):
    """
    Generates a random ID.

    This function creates a random alphanumeric ID with a specified length and optional prefix.

    :param key_len: The length of the random ID.
    :param key_prefix: An optional prefix for the ID.
    :return: The generated random ID.
    """
    return key_prefix + ''.join(random.choices(string.ascii_letters + string.digits, k=key_len))

# This function gets the experiment folder based on the experiment configuration.
def get_exp_folder(exp_cfg):
    """
    Gets the experiment folder based on the experiment configuration.

    This function constructs the path to the experiment folder using the experiment configuration.

    :param exp_cfg: The experiment configuration.
    :return: The path to the experiment folder.
    """
    exp_folder = os.path.join(get_out_folder("exp", **exp_cfg), exp_cfg["name"])
    if_not_folder_create(exp_folder)
    return exp_folder

# This function gets the output folder.
def get_out_folder(out_type, project_folder="../", **kwargs):
    """
    Gets the output folder path.

    This function constructs the path to the output folder based on the output type and project folder.

    :param out_type: The type of output folder (e.g., "exp").
    :param project_folder: The project folder path.
    :return: The path to the output folder.
    """
    out_folder = os.path.join(project_folder, 'out', out_type)
    if_not_folder_create(out_folder)
    return out_folder

# This function creates a folder if it doesn't exist.
def if_not_folder_create(folder_path):
    """
    Creates a folder if it doesn't exist.

    This function checks if a folder exists at the specified path and creates it if it doesn't.

    :param folder_path: The path to the folder.
    """
    if not os.path.isdir(folder_path):
        try:
            os.makedirs(folder_path)
            print(folder_path, "not found --> created")
        # catch folder already created by another process
        except FileExistsError:
            pass


# This function constructs the path to the experiment list file.
def get_exp_list(exp_cfg):
    """
    Constructs the path to the experiment list file.

    This function generates the file path for storing the list of experiments.

    :param exp_cfg: The experiment configuration.
    :return: The file path for the experiment list.
    """
    return get_exp_folder(exp_cfg) + "_exp_list.json"  # Change "_exp_list" name?

# This function constructs the path to an experiment file.
def get_exp_file(exp_cfg, exp_id):
    """
    Constructs the path to an experiment file.

    This function generates the file path for an individual experiment configuration.

    :param exp_cfg: The experiment configuration.
    :param exp_id: The ID of the experiment.
    :return: The file path for the experiment configuration file.
    """
    return os.path.join(get_exp_folder(exp_cfg), exp_id + ".json")

# This function retrieves all experiment data from the experiment list file.
def get_all_exp_list(exp_list_file):
    """
    Retrieves all experiment data from the experiment list file.

    This function reads and returns the content of the experiment list file, which contains experiment IDs.

    :param exp_list_file: The path to the experiment list file.
    :return: A dictionary containing experiment IDs and their corresponding configurations.
    """
    if os.path.isfile(exp_list_file):
        with open(exp_list_file, "r") as f:
            all_exps = json.load(f)
    else:
        all_exps = {}
    return all_exps


# This function retrieves the experiment ID based on the configuration.
def get_experiment_id(cfg, exp_cfg=None, nosave_removed=False, reset_random_seed=True, jsonify=True):
    """
    Retrieves the experiment ID based on the configuration.

    This function retrieves the experiment ID associated with the provided configuration.

    :param cfg: The main configuration dictionary.
    :param exp_cfg: The experiment configuration.
    :param nosave_removed: A flag indicating whether nosave keys have been removed.
    :param reset_random_seed: A flag indicating whether to reset the random seed.
    :return: A tuple containing a flag indicating whether the experiment was found and the experiment ID.
    """
    exp_cfg_was_None = False
    if exp_cfg is None:
        exp_cfg_was_None = True
        cfg, exp_cfg = separate_exp_cfg(cfg)
        # If exp_cfg is not provided, separate it from the main configuration

    exp_list_file = get_exp_list(exp_cfg)
    # Generate the file path for the experiment list file

    all_exps = get_all_exp_list(exp_list_file)
    # Retrieve all experiment data from the experiment list file

    if not nosave_removed:
        cfg, exp_cfg = remove_nosave_keys(cfg, exp_cfg)
        # Remove nosave keys from the configuration if necessary

    # Jsonify the configuration
    if jsonify:
        cfg = jsonify_cfg(copy.deepcopy(cfg))

    # This could be made better using binary search in the file, if kept sorted, instead of loading the whole dict
    cfg_hash = get_set_hashing(cfg, exp_cfg)
    # Calculate the hash of the configuration

    exp_id = all_exps.get(cfg_hash, None)
    # Try to retrieve the experiment ID based on the hash

    if exp_id is None:  # Hash not found
        exp_found = False
    else:  # Hash found
        if isinstance(exp_id, str):  # Only one exp_id
            exp_found, exp_id = check_json(cfg, exp_cfg, [exp_id])
        elif isinstance(exp_id, list):  # Same hashing for multiple experiments
            exp_found, exp_id = check_json(cfg, exp_cfg, exp_id)
        else:
            raise TypeError("READING; exp id is type ", type(exp_id))
            # Handle different types of experiment IDs

    if not exp_found:
        # Check if a file with the same ID is in the folder
        if reset_random_seed:
            random.seed(None)
            # Reset the random seed for generating a new experiment ID
        exp_id = generate_random_id(**exp_cfg)
        # Generate a new random experiment ID
        while os.path.isfile(get_exp_file(exp_cfg, exp_id)):
            exp_id = generate_random_id(**exp_cfg)
            # Ensure the generated ID is unique

    if not nosave_removed:
        cfg, exp_cfg = restore_nosave_keys(cfg, exp_cfg)
        # Restore the removed nosave keys if necessary

    if exp_cfg_was_None:
        cfg = combine_exp_cfg(cfg, exp_cfg)
        # Re-combine the main configuration and experiment configuration if exp_cfg was initially None

    return exp_found, exp_id

def set_experiment_id(exp_cfg, exp_id):
    """
    Set the experiment ID in the experiment configuration.

    :param exp_cfg: The experiment configuration.
    :param exp_id: The experiment ID to set.
    """
    exp_cfg["experiment_id"] = exp_id

def get_set_experiment_id(cfg, exp_cfg=None, nosave_removed=False, jsonify=True):
    """
    Get and set the experiment ID in the configuration.

    :param cfg: The configuration.
    :param exp_cfg: The experiment configuration.
    :param nosave_removed: Flag to indicate if nosave keys have been removed from the configuration.
    :return: A tuple (exp_found, exp_id) where exp_found is True if the experiment was found,
             and exp_id is the experiment ID.
    """
    exp_cfg_was_None = False
    if exp_cfg is None:
        exp_cfg_was_None = True
        cfg, exp_cfg = separate_exp_cfg(cfg)  # Separate the experiment configuration from the main configuration.

    exp_found, exp_id = get_experiment_id(cfg, exp_cfg, nosave_removed, jsonify)  # Get or generate an experiment ID.
    set_experiment_id(exp_cfg, exp_id)  # Set the experiment ID in the experiment configuration.

    if exp_cfg_was_None:
        cfg = combine_exp_cfg(cfg, exp_cfg)  # Combine the experiment configuration back into the main configuration.

    return exp_found, exp_id

# This function loads a single experiment configuration from a JSON file.
def load_single_json(exp_file):
    """
    Loads a single experiment configuration from a JSON file.

    This function reads and parses a single experiment configuration from a JSON file.

    :param exp_file: The path to the JSON file containing the experiment configuration.
    :return: The loaded experiment configuration as a dictionary.
    """
    if os.path.isfile(exp_file):
        with open(exp_file, "r") as f:
            cfg = ConfigObject(json.load(f))
            # Open and parse the JSON file into a dictionary using the ConfigObject class
    else:
        raise FileNotFoundError("Experiment " + exp_file + " doesn't exist")
        # Raise an exception if the file doesn't exist

    return cfg

# This function checks if a configuration matches specified key-value pairs.
def check_json(cfg, exp_cfg, exp_ids):
    """
    Checks if a configuration matches specified experiment IDs.

    This function compares the provided configuration with experiment configurations associated with specific IDs.

    :param cfg: The configuration to be checked.
    :param exp_cfg: The experiment configuration.
    :param exp_ids: A list of experiment IDs to check against.
    :return: A tuple containing a flag indicating whether a match was found and the matched experiment ID (or None).
    """
    for exp_id in exp_ids:
        exp_file = get_exp_file(exp_cfg, exp_id)
        new_cfg = load_single_json(exp_file)
        # Load the experiment configuration associated with the current exp_id

        if cfg == new_cfg:  # Experiment found
            return True, exp_id
            # Return True and the matching experiment ID if the configuration matches

    return False, None
    # Return False if no matching experiment was found among the provided IDs


# This function get experiments from a specified folder.
def get_experiments(name, project_folder="../", sub_cfg=None, check_type=None, **kwargs):
    """
    Get experiments from a specified folder.

    :param name: The name of the experiment.
    :param project_folder: The project folder where experiments are stored.
    :param sub_cfg: A subset of configuration to match or contain.
    :param check_type: Type of check to perform (match or contain).
    :param kwargs: Additional keyword arguments.
    :return: A dictionary containing experiment IDs and their corresponding configurations.
    """
    exp_folder = os.path.join(get_out_folder("exp", project_folder), name)
    all_experiments = {}

    for exp_filename in os.listdir(exp_folder):
        exp_id = exp_filename.split(".")[0]
        cfg = load_single_json(os.path.join(exp_folder, exp_filename))

        if check_type is None:
            # No check specified, include all experiments.
            cond = True
        elif "match" in check_type.lower():
            # Check if the experiment configuration matches the specified sub-configuration.
            cond = check_if_cfg_matching(cfg, **sub_cfg)
        elif "contain" in check_type.lower():
            # Check if the experiment configuration contains the specified sub-configuration.
            cond = check_if_cfg_contained(cfg, sub_cfg)
        else:
            raise ValueError("Check type " + check_type + " doesn't exist")

        if cond:
            # Include the experiment in the result if it meets the condition.
            all_experiments[exp_id] = cfg

    return all_experiments


# This function checks if a configuration matches specified key-value pairs.
def check_if_cfg_matching(cfg, **kwargs):
    """
    Checks if a configuration matches specified key-value pairs.

    This function compares a configuration with specified key-value pairs to determine if it matches.

    :param cfg: The configuration to be checked.
    :param kwargs: Key-value pairs to match against the configuration.
    :return: True if the configuration matches all specified key-value pairs, False otherwise.
    """
    for key, value in kwargs.items():
        if cfg[key] != value:
            return False
            # Return False if any key-value pair doesn't match

    return True
    # Return True if all specified key-value pairs match in the configuration


# This function checks if a configuration contains specified key-value pairs.
def check_if_cfg_contained(cfg, sub_cfg):
    """
    Checks if a configuration contains specified key-value pairs.

    This function determines if the provided configuration contains all specified key-value pairs.

    :param cfg: The configuration to be checked.
    :param sub_cfg: Key-value pairs to check for containment in the configuration.
    :return: True if the configuration contains all specified key-value pairs, False otherwise.
    """
    for key, value in sub_cfg.items():
        if isinstance(value, dict):
            all_good = check_if_cfg_contained(cfg[key], value)
            # Recursively check containment for nested dictionaries within the configuration
        else:
            all_good = cfg[key] == value
            # Check if the value associated with the key matches in the configuration

        if not all_good:
            return False
            # Return False if any key-value pair is not contained in the configuration

    return True
    # Return True if all specified key-value pairs are contained in the configuration


# This function calculates and sets the hash of a configuration.
def get_set_hashing(cfg, exp_cfg):
    """
    Calculates and sets the hash of a configuration.

    This function calculates the hash of the provided configuration and sets it in the experiment configuration.

    :param cfg: The configuration to be hashed.
    :param exp_cfg: The experiment configuration where the hash will be set.
    :return: The calculated hash.
    """
    exp_cfg["hash"] = hash_config(cfg)
    # Calculate the hash of the configuration using the hash_config function and set it in exp_cfg

    return exp_cfg["hash"]
    # Return the calculated hash


# This function saves an experiment configuration.
def save_experiment(cfg, exp_cfg=None, compute_exp_id=False, jsonify=True):
    """
    Saves an experiment configuration.

    This function saves an experiment configuration, including computing an experiment ID and handling related tasks.

    :param cfg: The experiment configuration to be saved.
    :param exp_cfg: The experiment-specific configuration.
    :param compute_exp_id: Whether to compute and set a new experiment ID.
    """

    exp_cfg_was_None = False
    if exp_cfg is None:
        exp_cfg_was_None = True
        cfg, exp_cfg = separate_exp_cfg(cfg)
        # If exp_cfg is not provided, separate it from cfg

    cfg, exp_cfg = remove_nosave_keys(cfg, exp_cfg)
    # Remove keys that should not be saved in the experiment

    experiment_id_was_missing = "experiment_id" not in exp_cfg
    # Check if experiment_id is missing in exp_cfg

    if compute_exp_id or experiment_id_was_missing:
        get_set_experiment_id(cfg, exp_cfg, nosave_removed=True, jsonify=jsonify)
        # Compute and set a new experiment ID if requested or if it was missing
    elif jsonify:
        cfg = jsonify_cfg(copy.deepcopy(cfg))

    save_hashing(cfg, exp_cfg)
    # Calculate and save the hash of the configuration

    save_config(cfg, exp_cfg)
    # Save the experiment configuration to a JSON file

    cfg, exp_cfg = restore_nosave_keys(cfg, exp_cfg)
    # Restore the keys that were removed for saving

    # Put option? = what to do if replacing an existing experiment

    if experiment_id_was_missing:
        exp_cfg.pop("experiment_id")
        exp_cfg.pop("hash")
        # Remove the experiment_id and hash if they were missing initially

    if exp_cfg_was_None:
        cfg = combine_exp_cfg(cfg, exp_cfg)
        # If exp_cfg was initially None, combine it back with cfg

# This function saves the experiment's hash in the experiment list file.
def save_hashing(cfg, exp_cfg):
    """
    Save the experiment's hash in the experiment list file.

    :param cfg: The experiment configuration.
    :param exp_cfg: The experiment-specific configuration.
    """
    # Get the path to the experiment list file
    exp_list_file = get_exp_list(exp_cfg)

    # Get the experiment ID from the experiment configuration
    exp_id = exp_cfg["experiment_id"]

    # Retrieve all existing experiments from the experiment list file
    all_experiments = get_all_exp_list(exp_list_file)

    # Calculate the hash of the configuration
    cfg_hash = get_set_hashing(cfg, exp_cfg)

    # Check if the hash already exists in the experiment list
    if cfg_hash in all_experiments:
        existing_exp_id = all_experiments[cfg_hash]

        # Check if there's a single experiment ID
        if isinstance(existing_exp_id, str):
            if exp_id != existing_exp_id:
                all_experiments[cfg_hash] = [existing_exp_id, exp_id]
                # Update with multiple IDs if necessary
        # Check if there are multiple experiment IDs with the same hash
        elif isinstance(existing_exp_id, list):
            if exp_id not in existing_exp_id:
                all_experiments[cfg_hash] = [*existing_exp_id, exp_id]
                # Append the new experiment ID
        else:
            raise TypeError(f"Unexpected exp_id type: {type(exp_id)}")

    else:
        all_experiments[cfg_hash] = exp_id
        # Add the new experiment ID to the list if it doesn't exist

    # Write the updated experiment list to the file
    with open(exp_list_file, 'w') as f:
        json.dump(all_experiments, f)

# This function saves the experiment configuration to a JSON file.
def save_config(cfg, exp_cfg):
    """
    Save the experiment configuration to a JSON file with proper formatting.

    :param cfg: The experiment configuration.
    :param exp_cfg: The experiment-specific configuration.
    :return: The saved configuration.
    """
    # Get the path to the experiment-specific JSON file
    exp_file = get_exp_file(exp_cfg, exp_cfg["experiment_id"])

    # Write the configuration to the file with sorted keys and indentation
    with open(exp_file, 'w') as f:
        json.dump(cfg, f, sort_keys=True, indent=4)

    return cfg  # Return the saved configuration


### TO DOCUMENT BETTER!!!
# This code provides functions for recursively converting non-JSON-serializable data structures to JSON-serializable format.
def jsonify_cfg(obj):
    if isinstance(obj, dict): # Recursively convert dictionaries
        for key, value in obj.items():
            obj[key] = jsonify_cfg(value)
    elif isinstance(obj, list): # Recursively convert lists
        for i, value in enumerate(obj):
            obj[i] = jsonify_cfg(value)
    elif isinstance(obj, tuple): # Recursively convert tuples
        obj = list(obj)
        obj = jsonify_cfg(obj)
    elif type(obj).__module__ == "numpy": # Convert numpy types
        obj = jsonify_numpys(obj)
    elif callable(obj):
        obj = jsonify_function(obj)
    return obj

# This function converts a NumPy array to a Python list.
def jsonify_numpys(value):
    if isinstance(value, np.ndarray): #If value is a numpy array, convert it to a list
        new_value = value.tolist()
    else:
        if hasattr(value, "item"): #If value is a single item using a numpy type, convert it to a python type
            new_value = value.item()
        else:
            new_value = value #Otherwise, leave it as is
            #raise TypeError("Unexpected type: ", type(value)) # Commented to avoid breaking code
    return new_value

def jsonify_function(value):
    return value.__name__



#TODO:
# Put defaults 
# Delete some experiments
# Change a parameter's name