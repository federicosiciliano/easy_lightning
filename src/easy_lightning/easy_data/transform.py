import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler

# Function to merge specified variables in the data dictionary
def merge_splits(data, merge_keys={"x": ["train_x", "val_x", "test_x"], "y": ["train_y", "val_y", "test_y"]},
                 concat_function=np.concatenate, del_after_merge=True, **kwargs):
    """
    Merge specified variables in the data dictionary into a single variable.

    :param data: Dictionary containing data variables.
    :param merge_keys: Dictionary specifying the variables to merge.
    :param concat_function: Function to concatenate variables.
    :param del_after_merge: Whether to delete original variables after merging.
    :param kwargs: Additional keyword arguments.
    :return: Updated data dictionary with merged variables.
    """
    for merged_var, split_vars in merge_keys.items():
        app = []
        for key in split_vars:
            if key in data:
                app.append(data[key])
                if del_after_merge:
                    del data[key]
        data[merged_var] = concat_function(app, axis=0)


def split_data(data, split_keys={"x": ["train_x", "val_x", "test_x"], "y": ["train_y", "val_y", "test_y"]}, test_sizes=0.2, train_sizes=None, split_random_state=21094, split_shuffle=True, split_stratify=None, del_after_split=False, **kwargs):
    """
    Splits the input data into training, validation, and test sets based on the specified keys and parameters.

    :param data: A dictionary-like object containing the data to be split.
    :param split_keys: A dictionary specifying the keys for splitting data.
    :param test_sizes: The size of the test set(s) as a fraction or number of samples.
    :param train_sizes: The size of the training set(s) as a fraction or number of samples.
    :param split_random_state: Seed for random number generation during splitting.
    :param split_shuffle: Whether to shuffle the data before splitting.
    :param split_stratify: Optional parameter for stratified splitting.
    :param del_after_split: Whether to delete the original merged data after splitting.
    :param kwargs: Additional keyword arguments (not used in this function).

    :return: A dictionary containing the split data.
    """
    num_splits = len(split_keys[list(split_keys.keys())[0]]) - 1

    # Check and convert single values to lists for consistency
    if isinstance(test_sizes, int) or isinstance(test_sizes, float):
        test_sizes = [test_sizes] * num_splits
    if train_sizes is None or isinstance(train_sizes, int) or isinstance(train_sizes, float):
        train_sizes = [train_sizes] * num_splits
    if split_random_state is None or isinstance(split_random_state, int):
        split_random_state = [split_random_state] * num_splits
    if isinstance(split_shuffle, bool):
        split_shuffle = [split_shuffle] * num_splits
    if not isinstance(split_stratify, tuple) or not isinstance(split_stratify, list):
        split_stratify = [split_stratify] * num_splits

    # Create a list to accumulate variables
    accum_vars = []
    for merged_var, split_vars in split_keys.items():
        accum_vars.append(split_vars[0])
        data[accum_vars[-1]] = data[merged_var]

    # Iterate through merged variables and split
    for merged_vars, test_size, train_size, random_state, shuffle, stratify in zip(
            list(zip(*split_keys.values()))[1:], test_sizes, train_sizes, split_random_state, split_shuffle, split_stratify):
        if not all([x in data for x in merged_vars]):
            # Split only if the merged variables are not already present in the data
            app = train_test_split(*[data[accum_var] for accum_var in accum_vars],
                                   test_size=test_size, train_size=train_size,
                                   random_state=random_state, shuffle=shuffle, stratify=stratify)

            cont = 0
            for accum_key, new_key in zip(accum_vars, merged_vars):
                data[accum_key] = app[cont]
                data[new_key] = app[cont + 1]
                cont += 2
        else:
            print("Split already existing")

    if del_after_split:
        # Delete the merged variables from data if specified
        for merged_var in split_keys.keys():
            del data[merged_var]
    return data

# TODO: BUG--> It changes dtype from float32 to float64? Why?
def scale_data(data, scaling_method=None, scaling_params={}, scaling_keys = {"train_x": ["train_x", "val_x", "test_x"]}, scaling_fit_params={}, scaling_transform_params={}, **kwargs):
    """
    Scales the data using a specified scaling method and parameters.

    :param data: A dictionary-like object containing the data to be scaled.
    :param scaling_method: The scaling method to use (e.g., 'StandardScaler', 'MinMaxScaler', etc.), or None to skip scaling.
    :param scaling_params: Additional parameters for the scaling method.
    :param scaling_keys: The keys for the data to be scaled. Each key represent the fit_key, the values are the keys to be scaled.
    :param scaling_fit_params: Additional parameters for fitting the scaler.
    :param scaling_transform_params: List of dictionaries with additional parameters for transforming each scaled dataset.
    :param kwargs: Additional keyword arguments (not used in this function).

    :return: A tuple containing the scaled data dictionary and the scaler used.
    """

    # data_scaler will store the scaler objects
    data_scalers = {}

    # Check if a scaling method is specified
    if scaling_method is not None:
        print("Scaling data with method:", scaling_method)
        for fit_key, transform_keys in scaling_keys.items():
            # Create a scaler object using the specified method and parameters
            data_scalers[fit_key] = getattr(preprocessing, scaling_method)(**scaling_params)
            
            # Fit the scaler on the specified data key
            reshape_and_scale(data[fit_key], data_scalers[fit_key], fit=True, **scaling_fit_params)

            # Apply the scaler to each key in scaling_keys using specified parameters
            for transform_key in transform_keys:
                data[transform_key] = reshape_and_scale(data[transform_key], data_scalers[fit_key], **scaling_transform_params)

    return data, data_scalers

def reshape_and_scale(data, data_scaler, fit=False, **params):
    """
    Reshapes and scales the data using a specified scaler.

    :param data: A dictionary-like object containing the data to be reshaped and scaled.
    :param scaler: The scaler to use for scaling the data.

    :return: The reshaped and scaled data dictionary.
    """

    if len(data.shape) > 2:
        orig_shape = data.shape
        data = data.reshape(-1)[:, None]
    if fit:
        data_scaler.fit(data, **params)
    else:
        data = data_scaler.transform(data)
    if len(orig_shape) > 2:
        data = data.reshape(orig_shape)

    return data

# TODO: BUG--> It changes dtype from float32 to float64? Why?
def one_hot_encode_data(data, encode_keys={"y", "train_y", "val_y", "test_y"}, encode_fit_key="train_y", onehotencoder_params={"sparse_output": False}, **kwargs):
    """
    Performs one-hot encoding on categorical data in the specified keys.

    :param data: A dictionary-like object containing the data to be one-hot encoded.
    :param encode_keys: The keys to be one-hot encoded.
    :param encode_fit_key: The key to use for fitting the one-hot encoder.
    :param onehotencoder_params: Additional parameters for the OneHotEncoder.
    :param kwargs: Additional keyword arguments (not used in this function).

    :return: The updated data dictionary with one-hot encoding applied.
    """
    # Ensure that encode_fit_key is in data, or raise an error
    if encode_fit_key not in data:
        raise ValueError(f"'{encode_fit_key}' is not present in the data.")

    # Determine which keys to encode (excluding the fit key)
    encode_keys = encode_keys.intersection(data.keys()).difference({encode_fit_key})

    # Create an instance of the OneHotEncoder with specified parameters
    enc = OneHotEncoder(**onehotencoder_params)

    # Fit the encoder on the specified key
    data[encode_fit_key] = one_hot_encode_matrix(enc, data[encode_fit_key], "fit_transform")

    # Transform the other keys using the same encoder
    for key in encode_keys:
        data[key] = one_hot_encode_matrix(enc, data[key], "transform")

    return data


def one_hot_encode_matrix(enc, arr, method="transform"):
    """
    Applies one-hot encoding to a matrix using a given encoder.

    :param enc: A one-hot encoder object (e.g., sklearn.preprocessing.OneHotEncoder).
    :param arr: The input matrix to be one-hot encoded.
    :param method: The method to apply, either 'fit_transform' or 'transform'.
    
    :return: The one-hot encoded matrix.
    """
    # Store the original shape of the input array
    orig_shape = arr.shape
    
    # Flatten the array, apply the specified method, and reshape it
    new_arr = np.array(getattr(enc, method)(arr.flatten()[:, None]))  # Flatten to (N, 1) shape
    new_arr = new_arr.reshape(*orig_shape, np.prod(new_arr.shape) // np.prod(orig_shape))  # Reshape to (*original shape, one_hot_encode_dim)
    
    return new_arr
