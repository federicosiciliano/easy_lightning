import numpy as np

def separate_columns(data, separate_keys={"x": ["y"]}, column_ids={"y": [-1]}, del_after_separate=False, **kwargs):
    """
    Separates specified columns in a NumPy data array into separate variables.

    Args:
        data (dict): A dictionary containing data arrays.
        separate_keys (dict): A dictionary mapping merged variables to separate variables.
        column_ids (dict): A dictionary specifying the columns to separate for each merged variable.
        del_after_separate (bool): If True, delete the merged variable after separation.

    Returns:
        dict: A dictionary containing the separated data arrays.
    """
    # Iterate over merged variables and their corresponding separate variables
    for merged_var, separate_vars in separate_keys.items():
        remaining_cols = list(range(data[merged_var].shape[1]))
        # Iterate over keys and column indices for separation
        for key in separate_vars:
            selected_cols = column_ids[key]
            # Separate the specified column and store it in the separate variable
            data[key] = data[merged_var][:, selected_cols]
            remaining_cols = [i for i in remaining_cols if i not in selected_cols]
        
        # Separate the remaining columns and store them in the last separate variable
        data[merged_var] = data[merged_var][:, remaining_cols]
        
        # Optionally delete the merged variable after separation
        if del_after_separate:
            del data[merged_var]
    
    return data

def separate_rows(data, separate_keys={"x": ["header_x"]}, row_ids={"header_x": [0]}, del_after_separate=False, **kwargs):
    """
    Separates specified rows in a NumPy data array into separate variables.

    Args:
        data (dict): A dictionary containing data arrays.
        separate_keys (dict): A dictionary mapping merged variables to separate variables.
        row_ids (dict): A dictionary specifying the rows to separate for each merged variable.
        del_after_separate (bool): If True, delete the merged variable after separation.

    Returns:
        dict: A dictionary containing the separated data arrays.
    """
    # Iterate over merged variables and their corresponding separate variables
    for merged_var, separate_vars in separate_keys.items():
        remaining_rows = list(range(data[merged_var].shape[0]))
        # Iterate over keys and row indices for separation
        for key in separate_vars:
            selected_rows = row_ids[key]
            # Separate the specified row and store it in the separate variable
            data[key] = data[merged_var][selected_rows]
            remaining_rows = [i for i in remaining_rows if i not in selected_rows]
        
        # Separate the remaining rows and store them in the last separate variable
        data[merged_var] = data[merged_var][remaining_rows]
        
        # Optionally delete the merged variable after separation
        if del_after_separate:
            del data[merged_var]
    
    return data

def separate_rows_and_columns(data, row_separate_keys={"x": ["header_x"]}, row_ids={"header_x": [0]},
                              column_separate_keys={"x": ["y"]}, column_ids={"y": [-1]},
                              del_after_separate=False, **kwargs):
    """
    Separates specified rows and columns in a NumPy data array into separate variables.

    Args:
        data (dict): A dictionary containing data arrays.
        row_separate_keys (dict): A dictionary mapping merged variables to separate variables for row separation.
        column_separate_keys (dict): A dictionary mapping merged variables to separate variables for column separation.
        row_ids (dict): A dictionary specifying the rows to separate for each merged variable.
        column_ids (dict): A dictionary specifying the columns to separate for each merged variable.
        del_after_separate (bool): If True, delete the merged variable after separation.

    Returns:
        dict: A dictionary containing the separated data arrays.
    """
    # Separate rows
    data = separate_rows(data, row_separate_keys, row_ids, del_after_separate)
    
    # Separate columns
    data = separate_columns(data, column_separate_keys, column_ids, del_after_separate)
    
    return data

#TODO: we could merge the above functions

def convert_types(data, type_dict={"x": np.float32, "y": np.float32}, **kwargs):
    """
    Converts data types for specified variables.

    Args:
        data (dict): A dictionary containing data arrays.
        type_dict (dict): A dictionary mapping variables to data types.

    Returns:
        dict: A dictionary containing the converted data arrays.
    """
    for key, dtype in type_dict.items():
        data[key] = data[key].astype(dtype)
    return data

def transpose(data, swap_dict={"x": (0,1)}, **kwargs):
    """
    Swaps axes for specified variables.

    Args:
        data (dict): A dictionary containing data arrays.
        swap_dict (dict): A dictionary mapping variables to axis swaps.

    Returns:
        dict: A dictionary containing the swapped data arrays.
    """
    for key, axes in swap_dict.items():
        data[key] = np.transpose(data[key], axes)
    return data

def sort_by_column(data, column_dict={"x": 0}, **kwargs):
    """
    Sorts data by specified columns.

    Args:
        data (dict): A dictionary containing data arrays.
        column_dict (dict): A dictionary mapping variables to columns.

    Returns:
        dict: A dictionary containing the sorted data arrays.
    """
    for key, column in column_dict.items():
        data[key] = data[key][np.argsort(data[key][:, column])]
    return data

def min_max_scale(data, scale_dict={"x": (0,255)}, **kwargs):
    """
    Scales specified variables to a specified range.

    Args:
        data (dict): A dictionary containing data arrays.
        scale_dict (dict): A dictionary mapping variables to scale ranges.

    Returns:
        dict: A dictionary containing the scaled data arrays.
    """
    for key, scale_range in scale_dict.items():
        data[key] = (data[key] - scale_range[0]) / (scale_range[1] - scale_range[0])
    return data

def mean_std_scale(data, scale_dict={"x": [0,1]}, **kwargs):
    """
    Scales specified variables to a specified range.

    Args:
        data (dict): A dictionary containing data arrays.
        scale_dict (dict): A dictionary mapping variables to scale ranges.

    Returns:
        dict: A dictionary containing the scaled data arrays.
    """
    for key, scale_range in scale_dict.items():
        while(len(scale_range[0].shape) < len(data[key].shape)):
            scale_range[0] = np.expand_dims(scale_range[0], axis=-1)
        while(len(scale_range[1].shape) < len(data[key].shape)):
            scale_range[1] = np.expand_dims(scale_range[1], axis=-1)
        data[key] = (data[key] - scale_range[0]) / scale_range[1]
    return data

def merge_data(data_list, ext):
    """
    Merge data from multiple files into a single dictionary.

    :param data_list: List of dictionaries containing data to be merged.
    :return: Merged data as a single dictionary.
    """
    out = {}
    for data in data_list:
        for key, value in data.items():
            if key not in out:
                out[key] = []
            out[key].append(value)

    for key, value in out.items():
        if ext in ["csv","npy"]:
            out[key] = np.stack(value)
        else:
            raise NotImplementedError("MERGING NOT IMPLEMENTED FOR", ext)
    return out
