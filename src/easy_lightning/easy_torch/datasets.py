import torch

# Define a custom PyTorch Dataset class named DictDataset
class DictDataset(torch.utils.data.Dataset):
    """
        Custom PyTorch Dataset class that takes a dictionary as input and returns items based on keys.

        Args:
            data (dict): Input dictionary containing data.
        Returns:
            dict: A dictionary where each key corresponds to a tensor item.
    """
    # Constructor to initialize the dataset with input data
    def __init__(self, data):
        self.data = data

    # Method to get an item from the dataset at a given index
    def __getitem__(self, index):
        return {key: value[index] for key, value in self.data.items()}

    # Method to get the length of the dataset
    def __len__(self):
        # Assumes that all values in the data dictionary have the same length
        return len(self.data[list(self.data.keys())[0]])
    

# class SequentialDataset(torch.utils.data.Dataset):
#     def __init__(self, data):
#         self.data = data
#         super().__init__()

#     def __getitem__(self, index):
#         batch = super().__getitem__(index)
#         for key,value in batch:
#             print(key,value)
#         print(sanaskjnas)

# # Method to pair input and output sequences based on specified parameters
# def pair_input_output(self, sequential_keys, padding_value, lookback, stride, lookforward, simultaneous_lookforward, out_seq_len, keep_last, drop_original=True):
#     key_to_use = sequential_keys[0]
#     max_len = self.data[key_to_use].shape[1]
#     if out_seq_len is None: out_seq_len = max_len

#     # Calculate input and output indices based on lookback, stride, and lookforward
#     # input_indices = torch.stack([torch.arange(a-lookback,a) for a in range(max_len-lookforward, lookback-1, -stride)][::-1])
#     input_indices = torch.stack([torch.arange(a-lookback,a) for a in range(max_len-lookforward-simultaneous_lookforward+1, max(lookback-1, max_len-lookforward-simultaneous_lookforward+1-out_seq_len), -stride)][::-1])
#     # output_indices = torch.stack([torch.arange(a-lookback,a) for a in range(max_len, lookback-1+lookforward, -stride)][::-1])
#     output_indices = torch.stack([torch.stack([torch.arange(b-simultaneous_lookforward+1,b+1) for b in torch.arange(a-lookback,a)]) for a in range(max_len, max(lookback-1+lookforward+simultaneous_lookforward-1,max_len-out_seq_len), -stride)][::-1])
    
#     # Get non-sequential keys in the data dictionary
#     non_sequential_keys = [key for key in self.data.keys() if key not in sequential_keys]

#     # Process each sequential key
#     for key in sequential_keys:
#         # Create input and output sequences based on calculated indices
#         self.data[f"in_{key}"] = self.data[key][:,input_indices]
#         self.data[f"out_{key}"] = self.data[key][:,output_indices]

#         # Remove output values where input is padding
#         input_is_padding = torch.isclose(self.data[f"in_{key}"], padding_value*torch.ones_like(self.data[f"in_{key}"]))
#         self.data[f"out_{key}"][input_is_padding] = padding_value

#         # Remove rows where all input or all output is padding
#         to_keep = torch.logical_and(
#             torch.logical_not(input_is_padding.all(-1)),
#             torch.logical_not(torch.isclose(self.data[f"out_{key}"], padding_value*torch.ones_like(self.data[f"out_{key}"])).all(-1).all(-1)))

#         self.data[f"in_{key}"] = self.data[f"in_{key}"][to_keep]
#         self.data[f"out_{key}"] = self.data[f"out_{key}"][to_keep]

#         # Remove output values if index is before out_seq_len from the end
#         # Option 1: keep same shape
#         # self.data[f"out_{key}"][:, :-out_seq_len] = padding_value
#         # Option 2: shorten array
#         self.data[f"out_{key}"] = self.data[f"out_{key}"][:, max(-keep_last,-out_seq_len+self.data[f"out_{key}"].shape[-1]-1):]
#         # Shorten by number of samples reserved to this split, also removing simultaneous_lookforward

#         # Optional: Squeeze out the last dimension if simultaneous_lookforward is 1
#         # if simultaneous_lookforward == 1:
#         #     self.data[f"out_{key}"] = self.data[f"out_{key}"].squeeze(-1)

#         # Optionally, drop the original key from the data dictionary
#         if drop_original:
#             del self.data[key]

#     # Repeat the indices of non-dropped rows for non-sequential keys
#     orig_rows_repeat = torch.where(to_keep)[0]

#     # Process each non-sequential key
#     for key in non_sequential_keys:
#         self.data[key] = self.data[key][orig_rows_repeat]