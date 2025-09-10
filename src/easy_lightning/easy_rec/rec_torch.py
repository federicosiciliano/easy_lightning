import torch
import easy_lightning.easy_torch.collators
import easy_lightning.easy_torch.datasets
import pytorch_lightning as pl
import multiprocessing
from copy import deepcopy
from . import model

#TODO: pass this function inside easy_torch
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

        # Convert each value in the data dictionary to a PyTorch tensor
        # for key, value in self.data.items():
        #     if isinstance(value, torch.Tensor):
        #         self.data[key] = value.clone().detach()
        #     else:
        #         self.data[key] = torch.tensor(value)

    # Method to get an item from the dataset at a given index
    def __getitem__(self, index):
        """
            Retrieves a single item from the dataset at the given index.

            Args:
                index (int): Index of the item to retrieve.
            Returns:
                dict: Dictionary of the same keys as `data`, each containing the element at `index`.
        """

        return {key: value[index] for key, value in self.data.items()}

    # Method to get the length of the dataset
    def __len__(self):
        """
            Returns the total number of samples in the dataset.
            Returns:
                int: Length of the dataset, inferred from the first key in the data dictionary.
        """
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

#TODO: which parent class to use?
#TODO: move this into easy_torch
class SequentialCollator:
    """
        A collator forfor preparing sequential input-output pairs with configurable lookback, lookforward, and simultaneous steps.
        
        Args:
            sequential_keys (list): List of keys in the batch data that represent sequential features.
            lookback (int): Number of time steps to look back for input sequences.
            padding_value (int, optional): Value used for padding sequences. 
            left_pad (bool, optional): Whether to pad sequences on the left. 
            lookforward (int, optional): Number of time steps to skip between input and output. 
            simultaneous_lookforward (int, optional): Number of simultaneous future time steps to predict. 
            simultaneous_lookback (int, optional): Number of simultaneous past time steps to include in output. 
            out_seq_len (int or float, optional): Length of output sequences. If float, treated as proportion of input length. 
            keep_last (int, optional): Number of last time steps to keep in output. 
            drop_original (bool, optional): Whether to remove original sequential keys from output. 
    """
    def __init__(self,
                 sequential_keys,
                 lookback,
                 padding_value=0, 
                 left_pad=True, 
                 lookforward=1, 
                 simultaneous_lookforward=1,
                 simultaneous_lookback = 0,
                 out_seq_len=None,
                 keep_last = None,
                 drop_original=True):
        
        self.sequential_keys = sequential_keys
        self.padding_value = padding_value
        self.left_pad = left_pad
        self.lookback = lookback
        
        self.lookforward = lookforward
        self.simultaneous_lookforward = simultaneous_lookforward
        self.simultaneous_lookback = simultaneous_lookback
        self.out_seq_len = out_seq_len
        
        self.keep_last = keep_last
        if keep_last is None:
            self.keep_last = lookback

        self.drop_original = drop_original

        if self.left_pad:
            self.pad_x_function = self.reverse
            self.pad_out_func = self.flip
        else:
            self.pad_x_function = self.identity
            self.pad_out_func = self.identity

        self.needed_length = self.lookback + self.lookforward + self.simultaneous_lookforward +self.simultaneous_lookback
        
    #Functions needed because AttributeError: Can't pickle local object 'SequentialCollator.__init__.<locals>.<lambda>'
    def identity(self, x):
        """
            Returns input unchanged.

            Args:
                x: Input to return unchanged. 
            Returns:
                The input x unchanged.
        """
        return x
    
    def reverse(self, x):
        """
            Reverses the order of elements in a sequence.

            Args:
                x: Input sequence to reverse. 
            Returns:
                Reversed sequence.
        """
        return x[::-1]
    
    def flip(self, x):
        """
            Flips a PyTorch tensor along dimension 1.

            Args:
                x (torch.Tensor): Input tensor to flip.    
            Returns:
                torch.Tensor: Tensor flipped along dimension 1.
        """
        return x.flip(dims=[1])
    
    def extra_pad(self, x):
        """
            Adds extra padding to ensure tensor meets minimum length requirements.

            Args:
                x (torch.Tensor): Input tensor to potentially pad.
            Returns:
                torch.Tensor: Padded tensor if padding was needed, otherwise original tensor.
        """
        if self.needed_length <= x.shape[1]:
            return x
        else:
            return torch.cat([x, torch.zeros((x.shape[0], self.needed_length - x.shape[1]),dtype=x.dtype)],dim=1)
            #return torch.cat([x, self.padding_value*torch.ones((x.shape[0], self.needed_length - x.shape[1]),dtype=x.dtype)],dim=1)

    def __call__(self, batch):
        """
            Processes a batch of sequential data.

            Args:
                batch (list): List of dictionaries, each containing sequential data with keys
                            specified in sequential_keys.         
            Returns:
                dict: Processed batch with input-output pairs created from sequential data.
        """
        seq_lens = torch.tensor([len(x[self.sequential_keys[0]]) for x in batch])

        out = self.main_call(batch, seq_lens)
    
        return out
    
    def main_call(self, batch, seq_lens):
        """
            Main processing logic for handling the batch data.
            Args:
                batch (list): List of dictionaries containing the batch data.
                seq_lens (torch.Tensor): Tensor containing the length of each sequence in the batch.
            Returns:
                dict: Processed batch data with padded sequences and input-output pairs.
        """
        out = {}
        
        # Pad the sequences in the data using specified parameters
        for key in batch[0].keys():
            if key in self.sequential_keys:
                out[key] = self.pad_list_of_tensors([x[key] for x in batch])
            else:
                out[key] = torch.stack([torch.tensor(x[key]) for x in batch])
        
        # Pair input and output sequences based on specified parameters
        out = self.pair_input_output(out, seq_lens)
        return out

    # Method to pad a list of tensors and return the padded sequence as a tensor
    def pad_list_of_tensors(self, list_of_tensors):
        """
            Pads a list of tensors to create a uniform batch tensor.

            Args:
                list_of_tensors (list): List of tensors or sequences to be padded.
            Returns:
                torch.Tensor: Batched and padded tensor with shape (batch_size, padded_length).
        """
        padded = torch.nn.utils.rnn.pad_sequence([torch.tensor(self.pad_x_function(x)) for x in list_of_tensors], batch_first=True, padding_value=self.padding_value)
        
        padded = self.extra_pad(padded)

        #Also add padding for simultaneous_lookforward
        sim_lookf_pad = self.padding_value*torch.ones((len(padded),self.simultaneous_lookforward-1))
        lookf_pad = self.padding_value*torch.ones((len(padded),self.lookforward))
        padded = torch.concat([sim_lookf_pad,padded,lookf_pad],dim=1)

        padded = self.pad_out_func(padded)

        # Change type to type of first non-empy list in list of tensors
        for x in list_of_tensors:
            if len(x)>0: break
        padded = padded.type(getattr(torch,str(type(x[0]).__name__)))

        return padded
    
    # Method to pair input and output sequences based on specified parameters
    # Now based on left_padding; TODO: reverse array if opposite
    def pair_input_output(self, data, seq_lens):
        """
            Creates input-output pairs from sequential data based on temporal relationships.

            Args:
                data (dict): Dictionary containing padded sequential data and other features.
                seq_lens (torch.Tensor): Original sequence lengths before padding.
            Returns:
                dict: Updated data dictionary with 'in_{key}' and 'out_{key}' pairs for each
                    sequential key. Original sequential keys are removed if drop_original is True.  
            Raises:
                ValueError: If computed current_index contains negative values.
        """
        if self.out_seq_len is None:
            out_seq_len = seq_lens
        else:
            if isinstance(self.out_seq_len, float):
                out_seq_len = torch.ceil((self.out_seq_len*seq_lens)).int()
            else:
                out_seq_len = self.out_seq_len*torch.ones_like(seq_lens)

        # decide current point t;
        # input goes from t-lookback+1 to t;
        # output goes from t+lookforward to t+lookforward+simultaneous_lookforward
        #TODO: check NEW output goes from t+lookforward-simultaneous_lookforward to 
        output_poss_end_ids = seq_lens #-self.lookforward+1
        output_poss_start_ids = torch.maximum(output_poss_end_ids-out_seq_len,torch.zeros_like(seq_lens))

        input_poss_start_ids = output_poss_start_ids - self.lookforward
        input_poss_end_ids = output_poss_end_ids - self.lookforward

        true_starting_point = data[self.sequential_keys[0]].shape[1] - (seq_lens+(self.simultaneous_lookforward-1))
        input_poss_start_ids += true_starting_point
        input_poss_end_ids += true_starting_point

        input_poss_start_ids = torch.minimum(input_poss_start_ids+(self.lookback-1),input_poss_end_ids-1)
        input_poss_start_ids = torch.maximum(input_poss_start_ids,(self.lookback-1)*torch.ones_like(seq_lens))

        #.int() floors the number, so max_len can't be selected (good, cause is out of bounds)
        # Generate random indices for output sequences
        rand = torch.randint(2**63 - 1, size=(len(seq_lens),))
        current_index = (rand % (input_poss_end_ids - input_poss_start_ids) + input_poss_start_ids).int()
        
        if (current_index < 0).any():
            raise ValueError("Some current index is negative")

        # subtract lookback
        input_indices = current_index.unsqueeze(1) - (torch.arange(self.lookback).flip(dims=[0])).unsqueeze(0)

        # Compute input indices based on output indices
        output_indices = input_indices + self.lookforward

        # Add simultaneous_lookforward and simultaneous_lookback to output indices
        output_indices = output_indices.unsqueeze(2) + torch.arange(-self.simultaneous_lookback,self.simultaneous_lookforward).unsqueeze(0).unsqueeze(0)

        # Process each sequential key
        for key in self.sequential_keys:
            # Create input and output sequences based on calculated indices
            data[f"in_{key}"] = data[key][torch.arange(data[key].shape[0]).unsqueeze(-1),input_indices]
            data[f"out_{key}"] = data[key][torch.arange(data[key].shape[0]).unsqueeze(-1).unsqueeze(-1),output_indices]

            # Remove output values if index is before out_seq_len from the end
            # Option 1: keep same shape
            # self.data[f"out_{key}"][:, :-out_seq_len] = padding_value
            # Option 2: shorten array
            to_keep = -min(self.keep_last,out_seq_len.max())
            data[f"out_{key}"] = data[f"out_{key}"][:, to_keep:]
            # Shorten by number of samples reserved to this split, also removing simultaneous_lookforward

            # Optional: Squeeze out the last dimension if simultaneous_lookforward is 1
            # if simultaneous_lookforward == 1:
            #     self.data[f"out_{key}"] = self.data[f"out_{key}"].squeeze(-1)

            if self.drop_original:
                del data[key]
        
        return data
    
class SmartPaddingSequentialCollator(SequentialCollator):
    """
        A smart padding variant of SequentialCollator that dynamically adjusts lookback and the needed length based on batch content.
    
        Inherits all parameters from SequentialCollator.

        Args:
            needed_length_backup (int): Backup of the original needed_length value.
            lookback_backup (int): Backup of the original lookback value.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.needed_length_backup = self.needed_length
        self.lookback_backup = self.lookback

    def __call__(self, batch):
        """
            Processes a batch with smart padding that adapts lookback to available sequence length.
            
            Args:
                batch (list): List of dictionaries, each containing sequential data with keys specified in sequential_keys.          
            Returns:
                dict: Processed batch with input-output pairs created from sequential data,
                    using the optimally adjusted lookback window for this specific batch.
        """
        seq_lens = torch.tensor([len(x[self.sequential_keys[0]]) for x in batch])

        # Set the lookback and needed length to the minimum of the current batch
        self.lookback = min(max(seq_lens),self.lookback)
        self.needed_length = self.needed_length_backup - self.lookback_backup + self.lookback

        out = self.main_call(batch, seq_lens)
        
        # Reset the needed length and lookback to their original values
        self.needed_length, self.lookback = self.needed_length_backup, self.lookback_backup

        return out
    

class RecommendationSequentialCollator(SequentialCollator):
    """
        A specialized sequential collator for recommendation systems with negative sampling and relevance scoring.
        
        Args:
            num_items (int): Total number of items in the recommendation catalog.
            primary_key (str, optional): Key name for the main sequential feature (item IDs). 
            relevance (str or None, optional): Relevance scoring strategy. Options: None, "fixed", "linear", "exponential", or a data key name. 
            normalize_relevance (bool, optional): Whether to normalize relevance scores. 
            num_positives (int, optional): Number of positive samples per timestep. 
            num_negatives (int, optional): Number of negative samples per timestep. 
            mask_value (int, optional): Value used for masking input items. If None and mask_prob > 0, defaults to num_items + 1.
            mask_prob (float, optional): Probability of masking input items. 
            negatives_relevance (float, optional): Relevance score assigned to negative samples. 
            possible_negatives (str or callable, optional): Strategy for selecting candidate negatives. Options: "different", "all", "same", or custom function.
            negatives_distribution (str or torch.Tensor, optional): Distribution for sampling negatives. Options: "uniform", "dynamic", or probability tensor.              
            id_key (str, optional): Key name for user/session identifiers. 
    """
    
    def __init__(self,
                 num_items,
                 primary_key="sid",
                 relevance=None,
                 normalize_relevance=False,
                 num_positives = 1,
                 num_negatives = 1,
                 mask_value = None,
                 mask_prob = 0,
                 negatives_relevance = 0,
                 possible_negatives = "different",
                 negatives_distribution = "uniform",
                 id_key="uid",
                 *args,**kwargs): #out_seq_len and padding_value are set by super().__init__
        
        super().__init__(*args,**kwargs)

        self.num_items = num_items
        self.primary_key = primary_key
        self.in_key = f"in_{primary_key}"
        self.out_key = f"out_{primary_key}"
        self.id_key = id_key

        # Set the number of positives and negatives
        self.num_positives = num_positives
        self.num_negatives = num_negatives

        self.negatives_relevance = float(negatives_relevance)

        if possible_negatives == "different":
            self.possible_negatives_function = self.different_possible_negatives
        elif possible_negatives == "all":
            self.possible_negatives_function = self.all_possible_negatives
        elif possible_negatives == "same":
            self.possible_negatives_function = self.same_possible_negatives
        elif callable(possible_negatives):
            self.possible_negatives_function = possible_negatives
        else:
            raise NotImplementedError(f"Unsupported possible_negatives: {possible_negatives}")
        
        self.negatives_distribution = negatives_distribution
        if self.negatives_distribution == "uniform":
            self.sample_from_negative_distribution = self.uniform_negatives
        elif self.negatives_distribution == "unique":
            self.sample_from_negative_distribution = self.unique_negatives
        elif isinstance(negatives_distribution, torch.Tensor):
            self.sample_from_negative_distribution = self.distr_negatives
        elif callable(negatives_distribution):
            self.sample_from_negative_distribution = negatives_distribution
        else:
            raise NotImplementedError(f"Unsupported negatives_distribution: {self.negatives_distribution}")
            
        self.relevance = relevance
        self.normalize_relevance = normalize_relevance
        # Set a relevance function based on the specified relevance type and data shape
        if relevance in {None, "fixed", "linear", "exponential"}:
            # Generate a relevance tensor based on the specified type and data shape
            self.relevance_function = self.generate_relevance_from_type
        elif isinstance(relevance, str):
            # Use an existing output key as the relevance function
            self.relevance_function = self.get_relevance_from_data
        else:
            # Raise an error for unsupported relevance types
            raise NotImplementedError(f"Unsupported relevance: {relevance}")

        if mask_prob > 0 and mask_value is None:
            mask_value = num_items+1
        self.mask_value = mask_value
        self.mask_prob = mask_prob

    def __call__(self, batch):
        """
            Processes a batch for recommendation tasks with negative sampling and relevance scoring.

            Args:
                batch (list): List of dictionaries containing recommendation sequences with user/session data.    
            Returns:
                dict: Processed batch containing:
                    - Input and output sequences (in_{primary_key}, out_{primary_key})
                    - Relevance scores for all items (positive and negative)
                    - Other features from the original batch
                    - NaN relevance scores for padded positions
        """
        out = super().__call__(batch)

        out["relevance"] = self.relevance_function(out).type(torch.float) # Add relevance scores to out #TODO: check type float

        if self.num_negatives > 0: # Add negative samples to the out
            timesteps = out[self.out_key].shape[1]
            negatives = self.sample_negatives([x[self.primary_key] for x in batch], timesteps, [x[self.id_key] for x in batch])

        out[self.in_key] = self.mask_input(out[self.in_key])

        out_is_padding = torch.isclose(out[self.out_key], self.padding_value*torch.ones_like(out[self.out_key]))
        not_to_use = out_is_padding
        if self.mask_value is not None:
            relevant_in = out[self.in_key][:, -out[self.out_key].shape[1]:]
            input_is_not_masking = torch.logical_not(torch.isclose(relevant_in, self.mask_value*torch.ones_like(relevant_in)))
            not_to_use = torch.logical_or(not_to_use, input_is_not_masking.unsqueeze(-1))

        all_not_to_use = not_to_use.all(-1)

        if self.num_negatives > 0:
            negatives[all_not_to_use] = self.padding_value # Pad negative if out (i.e. positives) is padding #.all(-1) for out because out can have multiple values, i.e. simultaneous_lookforward
            # Masked tensor are a prototype in torch, but could solve many issues with padding and not wanting to compute losses or metrics on padding
            #negatives = torch.masked.masked_tensor(negatives, out_is_padding.unsqueeze(-1))

            out[self.out_key] = torch.cat([out[self.out_key], negatives], dim=-1) #concatenate negatives to out[out_key]
        
            out_is_padding = torch.isclose(out[self.out_key], self.padding_value*torch.ones_like(out[self.out_key]))
            not_to_use = torch.logical_or(out_is_padding, not_to_use)

            negatives_relevance_tensor = self.negatives_relevance*torch.ones_like(negatives)
            #negatives_relevance_tensor[negative_is_positive.any(-2)] = (out["relevance"][:,:,:self.num_positives,None]*negative_is_positive)[negative_is_positive].type(out["relevance"].dtype) # Do not use negative if is positive, i.e. positives cannot be negatives
            out["relevance"] = torch.cat([out["relevance"], negatives_relevance_tensor], dim=-1) #concatenate negatives relevance to relevance

        out["relevance"][:,:,:not_to_use.shape[-1]][not_to_use] = float("nan") # Nan relevance if out is padding or input is not masked (if applicable)

        out["relevance"][:,:,not_to_use.shape[-1]:][all_not_to_use] = float("nan") # Nan relevance if all out is padding
        return out
    
    def different_possible_negatives(self, orig_seq, *args):
        """
            Gets candidate negative items that are different from items in the original sequence.
            
            Args:
                orig_seq: Original sequence of item IDs.
            Returns:
                torch.Tensor: Tensor of item IDs that are not present in the original sequence.
        """
        return torch.tensor(list(set(range(1, self.num_items + 1)).difference(orig_seq)))

    def all_possible_negatives(self, *args):
        """
            Gets all possible items as candidate negatives.

            Returns:
                torch.Tensor: Tensor containing all item IDs from 1 to num_items.
        """
        return torch.arange(1, self.num_items + 1)

    def same_possible_negatives(self, orig_seq, *args): #sample only from orig_seq
        """
            Gets candidate negatives only from items present in the original sequence.
            
            Args:
                orig_seq: Original sequence of item IDs. 
            Returns:
                torch.Tensor: Tensor of unique item IDs from the original sequence.
        """
        return torch.tensor(list(set(orig_seq)))

    def uniform_negatives(self, possible_negatives, n, *args):
        """
            Samples negatives uniformly from candidate negatives.
            
            Args:
                possible_negatives (torch.Tensor): Candidate negative item IDs.
                n (int): Number of negatives to sample.
            Returns:
                torch.Tensor: Uniformly sampled negative item IDs.
        """
        return possible_negatives[torch.randint(0, len(possible_negatives), (n,))]
    
    def unique_negatives(self, possible_negatives, n, *args):
        """
            Samples unique negatives from candidate negatives, padding if necessary.
            
            Args:
                possible_negatives (torch.Tensor): Candidate negative item IDs.
                n (int): Number of negatives to sample.
            Returns:
                torch.Tensor: Unique negative item IDs sampled, padded if fewer than n candidates.
        """
        negatives = possible_negatives[torch.randperm(len(possible_negatives))[:n]]
        if len(possible_negatives) <= n:
            negatives = torch.cat([negatives, self.padding_value*torch.ones(n-len(negatives),dtype=negatives.dtype)],dim=0)
        return negatives
    
    def distr_negatives(self, possible_negatives, n, *args):
        """
            Samples negatives according to a predefined probability distribution.
            
            Args:
                possible_negatives (torch.Tensor): Candidate negative item IDs.
                n (int): Number of negatives to sample.
            Returns:
                torch.Tensor: Negative item IDs sampled according to the distribution.
        """
        distr = self.negatives_distribution[possible_negatives]
        repl = True if len(possible_negatives) < n else True
        return possible_negatives[torch.multinomial(distr, n, replacement=repl)]

    # Method to sample negative items for a given set of indices
    def sample_negatives(self, original_sequences, t=1, id_keys=[]):
        """
            Samples negative items for each sequence in the batch from candidates using the specified distribution.
            
            Args:
                original_sequences (list): List of original item sequences for each user/session.
                t (int, optional): Number of timesteps. Defaults to 1.
                id_keys (list, optional): List of user/session identifiers. Defaults to empty list. 
            Returns:
                torch.Tensor: Tensor of shape (batch_size, t, max(num_negatives, 1)) containing negative item IDs for each sequence and timestep.
        """
        id_keys = id_keys if len(id_keys)>0 else [None]*len(original_sequences)
        if self.num_negatives == 0:
            return torch.zeros(len(original_sequences), t, self.num_negatives, dtype=torch.long)
        negatives = torch.zeros(len(original_sequences), self.num_negatives*t, dtype=torch.long)
        for i, (orig_seq,id_key) in enumerate(zip(original_sequences,id_keys)): #TODO: parallelize this
            # Get possible negative items that are not in the original sequence
            possible_negatives = self.possible_negatives_function(orig_seq, id_key)
            # Randomly sample num_negatives negative items
            negatives[i] = self.sample_from_negative_distribution(possible_negatives, self.num_negatives*t, id_key)
            #negatives[i] = possible_negatives[torch.randperm(len(possible_negatives))[:self.num_negatives*t]]
        negatives = negatives.reshape(len(original_sequences), t, max(self.num_negatives,1))
        return negatives
    
    def mask_input(self, input):
        """
            Applies random mask to items of input sequencesn for evaluation purposes.
            
            Args:
                input (torch.Tensor): Input tensor of item IDs to potentially mask.
            Returns:
                torch.Tensor: Input tensor with some items replaced by mask_value.
        """
        if self.mask_prob == 0: return input
        mask = torch.rand(input.shape) < self.mask_prob
        if self.out_seq_len is not None:
            mask[:,-self.out_seq_len:] = True #Always mask last items (for val / test purposes)
            mask[:,:-self.out_seq_len] = False #Never mask first items
        input_is_padding = torch.isclose(input, self.padding_value*torch.ones_like(input))
        mask[input_is_padding] = False
        input[mask] = self.mask_value
        return input

    # Function to generate a relevance tensor based on the specified relevance type and shape
    def generate_relevance_from_type(self, complete_data):
        """
            Generates relevance scores based on the specified relevance type:
                - None/"fixed": All items have relevance score of 1.0
                - "linear": Linear decay from 1.0 to 0.0 over sequence positions
                - "exponential": Exponential decay over sequence positions

            Args:
                complete_data (dict): Complete batch data containing output sequences.
            Returns:
                torch.Tensor: Relevance scores tensor matching the shape of output sequences, with values between 0 and 1, optionally normalized.
        """
        relevance_type = self.relevance
        data = complete_data[self.out_key]
        shape = data.shape
        if relevance_type is None or relevance_type == "fixed":
            # Generate a tensor of ones if the relevance type is None or fixed
            app = torch.ones(shape[-1])
        else:
            # Generate a tensor with values from 1 to 0 with equal spacing
            app = torch.linspace(0, 1, shape[-1])[::-1]
            # Adjust the tensor based on the relevance type
            if relevance_type == "linear":
                pass
            elif relevance_type == "exponential":
                app = torch.exp(app)

        # Normalize the tensor
        if self.normalize_relevance:
            app /= torch.sum(app)

        # Repeat the tensor to match the shape
        app = app.repeat(*shape[:-1], 1)

        return app
    
    def get_relevance_from_data(self, complete_data):
        """
            Extracts relevance scores from an existing data field.
            
            Args:
                complete_data (dict): Complete batch data containing the relevance field.
            Returns:
                torch.Tensor: Relevance scores extracted from the specified data key.
        """
        return complete_data[self.relevance]


class RecommendationSmartPaddingSequentialCollator(
        RecommendationSequentialCollator, SmartPaddingSequentialCollator
    ):
    pass
    
    
def prepare_rec_datasets(data,
                         split_keys={"train": ["sid", "timestamp", "rating", "uid"],
                                     "val": ["sid", "timestamp", "rating", "uid"],
                                     "test": ["sid", "timestamp", "rating", "uid"]},
                         **dataset_params
                         ):
    """
        Prepare recommendation datasets for training and evaluation.

        Args:
            data (dict): Input dictionary containing data.
            split_keys (dict): Dictionary specifying keys for different splits.
            **dataset_params: Additional parameters for dataset preparation.
        Returns:
            dict: Dictionary containing prepared recommendation datasets for each split.
    """

    datasets = {}
    for split_name, data_keys in split_keys.items():
        split_dataset_params = deepcopy(dataset_params)
        # Select specific parameters for this split
        for key, value in split_dataset_params.items():
            if isinstance(value, dict):
                if split_name in value.keys():
                    split_dataset_params[key] = value[split_name]
        
        # Get data
        data_to_use = {}
        for key in data_keys:
            if key in data:
                data_to_use[key] = data[key]
            else: #If key is not in data, try to get it using split_name
                data_to_use[key] = data[f"{split_name}_{key}"]

        # Create the Dataset
        datasets[split_name] = easy_lightning.easy_torch.datasets.DictDataset(data_to_use, **split_dataset_params)

    return datasets

def prepare_rec_collators(split_keys = ["train", "val", "test"],
                         collator_class = RecommendationSequentialCollator,
                         **collator_params):
    """
        Prepare recommendation data collators for training and evaluation.

        Args:
            data (dict): Input dictionary containing data.
            split_keys (list): List of split keys for data collators.
            original_seq_key (str): Key for original sequences in the data.
            **collator_params: Additional parameters for data collator preparation.
        Returns:
            dict: Dictionary containing prepared recommendation data collators for each split.
    """

    # Default collator parameters
    default_collator_params = {}

    # Combine default and custom collator parameters
    collator_params = dict(list(default_collator_params.items()) + list(collator_params.items()))
    
    collators = {}
    for split_name in split_keys:
        split_collator_params = deepcopy(collator_params)
        # Select specific parameters for this split
        for key, value in split_collator_params.items():
            if isinstance(value, dict):
                if split_name in value.keys():
                    split_collator_params[key] = value[split_name]
        split_collator_params = change_num_negative_if_float(split_collator_params)

        # orig_seq_id = original_seq_id if original_seq_id in data else f"{split_name}_{original_seq_id}"
        # orig_seq_key = original_seq_key if original_seq_key in data else f"{split_name}_{original_seq_key}"
        # original_seq = {k:v for k,v in zip(data[orig_seq_id],data[orig_seq_key])}

        # Create the DataCollator
        collators[split_name] = collator_class(**split_collator_params)

    return collators

def change_num_negative_if_float(collator_params):
    """
        Converts the 'num_negatives' parameter from a float to an integer count if necessary.

        Args:
            collator_params (dict): Dictionary of parameters for a data collator. Must contain `"num_items"` if `"num_negatives"` is a float.
        Returns:
            dict: The modified `collator_params` dictionary with `"num_negatives"` as an integer if it was originally a float.
        Raises:
            KeyError: If `"num_negatives"` is a float and `"num_items"` is not provided in `collator_params`.
    """
    if "num_negatives" in collator_params:
        if isinstance(collator_params["num_negatives"], float):
            collator_params["num_negatives"] = int(collator_params["num_negatives"]*collator_params["num_items"])
        # elif isinstance(collator_params["num_negatives"], dict):
        #     for key in collator_params["num_negatives"]:
        #         if isinstance(collator_params["num_negatives"][key], float):
        #             collator_params["num_negatives"][key] = int(collator_params["num_negatives"][key]*total_items)
    return collator_params

def prepare_rec_data_loaders(datasets,
                             split_keys = ["train", "val", "test"],
                             original_seq_key="sid",
                             **loader_params):
    """
        Prepare recommendation data loaders for training and evaluation.

        Args:
            datasets (dict): Dictionary containing prepared recommendation datasets.
            data (dict): Input dictionary containing data.
            split_keys (list): List of split keys for data loaders.
            original_seq_key (str): Key for original sequences in the data.
            **loader_params: Additional parameters for data loader preparation.
        Returns:
            dict: Dictionary containing prepared recommendation data loaders for each split.
    """                         
    # TODO: dict instead of list
    # I don't remember what I meant by this comment
    
    # Default loader parameters
    default_loader_params = {
        "num_workers": multiprocessing.cpu_count(),
        "pin_memory": True,
        "persistent_workers": True,
        "drop_last": {"train": False, "val": False, "test": False}, #TODO: check specifics about drop last: losing data?
        "shuffle": {"train": True, "val": False, "test": False},
    }
    # Combine default and custom loader parameters
    loader_params = dict(list(default_loader_params.items()) + list(loader_params.items()))
    
    loaders = {}
    for split_name in split_keys:
        split_loader_params = deepcopy(loader_params)
        # Select specific parameters for this split
        for key, value in split_loader_params.items():
            if isinstance(value, dict):
                if split_name in value.keys():
                    split_loader_params[key] = value[split_name]
        
        # Create the DataLoader
        loaders[split_name] = torch.utils.data.DataLoader(datasets[split_name], **split_loader_params)

    return loaders

def create_rec_model(name, seed=42, additional_module=None, **model_params):
    """
        Create a recommendation model.
        
        Args:
            name (str): Name of the recommendation model.
            seed (int): Random seed for weight initialization.
            **model_params: Additional parameters for model creation.
        Returns:
            torch.nn.Module: Instance of the recommendation model.
    """
    # Set a random seed for weight initialization
    pl.seed_everything(seed, verbose=False)
    # Get the model from the model module
    if hasattr(additional_module, name):
        model_module = additional_module
    elif hasattr(model, name):
        model_module = model
    else:
        raise NotImplementedError(f"The model {name} is not found in model submodule or additional module")

    return getattr(model_module, name)(**model_params)
