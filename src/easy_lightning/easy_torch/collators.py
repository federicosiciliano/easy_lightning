import torch
#TODO: which parent class to use?
class SequentialCollator:
    """
        A collator for preparing sequential input-output pairs with configurable lookback, lookforward, and simultaneous steps.
        
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
        return x
    
    def reverse(self, x):
        return x[::-1]
    
    def flip(self, x):
        return x.flip(dims=[1])
    
    def extra_pad(self, x):
        if self.needed_length <= x.shape[1]:
            return x
        else:
            return torch.cat([x, torch.zeros((x.shape[0], self.needed_length - x.shape[1]),dtype=x.dtype)],dim=1)
            #return torch.cat([x, self.padding_value*torch.ones((x.shape[0], self.needed_length - x.shape[1]),dtype=x.dtype)],dim=1)

    def __call__(self, batch):
        seq_lens = torch.tensor([len(x[self.sequential_keys[0]]) for x in batch])

        out = self.main_call(batch, seq_lens)
    
        return out
    
    def main_call(self, batch, seq_lens):
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
    # Now based on left_padding; TODO?: reverse array if opposite
    def pair_input_output(self, data, seq_lens):
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