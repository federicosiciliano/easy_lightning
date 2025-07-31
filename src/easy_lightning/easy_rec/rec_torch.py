import torch
import easy_torch.collators
import easy_torch.datasets
import pytorch_lightning as pl
import multiprocessing
from copy import deepcopy
from . import model
import easy_torch
    
class SmartPaddingSequentialCollator(easy_torch.collators.SequentialCollator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.needed_length_backup = self.needed_length
        self.lookback_backup = self.lookback

    def __call__(self, batch):
        seq_lens = torch.tensor([len(x[self.sequential_keys[0]]) for x in batch])

        # Set the lookback and needed length to the minimum of the current batch
        self.lookback = min(max(seq_lens),self.lookback)
        self.needed_length = self.needed_length_backup - self.lookback_backup + self.lookback

        out = self.main_call(batch, seq_lens)
        
        # Reset the needed length and lookback to their original values
        self.needed_length, self.lookback = self.needed_length_backup, self.lookback_backup

        return out
    
class RecommendationSequentialCollator(easy_torch.collators.SequentialCollator):
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
        
        if negatives_distribution not in {"uniform",'dynamic'} and not isinstance(negatives_distribution, torch.Tensor): #modifica_g : aggiunto il caso di campionamento dinamico dei negativi  durante il training
            raise NotImplementedError(f"Unsupported negatives_distribution: {negatives_distribution}")
        self.negatives_distribution = negatives_distribution
        if self.negatives_distribution == "uniform":
            self.sample_from_negative_distribution = self.uniform_negatives
        elif isinstance(negatives_distribution, torch.Tensor):
            self.sample_from_negative_distribution = self.distr_negatives
        elif callable(negatives_distribution):
            self.sample_from_negative_distribution = negatives_distribution
        else:
            raise NotImplementedError(f"Unsupported negatives_distribution: {negatives_distribution}")
            
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
        
            negatives_relevance_tensor = self.negatives_relevance*torch.ones_like(negatives)
            #negatives_relevance_tensor[negative_is_positive.any(-2)] = (out["relevance"][:,:,:self.num_positives,None]*negative_is_positive)[negative_is_positive].type(out["relevance"].dtype) # Do not use negative if is positive, i.e. positives cannot be negatives
            out["relevance"] = torch.cat([out["relevance"], negatives_relevance_tensor], dim=-1) #concatenate negatives relevance to relevance

        out["relevance"][:,:,:not_to_use.shape[-1]][not_to_use] = float("nan") # Nan relevance if out is padding or input is not masked (if applicable)

        out["relevance"][:,:,not_to_use.shape[-1]:][all_not_to_use] = float("nan") # Nan relevance if all out is padding
        return out
    
    def different_possible_negatives(self, orig_seq, *args):
        return torch.tensor(list(set(range(1, self.num_items + 1)).difference(orig_seq)))

    def all_possible_negatives(self, *args):
        return torch.arange(1, self.num_items + 1)

    def same_possible_negatives(self, orig_seq, *args): #sample only from orig_seq
        return torch.tensor(list(set(orig_seq)))

    def uniform_negatives(self, possible_negatives, n, *args):
        return possible_negatives[torch.randint(0, len(possible_negatives), (n,))]
    
    def distr_negatives(self, possible_negatives, n, *args):
        distr = self.negatives_distribution[possible_negatives]
        repl = True if len(possible_negatives) < n else True
        return possible_negatives[torch.multinomial(distr, n, replacement=repl)]

    # Method to sample negative items for a given set of indices
    def sample_negatives(self, original_sequences, t=1, id_keys=[]):
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

        # Create the DataLoader
        datasets[split_name] = easy_torch.datasets.DictDataset(data_to_use, **split_dataset_params)

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
    pl.seed_everything(seed)
    # Get the model from the model module
    if hasattr(additional_module, name):
        model_module = additional_module
    elif hasattr(model, name):
        model_module = model
    else:
        raise NotImplementedError(f"The model {name} is not found in model submodule or additional module")

    return getattr(model_module, name)(**model_params)
