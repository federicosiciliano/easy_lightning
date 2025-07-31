# Import necessary libraries
import pandas as pd  # Import Pandas library for data manipulation
import os  # Import the os library for working with the file system
import torch  # Import the PyTorch library for deep learning
import pytorch_lightning as pl  # Import PyTorch Lightning for training and logging
from .model import BaseNN  # Import the BaseNN class from the model module


def create_model(main_module, seed=42, **kwargs):
    """
    Create a PyTorch Lightning model.

    Args:
        main_module (nn.Module): The main module of the model.
        seed (int, optional): Random seed for reproducibility (default: 42).
        **kwargs: Additional keyword arguments to pass to the BaseNN constructor.

    Returns:
        BaseNN: A PyTorch Lightning model wrapping the main_module.
    """
    pl.seed_everything(seed, verbose=False) 
    # Create the model using the BaseNN class
    model = BaseNN(main_module, **kwargs)
    return model


def train_model(trainer, model, loaders, train_key="train", val_key="val", seed=42, tracker=None, profiler=None):
    """
    Trains a PyTorch Lightning model.

    Args:
        trainer (pl.Trainer): The PyTorch Lightning Trainer instance used to fit the model.
        model (pl.LightningModule): The model to be trained.
        loaders (Dict[str, DataLoader]): Dictionary containing DataLoaders.
        train_key (str): Key to select the training DataLoader from `loaders` (default: "train").
        val_key (Union[str, list[str], None]): Key or list of keys to select validation DataLoaders,
            or None to skip validation (default: "val").
        seed (int): Random seed for deterministic training (default: 42).
        tracker (Optional[object]): Optional tracker with `start()` and `stop()` methods.
        profiler (Optional[object]): Optional profiler with `start_profile()`, `stop_profile()`,
            and `print_model_profile(output_file)` methods.

    Returns:
        None
    """
    # Set a random seed for deterministic training
    pl.seed_everything(seed, verbose=False)

    # Check if validation data loaders are specified and handle them accordingly
    # (single validation DataLoader if `val_key` is a string, or multiple if `val_key` is a list)
    if val_key is not None:
        if isinstance(val_key, str):
            val_dataloaders = loaders[val_key]
        elif isinstance(val_key, list):
            val_dataloaders = {key: loaders[key] for key in val_key}
        else:
            raise NotImplementedError
    else:
        val_dataloaders = None
    
    # Start the tracker and profiler if they are provided
    if tracker is not None: tracker.start()
    if profiler is not None: profiler.start_profile()
    
    # Train the model
    trainer.fit(model, loaders[train_key], val_dataloaders)
    
    # Stop the tracker and profiler if they are provided
    if tracker is not None:
        tracker.stop()
    if profiler is not None:
        profiler.print_model_profile(output_file = f"{profiler.output_dir}/train_flops.txt")
        profiler.stop_profile()


def validate_model(trainer, model, loaders, loaders_key="val", seed=42):
    """
    Validates a PyTorch Lightning model.

    Args:
        trainer (pl.Trainer): The PyTorch Lightning Trainer instance used to run validation.
        model (pl.LightningModule): The trained model to be validated.
        loaders (Dict[str, DataLoader]): Dictionary of DataLoaders, keyed by names (e.g., 'train', 'val').
        loaders_key (str): Key used to select the validation DataLoader from `loaders` (default: "val").
        seed (int): Random seed for reproducibility during validation (default: 42).

    Returns:
        None

    """
    pl.seed_everything(seed, workers=True, verbose=False)

    # Validate the model using the trainer
    trainer.validate(model, loaders[loaders_key])


def test_model(trainer, model, loaders, test_key="test", tracker=None, profiler=None, seed=42):
    """
    Test a PyTorch Lightning model.

    Args:
        trainer (pl.Trainer): The PyTorch Lightning Trainer instance used to run validation.
        model (pl.LightningModule): The trained model to be tested.
        loaders (Dict[str, DataLoader]): Dictionary of DataLoaders, keyed by names (e.g., 'train', 'val').
        test_key (str): Key used to select the test DataLoader from `loaders` (default: "test").
        seed (int): Random seed for reproducibility during validation (default: 42).
        tracker (Optional[object]): Optional tracker with `start()` and `stop()` methods.
        profiler (Optional[object]): Optional profiler with `start_profile()`, `stop_profile()`,
            and `print_model_profile(output_file)` methods.

    Returns:
        None
    """
    # Set a random seed for reproducibility
    pl.seed_everything(seed, workers=True, verbose=False)

    # Start the tracker and profiler if they are provided
    if tracker is not None: tracker.start()
    if profiler is not None: profiler.start_profile()
    
    # Check if test data loaders are specified and handle them accordingly
    # (single test DataLoader if `test_key` is a string, or multiple if `test_key` is a list)
    if isinstance(test_key, str):
        test_dataloaders = loaders[test_key]
    elif isinstance(test_key, list):
        test_dataloaders = {key: loaders[key] for key in test_key}
    else:
        raise NotImplementedError

    # Test the model using the trainer
    trainer.test(model, test_dataloaders)
    
    # Stop the tracker and profiler if they are provided
    if tracker is not None:
        tracker.stop()
    if profiler is not None:
        profiler.print_model_profile(output_file = f"{profiler.output_dir}/test_flops.txt")
        profiler.stop_profile()

    # # (1) load the best checkpoint automatically (lightning tracks this for you during .fit())
    # trainer.test(ckpt_path="best")

    # # (2) load the last available checkpoint (only works if `ModelCheckpoint(save_last=True)`)
    # trainer.test(ckpt_path="last")

# Function to shutdown data loader workers in a distributed setting
def shutdown_dataloaders_workers():
    """
    Shutdown data loader workers in a distributed setting.

    Args:
        None

    Returns:
        None
    """
    # Check if PyTorch is distributed initialized
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()

# Function to load a PyTorch Lightning model from a checkpoint
def load_model(model_cfg, path, **kwargs):
    """
    Load a PyTorch Lightning model from a checkpoint.

    Args: 
        model_cfg (dict): Configuration parameters for the model.
        path (str): Path to the checkpoint file.
        **kwargs: Additional keyword arguments to pass to the BaseNN constructor.

    Returns:
        The loaded PyTorch Lightning model.
    """
    # Load the model from the checkpoint file using the BaseNN class
    model = BaseNN.load_from_checkpoint(path, **model_cfg, **kwargs)
    return model

# Function to load log data from a CSV file
def load_logs(name, exp_id, project_folder="../"):
    """
    Load log data from a CSV file.

    Args:
        name(str): Name of the log file.
        exp_id(str): Experiment ID.
        project_folder(str): Path to the project folder (default: "../").

    Returns:
        Loaded log data as a Pandas DataFrame.
    """
    # Construct the file path to the log data
    file_path = os.path.join(project_folder, "out", "log", name, exp_id, "lightning_logs", "version_0", "metrics.csv")

    # Load CSV data into a Pandas DataFrame
    logs = pd.read_csv(file_path)

    return logs