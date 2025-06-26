# Import necessary libraries
import pandas as pd  # Import Pandas library for data manipulation
import os  # Import the os library for working with the file system
import torch  # Import the PyTorch library for deep learning
import pytorch_lightning as pl  # Import PyTorch Lightning for training and logging
from .model import BaseNN  # Import the BaseNN class from the model module

# Function to create a neural network model
def create_model(main_module, seed=42, **kwargs):
    """
    Create a PyTorch Lightning model.

    Parameters:
    - main_module: The main neural network module.
    - loss: The loss function for training.
    - optimizer: The optimizer for updating model parameters.
    - metrics: Dictionary of evaluation metrics.
    - log_params: Parameters for logging.
    - seed: Random seed for reproducibility.

    Returns:
    - model: The PyTorch Lightning model.
    """
    pl.seed_everything(seed) # Set a random seed for weight initialization --> not needed?
    # Create the model using the BaseNN class
    model = BaseNN(main_module, **kwargs)
    return model

# Function to train a PyTorch Lightning model
def train_model(trainer, model, loaders, train_key="train", val_key="val", seed=42, tracker=None, profiler=None):
    """
    Train a PyTorch Lightning model.

    Parameters:
    - trainer: The PyTorch Lightning trainer.
    - model: The PyTorch Lightning model to be trained.
    - loaders: Dictionary of data loaders.
    - train_key: Key for the training data loader.
    - val_key: Key for the validation data loader (optional).

    Returns:
    - None
    """
    pl.seed_everything(seed) # Set a random seed for deterministic training

    # Check if validation data loaders are specified
    if val_key is not None:
        if isinstance(val_key, str):
            val_dataloaders = loaders[val_key]
        elif isinstance(val_key, list):
            val_dataloaders = {key: loaders[key] for key in val_key}
        else:
            raise NotImplementedError
    else:
        val_dataloaders = None
    
    if tracker is not None: tracker.start()
    if profiler is not None: profiler.start_profile()
    
    trainer.fit(model, loaders[train_key], val_dataloaders)
    
    if tracker is not None:
        tracker.stop()
    if profiler is not None:
        profiler.print_model_profile(output_file = f"{profiler.output_dir}/train_flops.txt")
        profiler.stop_profile()

# Function to validate a PyTorch Lightning model
def validate_model(trainer, model, loaders, loaders_key="val", seed=42):
    """
    Validate a PyTorch Lightning model.

    Parameters:
    - trainer: The PyTorch Lightning trainer.
    - model: The PyTorch Lightning model to be validated.
    - loaders: Dictionary of data loaders.
    - loaders_key: Key for the validation data loader.

    Returns:
    - None
    """
    pl.seed_everything(seed, workers=True)

    # Validate the model using the trainer
    trainer.validate(model, loaders[loaders_key])

# Function to test a PyTorch Lightning model
def test_model(trainer, model, loaders, test_key="test", tracker=None, profiler=None, seed=42):
    """
    Test a PyTorch Lightning model.

    Parameters:
    - trainer: The PyTorch Lightning trainer.
    - model: The PyTorch Lightning model to be tested.
    - loaders: Dictionary of data loaders.
    - test_key: Key for the test data loader.

    Returns:
    - None
    """
    pl.seed_everything(seed, workers=True)

    if tracker is not None: tracker.start()
    if profiler is not None: profiler.start_profile()
    
    if isinstance(test_key, str):
        test_dataloaders = loaders[test_key]
    elif isinstance(test_key, list):
        test_dataloaders = {key: loaders[key] for key in test_key}
    else:
        raise NotImplementedError

    # Test the model using the trainer
    trainer.test(model, test_dataloaders)
    
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

    Parameters:
    - None

    Returns:
    - None
    """
    # Check if PyTorch is distributed initialized
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()

# Function to load a PyTorch Lightning model from a checkpoint
def load_model(model_cfg, path, **kwargs):
    """
    Load a PyTorch Lightning model from a checkpoint.

    Parameters:
    - model_cfg: Configuration parameters for the model.
    - path: Path to the checkpoint file.

    Returns:
    - model: The loaded PyTorch Lightning model.
    """
    # Load the model from the checkpoint file using the BaseNN class
    model = BaseNN.load_from_checkpoint(path, **model_cfg, **kwargs)
    return model

# Function to load log data from a CSV file
def load_logs(name, exp_id, project_folder="../"):
    """
    Load log data from a CSV file.

    Parameters:
    - name: Name of the log file.
    - exp_id: Experiment ID.
    - project_folder: Path to the project folder (default: "../").

    Returns:
    - logs: Loaded log data as a Pandas DataFrame.
    """
    # Construct the file path to the log data
    file_path = os.path.join(project_folder, "out", "log", name, exp_id, "lightning_logs", "version_0", "metrics.csv")

    # Load CSV data into a Pandas DataFrame
    logs = pd.read_csv(file_path)

    return logs