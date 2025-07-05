easy_torch config
=================

Data Parameters
-----------------------------------------
- ``name (str)`` – Name of the dataset to use. Example: `MNIST`.
- ``source (str)`` – Data source identifier. Options include [`torchvision`, `uci`, `tfds`, `custom`]. Default is **torchvision**.
- ``merge_before_split (bool)`` – Whether to merge training/validation/test sets before splitting. Default to **False**.
- ``split_keys (dict)`` – Keys defining how to split the data. Example: `{"train_x": ["train_x", "val_x"], "train_y": ["train_y", "val_y"]}`.
- ``train_sizes (list of int)`` – Number of samples or percentage to use for training. Default to **[100, 100]**.
- ``test_sizes (list of float or int)`` – Size of the test set. Interpreted as a proportion or absolute number. Default to **[0.2]**.
- ``split_random_state (int)`` – Random seed used to ensure reproducibility of splits. Default to **21094**.
- ``one_hot_encode (bool)`` – Whether to apply one-hot encoding to target labels. Default to **True**.
- ``scaling (str or None)`` – Method to scale input features. Options in [`None`, `MinMax`, `Standard`]. Default to **MinMax**.


Loader Parameters
-----------------------------------------
- ``batch_size (int)`` – Number of samples per batch during training or evaluation. Default to **256**.
- ``num_workers (int)`` – Number of subprocesses to use for data loading. Default to **1**.
- ``dtypes (str)`` – Data type for tensors. Default to **float32**.


Trainer Parameters
-----------------------------------------
- ``accelerator (str)`` – Type of accelerator to use. Options in [`cpu`, `gpu`, `auto`]. Default to **auto**.
- ``enable_checkpointing (bool)`` – Whether to save checkpoints during training. Default to **True**.
- ``max_epochs (int)`` – Maximum number of training epochs. Default to **1**.
- ``callbacks (list of dict)`` – List of training callbacks.

  - ``EarlyStopping``  
    - ``monitor (str)`` – Metric to monitor for early stopping. Example: `val_loss`.
    - ``mode (str)`` – Direction of improvement. Options: [`min`, `max`]. Default: **min**.
    - ``patience (int)`` – Number of epochs without improvement before stopping. Default: **1**.

  - ``ModelCheckpoint``  
    - ``dirpath (str)`` – Directory to save model checkpoints. Example: `${__exp__.project_folder}/out/models/${__exp__.name}/`.
    - ``filename (str)`` – Name format for saved checkpoint files. Default: **best**.
    - ``save_top_k (int)`` – Number of best models to retain. Default: **1**.
    - ``save_last (bool)`` – Whether to save the last model checkpoint. Default: **True**.
    - ``monitor (str)`` – Metric to evaluate for saving. Example: `val_loss`.
    - ``mode (str)`` – Direction to optimize. Options: [`min`, `max`]. Default: **min**.

- ``logger (dict)`` – Logging configuration.
  - ``name (str)`` – Logger class name. Example: `CSVLogger`.
  - ``params.save_dir (str)`` – Directory to save logs. Example: `${__exp__.project_folder}/out/log/${__exp__.name}/`.


Neural Network Parameters
-----------------------------------------
- ``num_neurons (list of int)`` – List of neuron counts to sweep for hidden layers. Default to **[1, 2, 4, 8, 16, 32, 64, 128]**.
- ``num_layers (list of int)`` – Number of layers to sweep over. Default to **[1, 2, 3, 4, 5]**.
- ``lr (list of float)`` – Learning rate values to sweep over. Default to **[1.0e-2, 1.0e-3, 1.0e-4]**.
- ``activation_function (list of str)`` – List of activation functions to use. Options include [`Tanh`, `LeakyReLU`].


ResNet Parameters
-----------------------------------------
- ``name (str)`` – Name of the ResNet architecture. Example: `resnet18`.
- ``torchvision_params.weights (str or None)`` – Pretrained weights to use. Default: **Null** (no pretraining).
- ``optimizer.name (str)`` – Name of the optimizer. Example: `Adam`.
- ``optimizer.params.lr (float)`` – Learning rate. Default: **0.1**.
- ``optimizer.params.weight_decay (float)`` – Weight decay for optimizer. Default: **0.0005**.
- ``loss (str)`` – Loss function used for training. Example: `CrossEntropyLoss`.
- ``log_params.on_epoch (bool)`` – Whether to log metrics at the end of each epoch. Default: **True**.
- ``log_params.on_step (bool)`` – Whether to log metrics at every step. Default: **False**.


Experiment Metadata
-----------------------------------------
- ``__exp__.name (str)`` – Name of the experiment. Example: `prova`.
- ``__exp__.__imports__ (list of modules)`` – List of modules to import before parsing the config. Example: [`torchvision`].

``+loader_params`` and ``+trainer_params`` are shorthand inclusion directives for loading shared configurations, typically defined elsewhere in modular configuration files.
``+model`` indicates the model definition to be used (e.g., `resnet`, `nn`).
``seed`` may be defined globally via `${exp.seed}` to ensure reproducibility across runs.
