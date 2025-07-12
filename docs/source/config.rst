Config
========

EasyLightning is designed with modularity and customizability in mind. All experiment settings—ranging from datasets and data loaders to models and training routines—are defined through human-readable YAML configuration files.

These configuration files make it easy to:

- Reproduce experiments
- Swap models or datasets
- Customize the data pipeline
- Tune hyperparameters
- Define evaluation metrics and logging preferences

Each section of the YAML file corresponds to a major component of the pipeline, including:

- **Dataset parameters** – Define how data is loaded, filtered, and preprocessed.
- **Loader parameters** – Control batching and data pipeline settings.
- **Training parameters** – Specify hardware configuration, training duration, logging, and checkpointing.
- **Model parameters** – Configure model architecture, hyperparameters, and specific components.
- **Global and routing parameters** – Enable advanced data handling and specify how data flows through the system.

This structured configuration ensures consistency, reusability, and clarity across projects, making it easy to scale or adapt experiments to new scenarios with minimal effort.



**Note**:
Both **Easy Torch** and **Easy Rec** support seamless integration with **PyTorch** and **PyTorch Lightning**. Models, checkpoints, loss functions, and metrics can be directly referenced from these frameworks using string-based import paths in the YAML configuration (e.g., `torch.nn.CrossEntropyLoss`). This design provides full flexibility and extensibility while maintaining the simplicity of EasyLightning's unified configuration system.


--------
.. toctree::
   :maxdepth: 4

   easy_rec_config

Easy Torch
----------
.. toctree::
   :maxdepth: 4

   easy_torch_config

Special Characters
------------------

YAML configuration files in EasyLightning use special characters to control behavior in experiment definitions. Proper quoting and formatting are essential to avoid parsing errors.

Below are some special characters and their usage:

1. £ (Sweep Operator)

   The `£` prefix is used to define a hyperparameter sweep over a range of values.

   Example:
     £learning_rate:
       default: 0.001
       values: [0.001, 0.01, 0.1]

   In your `quick_start.py` script, you can iterate over the sweep like this:

     for _ in cfg.sweep(cfg["model"]["learning_rate"]):

   This enables automatic experimentation over multiple values. See `quick_start.py` for more details.

2. / (Exclude from Config and Experiment ID)

   The `/` prefix marks a parameter as **excluded** from being saved in the final config file and from affecting the `exp_id` (experiment identifier).

   Example:
     /learning_rate: 0.001

   This means the parameter will be used during execution but ignored when saving configuration files or generating experiment names.
