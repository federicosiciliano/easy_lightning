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

Easy Rec 
--------
.. toctree::
   :maxdepth: 4

   easy_rec_config

Easy Torch
----------
.. toctree::
   :maxdepth: 4

   easy_torch_config