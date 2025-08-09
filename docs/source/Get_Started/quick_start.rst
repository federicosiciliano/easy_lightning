Quick Start Guide
=================

This guide will walk you through running the included quick start examples for **easy-lightning**.  
Two example scripts are provided:

- ``quick_start_rec.py`` — A quick start using the ``rec`` backend.
- ``quick_start_torch.py`` — A quick start using the PyTorch backend.

Before You Begin
----------------

Make sure you have **easy-lightning** installed and have initialized your project:

.. code-block:: bash

    pip install easy-lightning
    easy-lightning init

Running the ``rec`` Quick Start
-------------------------------

The ``quick_start_rec.py`` script demonstrates a simple experiment using the **rec** backend.  
To launch it, run:

.. code-block:: bash

    python quick_start_rec.py

This will execute the experiment using the preconfigured settings in the script.  
Check the console output and generated logs for training progress and results.

Running the PyTorch Quick Start
-------------------------------

The ``quick_start_torch.py`` script shows how to set up and run a quick experiment using **PyTorch**.  
Run it with:

.. code-block:: bash

    python quick_start_torch.py

Like the ``rec`` example, this will run with a minimal setup so you can quickly verify your environment and get results.


Experiment Directory Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The base directory for saving experiments is defined in the global configuration using the ``exp_name`` field.  
This corresponds to a top-level folder that will contain all related experiment runs.

Within this folder, EasyLightning automatically organizes output files into the following structure:

- ``out/log/`` — Contains all logs and metric outputs, including training and evaluation statistics.
- ``out/exp/`` — Stores the full YAML configuration files used for each experiment.
- ``out/models/`` — Contains the saved model weights (checkpoints) generated during training.

For more details on how to configure the experiment directory and related settings, see the :doc:`Configuration <../User_Guide/config>` page in the documentation.


Next Steps
----------

After running these quick start scripts:

- Review the code in each script to understand the configuration and workflow.
- Modify the parameters to match your dataset or model.
- Explore advanced usage in the full documentation.
