Custom Callbacks
====================================

This guide explains how to create your own custom callback class and make it
discoverable by the training library via the ``additional_module`` argument in
``easy_torch.preparation.complete_prepare_trainer``.

.. contents::
   :depth: 2
   :local:


1. Folder Structure
--------------------

Make sure your project is organized as follows:

.. code-block:: text

   project_root/
   ├── ntb/
   │   └── your_notebook.ipynb
   └── src/
       ├── __init__.py
       └── my_additional_module/
           ├── __init__.py
           └── callbacks.py

The ``my_additional_module`` folder will contain all your custom components,
including callbacks.


2. Define Your Callback
------------------------

Inside ``src/my_additional_module/callbacks.py``, define your custom callback
class. The class must inherit from ``pytorch_lightning.callbacks.Callback`` (or the
equivalent base callback class used by your framework).

.. code-block:: python

   # src/my_additional_module/callbacks.py
   import pytorch_lightning as pl

   class CustomCallback(pl.callbacks.Callback):
       """Logs learning rate at the end of each training epoch."""

       def on_train_epoch_end(self, trainer, pl_module):
           lr = trainer.optimizers[0].param_groups[0]["lr"]
           print(f"Current learning rate: {lr:.6f}")


3. Package Initialization
-------------------------

Ensure the following minimal ``__init__.py`` files exist:

.. code-block:: python

   # src/__init__.py
   from . import my_additional_module

.. code-block:: python

   # src/my_additional_module/__init__.py
   from . import callbacks
   from .callbacks import CustomCallback  # optional


4. Make the Package Importable
-------------------------------

If your notebook or script is located in ``ntb/``, add the project root
directory to Python’s module search path at the top of your notebook:

.. code-block:: python

   import sys, os
   sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

Then import your module:

.. code-block:: python

   from src import my_additional_module


5. Reference the Callback in Your Config
------------------------------------------
In your configuration file or dictionary, specify the callback by name:

.. code-block:: python

   cfg = {
       "model": {
           "trainer_params": {
               "callbacks": [
                   {"CustomCallback": {"param1": "value1"}}
               ]
           }
       }
   }


6. Initialize the Trainer
--------------------------

When creating your trainer, pass the additional module that includes the
callbacks:

.. code-block:: python

   from src import my_additional_module

   trainer = easy_torch.preparation.complete_prepare_trainer(
       cfg,
       experiment_id,
       additional_module=my_additional_module
   )

