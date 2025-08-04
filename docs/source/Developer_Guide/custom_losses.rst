Custom Loss
=====================================================

1. How to Define a Loss in ``model.yaml``
-----------------------------------------

You can define the loss in two ways:

**Simple format**
::

    loss: NameLoss

**Advanced format (for configurable losses)**
::

    loss:
      internal_name_loss:  # internal key used by the model to log/save loss values
        name: NameLoss      # name of the loss class
        params:             # parameters passed to the loss constructor
          param1: abc
          param2: 123

2. How to Load the Loss in Your Code
------------------------------------

In your ``main.py``, use the following:
::

    from src import your_module
    from easy_torch.preparation import prepare_loss

    your_loss = prepare_loss(cfg["model"]["loss"], your_module)

The function will try to locate and load the class ``NameLoss`` by searching in this order:

1. ``your_module``
2. ``easy_lightning.losses``
3. ``torch.nn``

If the class is not found, an error will be raised.

**Loading from multiple modules**
::

    your_loss = prepare_loss(cfg["model"]["loss"], [your_module, another_module])

3. Using the Loss in the Standard Pipeline
------------------------------------------

If you're using the standard EasyLightning pipeline, you can inject the loss into:
::

    model_params["loss"]

This allows the training loop to automatically use the configured loss.

Additional Notes
----------------

For additional details about how inputs and outputs are handled internally, see the ``step_routing`` section.
