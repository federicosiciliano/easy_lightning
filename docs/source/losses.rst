Defining and Loading Metrics in EasyLightning
=============================================

1. How to Define Metrics in ``model.yaml``
------------------------------------------

You can define metrics in two ways:

**Simple format**
::

    metrics:
      - MetricName

**Advanced format (for metrics that return dictionaries)**
::

    metrics:
      - FakeMetricCollection:
          metric_class: MetricName

If your metric returns a dictionary (e.g., multiple values like ``{"ndcg@5": ..., "ndcg@10": ...}``), you need to wrap it using ``FakeMetricCollection`` to avoid issues with Lightning and TorchMetrics.

2. How to Load Metrics in Your Code
-----------------------------------

In your ``main.py``, use the following:
::

    from src import your_module
    from easy_torch.preparation import prepare_metrics

    metrics = prepare_metrics(cfg["model"]["metrics"], your_module)

The function will try to locate and load each metric class by searching in this order:

1. ``your_module``
2. ``easy_lightning.metrics``
3. ``torchmetrics``

If a class is not found, an error will be raised.

**Loading from multiple modules**
::

    metrics = prepare_metrics(cfg["model"]["metrics"], [your_module, another_module])

3. Using the Metrics in the Standard Pipeline
---------------------------------------------

If you're using the standard EasyLightning pipeline, you can inject the metric list or collection into:
::

    model_params["metrics"]

This ensures that all metrics are correctly used and logged during training and evaluation.

Additional Notes
----------------

For additional details about how inputs and outputs are handled internally, see the ``step_routing`` section.
