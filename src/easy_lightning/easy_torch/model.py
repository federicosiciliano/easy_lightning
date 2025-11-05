# Import necessary libraries
import torch
import pytorch_lightning as pl
import torchmetrics

class BaseNN(pl.LightningModule):
    """ Base class for a neural network model in PyTorch Lightning.
    This class serves as a base for creating neural network models with customizable components such as the 
    main module, loss function, optimizer, and metrics.
    It also provides methods for logging, computing model outputs, losses, and metrics.
    Args:
        main_module (torch.nn.Module): The main neural network module.
        loss (torch.nn.Module or dict): The primary loss function or a dictionary of loss functions.
        optimizer (callable): The optimizer function to be used for training.
        scheduler (callable or dict, optional): Learning rate scheduler function or a dictionary containing scheduler configuration.
        metrics (dict): A dictionary of metrics to be used for evaluation.
        log_params (dict): Parameters for logging, such as whether to log on epoch end.
        step_routing (dict): A dictionary defining how batch and model output are routed to the model, loss, and metrics.
    """
    def __init__(self, main_module, loss, optimizer, scheduler=None, metrics={}, log_params={},
                 step_routing = {"model_input_from_batch":[0],
                                 "loss_input_from_batch": [1], "loss_input_from_model_output": None,
                                 "metrics_input_from_batch": [1], "metrics_input_from_model_output": None},
                 **kwargs):
        super().__init__()

        # Store the main neural network module
        self.main_module = main_module

        # Store the optimizer and scheduler
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Store the primary loss function
        self.loss = loss
        
        # Define the metrics to be used for evaluation
        self.metrics = metrics

        # Define how batch and model output are routed to model, loss and metrics
        self.step_routing = step_routing

        # Define a custom logging function
        self.log_params = log_params


    def log(self, name, value):
        """
        Custom logging function that handles logging of metrics and values.

        Args:
            name (str): Base name for the metric or value.
            value (Any): Value to log, which can be a scalar, dict, or torchmetrics.MetricCollection.
        """
        original_log_function = super().log
        if value is not None:
            # If value is a dictionary or a MetricCollection, log each item separately
            if isinstance(value, dict) or isinstance(value, torchmetrics.MetricCollection):
                for key,value_to_log in value.items():
                    log_name = "_".join([x for x in [name, key] if x is not None and x != ""])
                    self.log(log_name, value_to_log)
                    # if to_log.size() != 1 and len(to_log.size()) != 0: #Save metrics in batch;
                    #     if split_name == "test":
                    #         save_path = os.path.join(self.logger.save_dir, self.logger.name, f'version_{self.logger.version}',f"metrics_per_sample.csv")
                    #         with open(save_path, 'a') as f_object:
                    #             writer_object = csv.writer(f_object)
                    #             writer_object.writerow([log_key,*to_log.cpu().detach().tolist()])
                    #             f_object.close()
                    # else:
            # Otherwise, log the value directly
            else:
                original_log_function(name, value, **self.log_params)

        # original_log_function = super().log
        # if value is not None:
        #     if isinstance(value, dict) or isinstance(value, torchmetrics.MetricCollection):
        #         for key,value_to_log in value.items():
        #             log_name = "_".join([x for x in [name, key] if x is not None and x != ""])
        #             #self.log(log_name, value_to_log)
        #             if isinstance(value, torchmetrics.MetricCollection):
        #                 to_log = value_to_log.compute()
        #                 if to_log.size() != 1 and len(to_log.size()) != 0: #Save metrics in batch;
        #                     #if split_name == "test":
        #                         save_path = os.path.join(self.logger.save_dir, self.logger.name, f'version_{self.logger.version}',f"metrics_per_sample.csv")
        #                         with open(save_path, 'a') as f_object:
        #                             writer_object = csv.writer(f_object)
        #                             writer_object.writerow([log_name,*to_log.cpu().detach().tolist()])
        #                             f_object.close()
        #                         value_to_log.reset()
        #                 else:
        #                     self.log(log_name, value_to_log)
        #             else:
        #                 self.log(log_name, value_to_log)
        #     else:
        #         original_log_function(name, value, **self.log_params)

    
    def forward(self, *args, **kwargs):
        """
        Forward pass through the main module.

        Returns:
            torch.Tensor or Any: Output of the main model.
        """
        return self.main_module(*args, **kwargs)


    def configure_optimizers(self):
        """
        Configure the optimizer(s) and learning rate scheduler(s) for the model.

        Returns
        -------
        dict
            A dictionary containing the optimizer and optionally the learning rate scheduler.
            The dictionary can contain:
            
            - "optimizer": The optimizer instance.
            - "lr_scheduler": A dictionary or callable for the learning rate scheduler.
        """
        optimizer = self.optimizer(self.parameters())

        return_dict = {"optimizer": optimizer}

        if self.scheduler is not None:
            if isinstance(self.scheduler, dict):
                # If scheduler is a dict, we assume it contains the information to create a scheduler
                self.scheduler["scheduler"] = self.scheduler["scheduler"](optimizer)
                return_dict['lr_scheduler'] = self.scheduler
            else:  # If scheduler is not a dict, we assume it is a callable that returns a scheduler
                return_dict['lr_scheduler'] = self.scheduler(optimizer)
        return return_dict
    
    # def lr_scheduler_step(self, scheduler, metric):
    #     scheduler.step(epoch=self.current_epoch)  # if scheduler need the epoch value

    def step(self, batch, batch_idx, dataloader_idx, split_name): #not a lightning method
        #TODO: what to do with batch_idx and dataloader_idx?
        """
        Common step function for processing a batch.

        Args:
            batch (Any): Input batch from the dataloader.
            batch_idx (int): Index of the batch.
            dataloader_idx (int): Index of the dataloader (used for multi-dataloader scenarios).
            split_name (str): One of ["train", "val", "test", "predict"].

        Returns:
            dict: Dictionary containing model output, loss (if applicable), and metrics (if applicable).
        """
        # Compute the model output
        # Use the routing defined in step_routing to get the model input from the batch
        model_output = self.compute_model_output(batch, self.step_routing["model_input_from_batch"])
        lightning_module_return = {"model_output": model_output}

        # If loss is defined, compute the loss using the routing for loss input and the model output
        if self.loss is not None:
            lightning_module_return["loss"] = self.compute_loss(self.loss, batch, self.step_routing["loss_input_from_batch"],
                                    model_output, self.step_routing["loss_input_from_model_output"],
                                    split_name, dataloader_idx)

        # If metrics are defined, compute the metrics using the routing for metrics input and the model output
        if len(self.metrics)>0:
            lightning_module_return["metric_values"] = self.compute_metrics(batch, self.step_routing["metrics_input_from_batch"],
                                                model_output, self.step_routing["metrics_input_from_model_output"],
                                                split_name, dataloader_idx)

        return lightning_module_return

    def compute_model_output(self, batch, model_input_from_batch):
        """ 
        Compute the model output given a batch and the routing for model input.

        Args:
            batch (Any): Input batch from the dataloader.
            model_input_from_batch (list or dict): Routing for model input from the batch.
        Returns:
            torch.Tensor or Any: Output of the model.   
        """
        model_input_args, model_input_kwargs = self.get_input_args_kwargs((batch, model_input_from_batch))

        model_output = self(*model_input_args, **model_input_kwargs)
        
        return model_output
    
    def get_input_args_kwargs(self, *args):
        """ 
        Get postional arguments and keyword arguments from the provided args.

        Args:
            *args: A tuple of objects and their corresponding keys.
        
        Returns:
            input_args (list): List of positional arguments extracted from the objects.
            input_kwargs (dict): Dictionary of input keyword arguments extracted from the objects.
        """
        input_args, input_kwargs = [],{}
        for obj,keys in args:
            # # If keys is a single int or str, convert it to a list for uniformity
            if isinstance(keys, int) or isinstance(keys, str):
                keys = [keys]
            # if keys is a list, extract multiple elements and append to postional args
            if isinstance(keys, list):
                input_args += [obj[i] for i in keys]
            # if key is a dictionary, we assume it is a mapping of keys to indices or None
            elif isinstance(keys, dict):
                for k,i in keys.items():
                    # If index is None, pass the whole object as that keyword argument
                    if i is None:
                        input_kwargs[k] = obj
                    # Otherwise, extract and value the value at the index i
                    else:
                        input_kwargs[k] = obj[i]
            # if keys is None, we assume the whole object is the input 
            elif keys is None:
                input_args.append(obj)
            else:
                raise NotImplementedError("keys type not recognized")
        return input_args, input_kwargs

    def compute_loss(self, loss_object, batch, loss_input_from_batch, model_output, loss_input_from_model_output, split_name, dataloader_idx):
        """
        Compute the loss given a batch and the routing for loss input.
        Args:
            loss_object (torch.nn.Module or dict): The loss function or a dictionary of loss functions.
            batch (Any): Input batch from the dataloader.               
            loss_input_from_batch (list or dict): Routing for loss input from the batch.
            model_output (torch.Tensor or Any): Output of the model.
            loss_input_from_model_output (list or dict): Routing for loss input from the model output.
            split_name (str): Data split name.
            dataloader_idx (int): Index of the dataloader (used for multi-dataloader scenarios).
        Returns:
            torch.Tensor: Computed loss value.
        """
        
        if isinstance(loss_object, torch.nn.ModuleDict):
            # If loss_object contains a specific loss function for the split_name, compute that loss
            if split_name in loss_object:
                loss = self.compute_loss(loss_object[split_name], batch, loss_input_from_batch, model_output, loss_input_from_model_output, split_name, dataloader_idx)
            # Otherwise, we compute the loss for each loss function in the dictionary
            else:
                loss = torch.tensor(0.0, device=self.device)
                # Compute each loss using the _compute method -which will handle the routing for each loss function-,
                # multiply it by its weight (if it exists), and accumulate the total loss
                for i, (loss_name, loss_func) in enumerate(loss_object.items()):
                    single_loss = self._compute(loss_name, loss_func, batch, loss_input_from_batch, model_output, loss_input_from_model_output, split_name)
                    weight = getattr(loss_object[loss_name], '__weight__', 1.0) # get weight if it exists
                    loss += weight * single_loss
                self.log(split_name+'_loss', loss)    
        elif isinstance(loss_object, torch.nn.ModuleList):
            # Use the dataloader_idx to select the appropriate loss function from the list and compute the loss
            loss = self.compute_loss(loss_object[dataloader_idx], batch, loss_input_from_batch, model_output, loss_input_from_model_output, split_name, dataloader_idx)
        else:
            loss = self._compute("loss", loss_object, batch, loss_input_from_batch, model_output, loss_input_from_model_output, split_name)
        return loss
    
    # Compute metrics given a batch and the routing for metrics input
    def compute_metrics(self, batch, metrics_input_from_batch, model_output, metrics_input_from_model_output, split_name, dataloader_idx):
        """
        Compute metrics using the specified metric functions.

        Args:
            batch (Any): Input batch from the dataloader.   
            metrics_input_from_batch (list or dict): Routing for metrics input from the batch.
            model_output (torch.Tensor or Any): Output of the model.
            metrics_input_from_model_output (list or dict): Routing for metrics input from the model output.
            split_name (str): Data split name.
            dataloader_idx (int): Index of the dataloader (used for multi-dataloader scenarios).
        
        Returns:
            dict: Dictionary containing computed metric values.
        """
        
        metric_values = {}
        for metric_name, metric_func in self.metrics[split_name][dataloader_idx].items():
            metric_values[metric_name] = self._compute(metric_name, metric_func, batch, metrics_input_from_batch, model_output, metrics_input_from_model_output, split_name)
        return metric_values
    
    # Compute a metric or loss given the name, function, batch, and routing information
    def _compute(self, name, func, batch, input_from_batch, model_output, input_from_model_output, split_name):
        """
        Compute a loss or metric value by extracting inputs from batch and model_output according to routing,
        and then applying the given function `func`.

        Args:
            name (str): Name of the metric or loss function.
            func (callable): The metric or loss function to compute.
            batch (Any): Input batch from the dataloader.
            input_from_batch (list or dict): Routing for inputs from the batch.
            model_output (torch.Tensor or Any): Output of the model.
            input_from_model_output (list or dict): Routing for inputs from the model output.
            split_name (str): Data split name.

        Returns:
            Any: Computed value from the function `func`, which can be a scalar, tensor, or a torchmetrics.MetricCollection.
        """
        # Extract the routing for inputs from the batch and model output
        batch_routing = self.get_key_if_dict_and_exists(input_from_batch, name)
        output_routing = self.get_key_if_dict_and_exists(input_from_model_output, name)

        # Get the input arguments and keyword arguments for the function
        input_args, input_kwargs = self.get_input_args_kwargs((batch, batch_routing), (model_output, output_routing))

        # if isinstance(func, torchmetrics.Metric): #This can compute the metric across batches #TODO? choose if we want to compute the metric across batches or not
        #     print("COMPUTING METRIC", split_name)
        #     func.update(*input_args,**input_kwargs)
        #     value = func#.compute()
        #     print("TOT:",func.total)
        # else:

        # Compute the value using the function `func` with the provided input arguments and keyword arguments
        # If `func` is a torchmetrics.Metric, it will return the value for the current batch
        value = func(*input_args, **input_kwargs) 

        # If `func` is a torchmetrics.Metric or MetricCollection, we will log it directly
        # Note: This won't work if the TorchMetrics.Metric returns a dict instead of a tensor
        if isinstance(func, torchmetrics.Metric) or isinstance(func, torchmetrics.MetricCollection): 
            value = func

        # Log the value 
        log_name = split_name+'_'+name
        self.log(log_name, value)

        return value
    
    def get_key_if_dict_and_exists(self, obj, key):
        # If obj is a dictionary and contains the key, return the value associated with the key.
        # Otherwise, return the object itself.
        # This is useful for routing inputs from batch or model output.
        if isinstance(obj, dict) and key in obj:
            return obj[key]
        else:
            return obj

    # Training step
    def training_step(self, batch, batch_idx, dataloader_idx=0): return self.step(batch, batch_idx, dataloader_idx, "train")

    # Validation step
    def validation_step(self, batch, batch_idx, dataloader_idx=0): return self.step(batch, batch_idx, dataloader_idx, "val")

    # Test step
    def test_step(self, batch, batch_idx, dataloader_idx=0): return self.step(batch, batch_idx, dataloader_idx, "test")
    
    # Predict step
    def predict_step(self, batch, batch_idx, dataloader_idx=0): return self.step(batch, batch_idx, dataloader_idx, "predict")

    # def on_train_epoch_end(self) -> None:
    #     self.on_epoch_end("train")
    
    # def on_validation_epoch_end(self) -> None:
    #     self.on_epoch_end("val")

    # def on_test_epoch_end(self) -> None:
    #     self.on_epoch_end("test")

    # def on_epoch_end(self, *args, **kwargs):
    #     pass

    # def on_epoch_end(self, *args, **kwargs):
    #     # Step through each scheduler
    #     for scheduler in self.lr_schedulers():
    #         scheduler.step()

    # def on_epoch_end(self, split_name): #not a lightning method
    #     if self.log_params.get("on_epoch", False):
    #         for dataloader_idx, metric_dict in enumerate(self.metrics[split_name]):
    #             for metric_name, metric_func in metric_dict.items():
    #                 log_name = f"{split_name}_{metric_name}"
    #                 print(metric_func.total)
    #                 print("LOGGING", log_name, f"epoch/dataloader_{dataloader_idx}")
    #                 self.log(log_name, metric_func.compute(), log_params={**self.log_params, "on_step": False}, suffix=f"epoch", dataloader_idx=dataloader_idx)
    #                 metric_func.reset()
    #     self.reset_metrics(self.metrics[split_name])
    
    # def reset_metrics(self, metrics):
    #     if isinstance(metrics, torchmetrics.Metric):
    #         metrics.reset()
    #         print("RESET", metrics)
    #     elif isinstance(metrics, torch.nn.ModuleList) or isinstance(metrics, list):
    #         for metric in metrics:
    #             self.reset_metrics(metric)
    #     elif isinstance(metrics, torch.nn.ModuleDict) or isinstance(metrics, dict):
    #         for metric in metrics.values():
    #             self.reset_metrics(metric)

# Define functions for getting and loading torchvision models
def get_torchvision_model(*args, seed=42, **kwargs):
    pl.seed_everything(seed) # Is this really useful?
    return torchvision_utils.get_torchvision_model(*args, **kwargs)

def get_torchvision_model_as_decoder(example_datum, *args, **kwargs):
    forward_model = torchvision_utils.get_torchvision_model(*args, **kwargs)
    inverted_model = torchvision_utils.invert_model(forward_model, example_datum)
    return inverted_model

def load_torchvision_model(*args, **kwargs): return torchvision_utils.load_torchvision_model(*args, **kwargs)

# Define an Identity module
class Identity(torch.nn.Module):
    """
    An Identity module that returns the input as is.
    This module can be used as a placeholder in a neural network architecture.
    It does not perform any operation on the input and simply returns it.
    Args:
        None
    Returns:
        torch.Tensor: The input tensor is returned unchanged.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

# Define a LambdaLayer module
class LambdaLayer(torch.nn.Module):
    """ 
    A LambdaLayer module that applies a custom function to the input.
    It is useful for applying custom transformations or operations in a neural network.
    Args:
        lambd (callable): A function that takes a tensor as input and returns a tensor as output.
    Returns:
        torch.Tensor: The output tensor after applying the custom function.          
    """
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

# Class MLP (Multi-Layer Perceptron) (commented out for now)
# class MLP(BaseNN):
#     def __init__(self, input_size, output_size, neurons_per_layer, activation_function=None, lr=None, loss = None, acc = None, **kwargs):
#         super().__init__()

#         layers = []
#         in_size = input_size
#         for out_size in neurons_per_layer:
#             layers.append(torch.nn.Linear(in_size, out_size))
#             if activation_function is not None:
#                 layers.append(getattr(torch.nn, activation_function)())
#             in_size = out_size
#         layers.append(torch.nn.Linear(in_size, output_size))
#         self.main_module = torch.nn.Sequential(*layers)

# Import additional libraries
from . import torchvision_utils # put here otherwise circular import
