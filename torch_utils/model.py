# Import necessary libraries
import torch
import pytorch_lightning as pl
from .losses import NCODLoss
import torch.optim as optim
#NCODLoss has manual optmization as written here https://lightning.ai/docs/pytorch/stable/model/manual_optimization.html# according
#to the paper https://github.com/RSTLess-research/NCOD-Learning-with-noisy-labels/tree/main

# Define the BaseNN class
class BaseNN(pl.LightningModule):
    def __init__(self, main_module, loss, optimizer, metrics={}, log_params={}, **kwargs):
        super().__init__()

        # Store the main neural network module
        self.main_module = main_module

        # Store the optimizer function
        self.optimizer = optimizer

        # Store the primary loss function
        self.loss = loss
        
        # Check if the loss is an instance of NCODLoss and set automatic_optimization accordingly to False if True
        self.automatic_optimization = not isinstance(self.loss, NCODLoss)
        print("Using automatic optimization set to: ",self.automatic_optimization)

        # Define the metrics to be used for evaluation
        self.metrics = metrics

        # Prototype for customizing logging for multiple losses (if needed)
        # self.losses = loss
        # self.loss_log_params = {}
        # self.loss_weights = {}
        # for loss_name,loss_obj in self.losses.items():
        #     if isinstance(loss_obj, dict):
        #         self.losses[loss_name] = loss_obj["loss"]
        #         self.loss_log_params[loss_name] = loss_obj.get("log_params", {})
        #         self.loss_weights[loss_name] = loss_obj.get("weight", 1.0)
        #     else:
        #         self.losses[loss_name] = loss_obj
        #         self.loss_log_params[loss_name] = {}
        #         self.loss_weights[loss_name] = 1.0


        # Prototype for customizing logging for multiple metrics (if needed)
        # self.metrics = {}
        # self.metrics_log_params = {}
        # for metric_name, metric_obj in self.metrics.items():
        #     if isinstance(metric_obj, dict):
        #         self.metrics[metric_name] = metric_obj["metric"]
        #         self.metrics_log_params[metric_name] = metric_obj.get("log_params", {})
        #     else:
        #         self.metrics[metric_name] = metric_obj
        #         self.metrics_log_params[metric_name] = {}

        # Define a custom logging function
        self.custom_log = lambda name, value: self.log(name, value, **log_params)

    # Define the forward pass of the neural network
    def forward(self, x):
        return self.main_module(x)

    # Configure the optimizer for training
    def configure_optimizers(self):
        if isinstance(self.loss, NCODLoss):
            optimizer1 = self.optimizer(self.main_module.parameters())
            optimizer2 = optim.Adam(self.loss.parameters(), lr=0.001)
            # Define learning rate schedulers
            scheduler1 = {
                'scheduler': optim.lr_scheduler.MultiStepLR(optimizer1, milestones=[80,120], gamma=0.1),
                'interval': 'epoch',
                'frequency': 1
            }
            print("USING OPTIMIZERS FOR NCOD_LOSSS...")
            return [optimizer1, optimizer2], [scheduler1]
            
        optimizer1 = self.optimizer(self.parameters())   
        return optimizer1

    #def on_epoch_end(self):
        # Step through each scheduler
        #print("UPDATING THE LR....")
        #for scheduler in self.lr_schedulers():
            #scheduler.step()     

    # Define a step function for processing a batch
    def step(self, batch, batch_idx, split):
        x, y, index = batch
        
        #https://github.com/RSTLess-research/NCOD-Learning-with-noisy-labels/tree/main
        if isinstance(self.loss, NCODLoss):
            y_hat,out = self(x)
            #print("SIZE OUTPUT",y_hat.size())
            #print("SIZE ENCODER FEATURES", out.size())
            loss = self.loss(index, y_hat, y, out, batch_idx, self.current_epoch)
            
        else:
            y_hat,out = self(x)
            loss = self.loss(y_hat, y)

        self.custom_log(split+'_loss', loss)
        
        # Compute other metrics
        for metric_name, metric_func in self.metrics.items():
            metric_value = metric_func(y_hat, y)
            self.custom_log(split+'_'+metric_name, metric_value)

        if split == "train" and isinstance(self.loss, NCODLoss):
            # Perform the backward pass to calculate gradients
            self.manual_backward(loss)
            
            # Loop over all optimizers
            for optimizer_idx, optimizer in enumerate(self.optimizers()):
                #print("MANUAL OPTIMIZATION OF NCODLOSS: STEP AND ZERO GRAD FOR OPTIMIZER: ", optimizer_idx)
                
                # Update parameters of the current optimizer
                optimizer.step()

                # Zero gradients of the current optimizer
                optimizer.zero_grad()

                if optimizer_idx == 0 and self.trainer.is_last_batch and (self.trainer.current_epoch + 1) % 1 == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    print(current_lr)
                
            schedulers = self.lr_schedulers()
            if isinstance(schedulers, list):  # Check if it's a list of schedulers
                if (self.trainer.current_epoch == 0 and self.trainer.is_last_batch) or (self.trainer.current_epoch == 9 and self.trainer.is_last_batch):
                    for scheduler in schedulers:
                        print("SCHEDULERS")
                        scheduler.step()
            else:  # If it's a single scheduler object
                if (self.trainer.current_epoch == 0 and self.trainer.is_last_batch) or (self.trainer.current_epoch == 9 and self.trainer.is_last_batch):
                    print("SINGLE SCHEDULER")
                    schedulers.step()
                    
        return loss

    # Training step
    def training_step(self, batch, batch_idx): return self.step(batch, batch_idx, "train")

    # Validation step
    def validation_step(self, batch, batch_idx, dataloader_idx=0): return self.step(batch, batch_idx, "val")

    # Test step
    def test_step(self, batch, batch_idx): return self.step(batch, batch_idx, "test")

# Define functions for getting and loading torchvision models
def get_torchvision_model(*args, **kwargs): return torchvision_utils.get_torchvision_model(*args, **kwargs)

def load_torchvision_model(*args, **kwargs): return torchvision_utils.load_torchvision_model(*args, **kwargs)

# Define an Identity module
class Identity(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

# Define a LambdaLayer module
class LambdaLayer(torch.nn.Module):
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
from . import torchvision_utils  # put here otherwise circular import
