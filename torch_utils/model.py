import torch
import pytorch_lightning as pl

class BaseNN(pl.LightningModule):
    def __init__(self, main_module, loss, optimizer, metrics={}, log_params={}, **kwargs):
        super().__init__()

        self.main_module = main_module

        self.optimizer = optimizer

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

        self.loss = loss

        self.metrics = metrics

        # Prototype for log different for metric
        # self.metrics = {}
        # self.metrics_log_params = {}
        # for metric_name,metric_obj in self.metrics.items():
        #     if isinstance(metric_obj, dict):
        #         self.metrics[metric_name] = metric_obj["metric"]
        #         self.metrics_log_params[metric_name] = metric_obj.get("log_params", {})
        #     else:
        #         self.metrics[metric_name] = metric_obj
        #         self.metrics_log_params[metric_name] = {}

        self.custom_log = lambda name, value: self.log(name, value, **log_params)

    def forward(self, x):
        return self.main_module(x)

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters())
        return optimizer

    def step(self, batch, batch_idx, split):
        x,y = batch

        y_hat = self(x)

        loss = self.loss(y_hat, y)

        self.custom_log(split+'_loss', loss)

        #on_step=False, on_epoch=True, logger=True

        #compute other metrics
        for metric_name,metric_func in self.metrics.items():
            metric_value = metric_func(y_hat,y)
            self.custom_log(split+'_'+metric_name, metric_value)
        
        return loss

    def training_step(self, batch, batch_idx): return self.step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx, dataloader_idx=0): return self.step(batch, batch_idx, "val")
        
    def test_step(self,batch,batch_idx): return self.step(batch, batch_idx, "test")

def get_torchvision_model(*args, **kwargs): return torchvision_utils.get_torchvision_model(*args, **kwargs)

def load_torchvision_model(*args, **kwargs): return torchvision_utils.load_torchvision_model(*args, **kwargs)


class Identity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x


class LambdaLayer(torch.nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)


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


from . import torchvision_utils #put here otherwise circular import