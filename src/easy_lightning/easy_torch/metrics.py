import torch.nn.functional as F
import torchmetrics
import torch

# Custom Accuracy to compute accuracy with Soft Labels as a torchmetrics.Metric
class SoftLabelsAccuracy(torchmetrics.Metric):
    def __init__(self):
        super().__init__()
        # Initialize state variables for correct predictions and total examples
        self.add_state("correct", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, input: torch.Tensor, target: torch.Tensor):
        # Update correct predictions and total examples
        self.correct += torch.sum(input.argmax(dim=1) == target.argmax(dim=1))
        self.total += target.shape[0]

    def compute(self):
        # Compute accuracy as the ratio of correct predictions to total examples
        return self.correct.float() / self.total
    
# Function to compute accuracy for neural network predictions
# def nn_accuracy(y_hat, y):
#     # Apply softmax to predictions and get the class with the highest probability
#     soft_y_hat = F.softmax(y_hat).argmax(dim=-1)
#     soft_y = y.argmax(dim=-1)
    
#     # Calculate accuracy by comparing predicted and actual class labels
#     acc = (soft_y_hat.int() == soft_y.int()).float().mean()
#     return acc

# Custom Accuracy to compute accuracy with Soft Labels as a torch.Module
# class SoftLabelsAccuracy(torch.nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, preds: torch.Tensor, target: torch.Tensor):
#         # Calculate accuracy by comparing predicted and actual class labels
#         return (preds.argmax(dim=1) == target.argmax(dim=1)).float().mean()

class BatchLength(torchmetrics.Metric):
    """
    A metric to compute the average batch length.

    Args:
        batch_size (int): The size of the batch.
    """
    def __init__(self, batch_dim=1):
        super().__init__()
        self.out_keys = ["avg", "max", "min"]
        self.batch_dim = batch_dim
        self.add_state("total", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("min", default=torch.tensor(float("inf")), dist_reduce_fx="min")
        self.add_state("max", default=torch.tensor(float("-inf")), dist_reduce_fx="max")

    def update(self, batch: torch.Tensor, *args, **kwargs):
        """
        Updates the metric with the current batch length.

        Args:
            batch_length (torch.Tensor): The length of the current batch.
        """
        length = torch.tensor(batch.shape[self.batch_dim], dtype=torch.float32)
        self.total += length
        self.min = torch.minimum(self.min, length)
        self.max = torch.maximum(self.max, length)
        self.count += 1

    def compute(self):
        """
        Computes and returns the average batch length.
        """
        out = {
            "min": self.min,
            "max": self.max,
            "avg": self.total / self.count
        }
        return out
    
class FakeMetricCollection(torchmetrics.MetricCollection):
    # A collection of fake metrics that actually call just once the update / compute part
    def __init__(self, metric_class, keys_name="out_keys", *args, **kwargs):
        metric = metric_class(*args, **kwargs)
        keys = sorted(getattr(metric, keys_name))
        primary_key = str(keys[0])
        metrics = {primary_key: make_fake_class(metric_class)(primary_key, *args, **kwargs)}
        for k in keys[1:]:
            metrics[str(k)] = FakeMetric(metrics[primary_key], key=k)
        super().__init__(metrics)

def make_fake_class(base_class):
    class FakeTrueMetric(base_class):
        def __init__(self, key, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.primary_key = str(key)
            
        def compute(self, *args, **kwargs):
            out = super().compute(*args, **kwargs)
            for key,value in out.items():
                setattr(self, str(key), value)
            return out.get(self.primary_key)

    return FakeTrueMetric

class FakeMetric(torchmetrics.Metric):
    def __init__(self, true_metric, key, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.true_metric = true_metric
        self.key = key
    
    def update(self, *args, **kwargs):
        pass

    def compute(self, *args, **kwargs):
        return getattr(self.true_metric, self.key, None)