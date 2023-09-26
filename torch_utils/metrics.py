import torch.nn.functional as F
import torchmetrics
import torch

# Function to compute accuracy for neural network predictions
def nn_accuracy(y_hat, y):
    # Apply softmax to predictions and get the class with the highest probability
    soft_y_hat = F.softmax(y_hat).argmax(dim=-1)
    soft_y = y.argmax(dim=-1)
    
    # Calculate accuracy by comparing predicted and actual class labels
    acc = (soft_y_hat.int() == soft_y.int()).float().mean()
    return acc

# Custom Accuracy to compute accuracy with Soft Labels as a torch.Module
class SoftLabelsAccuracy(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds: torch.Tensor, target: torch.Tensor):
        # Calculate accuracy by comparing predicted and actual class labels
        return (preds.argmax(dim=1) == target.argmax(dim=1)).float().mean()

# Custom Accuracy to compute accuracy with Soft Labels as a torchmetrics.Metric
class SoftLabelsAccuracy2(torchmetrics.Metric):
    def __init__(self):
        super().__init__()
        # Initialize state variables for correct predictions and total examples
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # Update correct predictions and total examples
        self.correct += torch.sum(preds.argmax(dim=1) == target.argmax(dim=1))
        self.total += target.shape[0]

    def compute(self):
        # Compute accuracy as the ratio of correct predictions to total examples
        return self.correct.float() / self.total
