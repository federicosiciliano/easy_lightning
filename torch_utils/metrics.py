import torch.nn.functional as F
import torchmetrics
import torch

def nn_accuracy(y_hat,y):
    soft_y_hat = F.softmax(y_hat).argmax(dim=-1)
    soft_y = y.argmax(dim=-1)
    acc = (soft_y_hat.int() == soft_y.int()).float().mean()

    return acc

#Custom Accuracy to compute accuracy with Soft Labels as torch.Module
class SoftLabelsAccuracy(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds: torch.Tensor, target: torch.Tensor):
        return (preds.argmax(dim=1) == target.argmax(dim=1)).float().mean()


#Custom Accuracy to compute accuracy with Soft Labels
class SoftLabelsAccuracy2(torchmetrics.Metric):
    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        #preds, target = self._input_format(preds, target)
        #assert preds.shape == target.shape

        self.correct += torch.sum(preds.argmax(dim=1) == target.argmax(dim=1))
        self.total += target.shape[0]

    def compute(self):
        return self.correct.float() / self.total