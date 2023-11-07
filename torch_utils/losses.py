import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

#https://openaccess.thecvf.com/content_cvpr_2017/papers/Patrini_Making_Deep_Neural_CVPR_2017_paper.pdf
class ForwardNRL(nn.Module):
    def __init__(self, noise_rate, num_classes):
        super(ForwardNRL, self).__init__()
        self.noise_rate = noise_rate
        self.num_classes = num_classes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.matrix = self._construct_matrix(self.noise_rate, self.num_classes, self.device)
     

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = F.softmax(input, dim=1)
        p = torch.matmul(p, self.matrix.t())
        p = torch.log(p)
        res = p * target    
        loss = -torch.sum(res, dim=1)

        return torch.mean(loss)
    
    def _construct_matrix(self, noise_rate, num_classes, device):
        diagonal = 1 - noise_rate
        rest = noise_rate / (num_classes - 1)
        matrix = torch.full((num_classes, num_classes), rest, device=device)
        matrix.fill_diagonal_(diagonal)
        return matrix


#https://openaccess.thecvf.com/content_cvpr_2017/papers/Patrini_Making_Deep_Neural_CVPR_2017_paper.pdf
class BackwardNRL(nn.Module):
    def __init__(self, noise_rate, num_classes):
        super(BackwardNRL, self).__init__()
        self.noise_rate = noise_rate
        self.num_classes = num_classes
        self.matrix = self._construct_matrix(self.noise_rate, self.num_classes)
        self.matrix_inv = torch.inverse(self.matrix)

    def forward(self, input, target):
        log_probs = F.log_softmax(input, dim=1)
        num_samples = target.size(0)
        a = torch.matmul(self.matrix_inv, log_probs.t())
        i = a.t() * target
        m = torch.sum(i, dim=1)
        loss = -torch.mean(m)
        return loss

    def _construct_matrix(self, noise_rate, num_classes):
        diagonal = 1 - noise_rate
        rest = noise_rate / (num_classes - 1)
        matrix = torch.full((num_classes, num_classes), rest)
        matrix = matrix.fill_diagonal_(diagonal)
        return matrix
        
#https://github.com/dmizr/phuber/blob/master/phuber/loss.py
class GCELoss(nn.Module):
    """
    Computes the Generalized Cross Entropy (GCE) loss, which is especially useful for 
    training deep neural networks with noisy labels.
    Refer to "Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels"
    <https://arxiv.org/abs/1805.07836>
    
    Attributes:
        q (float): Box-Cox transformation parameter. Must be in (0,1].
        epsilon (float): A small value to avoid undefined gradient.
        softmax (nn.Softmax): Softmax function to convert raw scores to probabilities.
    """
    
    def __init__(self, q: float = 0.7) -> None:
        """
        Initializes the GCELoss module.
        
        Args:
            q (float): Box-Cox transformation parameter. Default is 0.7.
        """
        super().__init__()
        self.q = q
        self.epsilon = 1e-9  # A small value to avoid division by zero or log(0)
        self.softmax = nn.Softmax(dim=1)  # Softmax function to get probabilities
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the GCE loss between the predictions and targets.
        
        Args:
            param input: Predictions from the model (before softmax)
                          shape: (batch_size, num_classes)
            param target: True labels (one-hot encoded)
                           shape: (batch_size, num_classes)
            
        Returns:
            torch.Tensor: The mean GCE loss.
        """
        # Apply softmax to the raw scores to get probabilities
        p = self.softmax(input)
        
        # Multiply the softmax probabilities by the one-hot targets
        # and sum across classes to get the correct class probability
        p = torch.sum(p * target, dim=1)
        
        # Add epsilon to avoid undefined gradient due to log(0) or division by zero
        p += self.epsilon
        
        # Compute the GCE loss based on the selected probability and the Box-Cox transformation parameter
        loss = (1 - p ** self.q) / self.q
        
        # Return the mean loss
        return torch.mean(loss)


cross_entropy_val = nn.CrossEntropyLoss

mean = 1e-8
std = 1e-9
encoder_features = 512

class NCODLoss(nn.Module):
    def __init__(self, sample_labels, num_examp=50000, num_classes=100, ratio_consistency=0, ratio_balance=0, total_epochs=4000):
        super(NCODLoss, self).__init__()

        self.num_classes = num_classes
        self.USE_CUDA = torch.cuda.is_available()
        self.num_examp = num_examp

        self.ratio_consistency = ratio_consistency
        self.ratio_balance = ratio_balance

        self.u = nn.Parameter(torch.empty(num_examp, 1, dtype=torch.float32))
        self.init_param(mean=mean, std=std)

        self.beginning = True
        self.prevSimilarity = torch.rand((num_examp, encoder_features))
        self.masterVector = torch.rand((num_classes, encoder_features))
        self.sample_labels = sample_labels
        self.bins = []

        for i in range(0, num_classes):
            self.bins.append(np.where(self.sample_labels == i)[0])

    def init_param(self, mean=1e-8, std=1e-9):
        torch.nn.init.normal_(self.u, mean=mean, std=std)

    def forward(self, index, outputs, label, out, flag, epoch):
        if len(outputs) > len(index):
            output, output2 = torch.chunk(outputs, 2)
            out1, out2 = torch.chunk(out, 2)
        else:
            output = outputs
            out1 = out

        eps = 1e-4

        u = self.u[index]

        if flag == 0:
            if self.beginning:
                percent = math.ceil((50 - (50 / total_epochs) * epoch) + 50)
                for i in range(0, len(self.bins)):
                    class_u = self.u.detach()[self.bins[i]]
                    bottomK = int((len(class_u) / 100) * percent)
                    important_indexs = torch.topk(class_u, bottomK, largest=False, dim=0)[1]
                    self.masterVector[i] = torch.mean(
                        self.prevSimilarity[self.bins[i]][important_indexs.view(-1)], dim=0
                    )

            masterVector_norm = self.masterVector.norm(p=2, dim=1, keepdim=True)
            masterVector_normalized = self.masterVector.div(masterVector_norm)
            self.masterVector_transpose = torch.transpose(masterVector_normalized, 0, 1)
            self.beginning = True

        self.prevSimilarity[index] = out1.detach()

        prediction = F.softmax(output, dim=1)

        out_norm = out1.detach().norm(p=2, dim=1, keepdim=True)
        out_normalized = out1.detach().div(out_norm)

        similarity = torch.mm(out_normalized, self.masterVector_transpose)
        similarity = similarity * label
        sim_mask = (similarity > 0.000).type(torch.float32)
        similarity = similarity * sim_mask

        u = u * label

        prediction = torch.clamp((prediction + u.detach()), min=eps, max=1.0)
        loss = torch.mean(-torch.sum((similarity) * torch.log(prediction), dim=1))

        label_one_hot = self.soft_to_hard(output.detach())

        MSE_loss = F.mse_loss((label_one_hot + u), label, reduction="sum") / len(label)
        loss += MSE_loss

        if self.ratio_balance > 0:
            avg_prediction = torch.mean(prediction, dim=0)
            prior_distr = 1.0 / self.num_classes * torch.ones_like(avg_prediction)

            avg_prediction = torch.clamp(avg_prediction, min=eps, max=1.0)

            balance_kl = torch.mean(-(prior_distr * torch.log(avg_prediction)).sum(dim=0))

            loss += self.ratio_balance * balance_kl

        if (len(outputs) > len(index)) and (self.ratio_consistency > 0):
            consistency_loss = self.consistency_loss(output, output2)

            loss += self.ratio_consistency * torch.mean(consistency_loss)

        return loss

    def consistency_loss(self, output1, output2):
        preds1 = F.softmax(output1, dim=1).detach()
        preds2 = F.log_softmax(output2, dim=1)
        loss_kldiv = F.kl_div(preds2, preds1, reduction="none")
        loss_kldiv = torch.sum(loss_kldiv, dim=1)
        return loss_kldiv

    def soft_to_hard(self, x):
        with torch.no_grad():
            return (torch.zeros(len(x), self.num_classes)).cuda().scatter_(1, (x.argmax(dim=1)).view(-1, 1), 1)
