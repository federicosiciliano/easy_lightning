import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

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

