import torch

class SequentialBCEWithLogitsLoss(torch.nn.BCEWithLogitsLoss):
    """
        Custom loss function for sequential binary classification tasks that extends
        PyTorch's BCEWithLogitsLoss and ignores NaN values in the target tensor in the loss calculation.

        Inherits:
            torch.nn.BCEWithLogitsLoss
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input, target):
        """
            Computes the binary cross-entropy loss with logits, ignoring any targets that are NaN.

            Args:
                input (Tensor): Predicted logits.
                target (Tensor): Target tensor of the same shape as input. NaN values are ignored.

            Returns:
                Tensor: The computed scalar loss, averaged over non-NaN elements.
            """
        is_not_nan = ~torch.isnan(target)
        return super().forward(input[is_not_nan], target[is_not_nan])
    
# class SequentialCrossEntropyLoss(torch.nn.Module): #torch.nn.CrossEntropyLoss):
#     def __init__(self, eps = 1e-6, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.eps = eps

#     def forward(self, input, target):
#         is_not_nan = ~torch.isnan(target)
#         print("A")
        
#         #Manual computation, cause CrossEntropyLoss returns nan
#         new_target = target
#         new_target[~is_not_nan] = 0
#         print("B")
        
#         exps = torch.exp(input) * is_not_nan
#         exps_sum = exps.sum(dim=-1)
#         print("exps_sum", exps_sum)

#         exps_div = exps/(exps_sum.unsqueeze(-1)+self.eps)
#         exps_div = exps_div * is_not_nan
#         print("exps_div", exps_div)

#         loss = exps_div*torch.log(exps_div)*new_target
#         print("E")

#         loss = -loss.sum(dim=-1)[is_not_nan.any(dim=-1)]
#         print("F")

#         output = loss.mean()
#         print("output", output)

#         # Commented code cause CrossEntropyLoss returns nan
#         # target[is_nan] = 0
#         # input[is_nan] = -100

#         # all_items_nans = is_nan.all(dim=-1)

#         # new_target = target[~all_items_nans]
#         # new_input = input[~all_items_nans]

#         # output = super().forward(input, target)

#         return output

class SequentialBPR(torch.nn.Module):
    """
        Sequential version of the Bayesian Personalized Ranking (BPR) loss
        for recommendation tasks over sequences that encourages the model to rank positive items higher than negative items
        within the same timestep. 

        Args:
            clamp_max (float, optional): Maximum value for clamping the logit differences to prevent numerical instability.
    """

    def __init__(self, clamp_max=20,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clamp_max = clamp_max

    def forward(self, input, target):
        """
            Computes the Sequential BPR loss, by computing pairwise BPR loss between positive and negative items
            within each timestep. 
            
            Args:
                input (Tensor): Predicted item scores of shape (batch_size, timesteps, num_items).
                            Contains the model's predictions for each item at each timestep.
                target (Tensor): Target relevance tensor of shape (batch_size, timesteps, num_items).
                            Binary relevance scores where 1 indicates positive items, 
                            0 indicates negative items, and NaN values are ignored.
            
            Returns:
                Tensor: Scalar BPR loss averaged over all valid timesteps with both positive
                    and negative items present.
        """

        # Input shape: (batch_size, timesteps, num_items)
        # Output shape: (batch_size, timesteps, num_items)
        is_not_nan = ~torch.isnan(target)

        # Change relevance from 0,1 to -1,1
        new_target = target * 2 - 1
        new_target[~is_not_nan] = 0

        # pair positive and negative items in same timestep
        positive_items = new_target > 0
        negative_items = new_target < 0
        item_pairs = (negative_items.unsqueeze(-1) * positive_items.unsqueeze(-2)).float()
        
        item_per_relevance = input.unsqueeze(-1) - input.unsqueeze(-2)
        item_per_relevance = torch.log(1+torch.exp(torch.clamp(item_per_relevance, max=self.clamp_max)))

        # item_per_relevance has shape (N,T,I,I)
        # item_pairs has shape (N,T,I,I)
        # We want shape (N,T,1). summing on last two dimensions if item_pairs is True
        bpr = torch.einsum('ntij,ntij->nt', item_per_relevance, item_pairs)

        bpr = bpr[is_not_nan.any(dim=-1)].mean()

        return bpr
    
# class SequentialCrossEntropyLoss(torch.nn.Module):
#     def __init__(self, eps = 1e-6, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.eps = eps

#     def forward(self, input, target):
#         # Input shape: (batch_size, timesteps, num_items)
#         # Output shape: (batch_size, timesteps, num_items)
#         is_not_nan = ~torch.isnan(target)

#         new_target = target
#         new_target[~is_not_nan] = 0

#         item_softmax = torch.nn.functional.softmax(input, dim=-1)

#         item_per_relevance = (torch.log(item_softmax)*new_target).sum(-1)
        
#         ce = item_per_relevance[is_not_nan.any(dim=-1)]

#         ce = ce.mean()

#         return ce

class SequentialCrossEntropyLoss(torch.nn.CrossEntropyLoss):
    """
        Custom cross-entropy loss function for sequential classification tasks, to handle sequences where some targets
        might be missing (represented as NaN). It applies the loss only to valid (non-NaN) target positions.

        Inherits:
            torch.nn.CrossEntropyLoss
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input, target):
        """
            Computes the cross-entropy loss for sequential data, filtering out timesteps where all target values are NaN, then
            computes the standard cross-entropy loss on the remaining valid timesteps.
            NaN values within valid timesteps are set to 0 before loss computation.
            
            Args:
                input (Tensor): Predicted logits of shape (batch_size, timesteps, num_items).
                            Contains the model's predictions for each item at each timestep.
                target (Tensor): Target tensor of shape (batch_size, timesteps, num_items).
                            Target probabilities or class indices. 
            
            Returns:
                Tensor: The computed cross-entropy loss, averaged over valid timesteps.
        """
        # Input shape: (batch_size, timesteps, num_items)
        # Output shape: (batch_size, timesteps, num_items)
        is_not_nan = ~torch.isnan(target)

        new_target = target + 0 # Without + 0, new_target will be a view of target and will change target (right?)
        new_target[~is_not_nan] = 0

        new_target = new_target[is_not_nan.any(dim=-1)]
        new_input = input[is_not_nan.any(dim=-1)]

        return super().forward(new_input, new_target)
    

class SequentialGeneralizedBCEWithLogitsLoss(SequentialBCEWithLogitsLoss):
    """
        Generalized Binary Cross-Entropy loss with logits for sequential data
        that applies different treatments to positive and negative samples based on a
        beta parameter. 
        
        Inherits NaN handling capabilities from SequentialBCEWithLogitsLoss.

        Args:
            beta (float): Beta parameter controlling the gamma transformation strength.
                         When beta = 0, only negative samples are used.
                         When beta > 0, positive samples undergo gamma transformation.
            eps (float, optional): Small epsilon value to prevent numerical instability
                                 in the gamma transformation. 
    """
    
    def __init__(self, beta, eps = 1e-6, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.eps = eps
    
    def forward(self, input, target):
        """
            Computes the generalized binary cross-entropy loss with logits.
            
            Args:
                input (Tensor): Predicted logits.
                target (Tensor): Target tensor of the same shape as input. Values > 0.5 are
                            considered positive samples. NaN values are ignored.
            
            Returns:
                Tensor: The computed scalar loss.
        """
        is_positive = target > 0.5
        new_input, new_target = input+0, target+0 #to force copy?
        
        if self.beta == 0:
            new_input,new_target = new_input[~is_positive], new_target[~is_positive]
        else:
            new_input[is_positive] = self.gamma_transformation(new_input[is_positive])

        return super().forward(new_input, new_target)
    
    def gamma_transformation(self, scores):
        """
            Applies gamma transformation to input scores, that adjusts the contribution of positive samples to the loss
            based on the beta parameter,
            
            Args:
                scores (Tensor): Input logits to transform.
            
            Returns:
                Tensor: Transformed logits of the same shape as input.
        """
        return -torch.log((1+torch.exp(-scores))**self.beta-1+self.eps)
