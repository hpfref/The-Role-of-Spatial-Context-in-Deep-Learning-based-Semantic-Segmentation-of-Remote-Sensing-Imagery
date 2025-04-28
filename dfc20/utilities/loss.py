import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, weight=None, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth  # To avoid division by zero
        self.weight = weight  # Optional class weights

    def forward(self, logits, targets):
        # Apply softmax or sigmoid depending on your output
        # Assuming the output is raw logits, we apply a softmax function here.
        probs = torch.softmax(logits, dim=1)  # For multi-class

        # Flatten the prediction and the target to make sure they are the same size
        probs = probs.view(-1, probs.shape[1])  # Shape [B * H * W, C]
        targets = targets.view(-1)  # Shape [B * H * W]

        # One-hot encode targets
        targets_onehot = torch.zeros_like(probs)
        targets_onehot.scatter_(1, targets.unsqueeze(1), 1)  # One-hot encoding

        # Calculate intersection and union
        intersection = (probs * targets_onehot).sum(dim=0)
        union = probs.sum(dim=0) + targets_onehot.sum(dim=0)

        # If weights are provided, apply them
        if self.weight is not None:
            # Apply the weights per class
            dice_loss = 1 - (2. * intersection + self.smooth) / (union + self.smooth)
            weighted_dice_loss = (self.weight * dice_loss).mean()  # Weighted mean of Dice loss
            return weighted_dice_loss
        else:
            # Standard Dice loss (no weighting)
            dice_loss = 1 - (2. * intersection + self.smooth) / (union + self.smooth)
            return dice_loss.mean()  # Mean across classes

class ComboLoss(nn.Module):
    def __init__(self, weight=None, dice_weight=0.5, ce_weight=0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight)
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, inputs, targets):
        # CrossEntropy expects [B, C, H, W], targets = [B, H, W]
        ce_loss = self.ce(inputs, targets)

        # Dice Loss: convert targets to one-hot
        num_classes = inputs.shape[1]
        targets_onehot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

        inputs_soft = F.softmax(inputs, dim=1)
        dims = (0, 2, 3)

        intersection = (inputs_soft * targets_onehot).sum(dims)
        union = inputs_soft.sum(dims) + targets_onehot.sum(dims)
        dice_loss = 1 - ((2. * intersection + 1e-5) / (union + 1e-5)).mean()

        return self.ce_weight * ce_loss + self.dice_weight * dice_loss