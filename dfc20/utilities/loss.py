import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, weight=None, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth  # To avoid division by zero
        self.weight = weight  # Optional class weights

    def forward(self, logits, targets):
      probs = torch.softmax(logits, dim=1)
      targets_onehot = F.one_hot(targets, num_classes=probs.shape[1]).permute(0, 3, 1, 2).float()
      dims = (0, 2, 3)  # Batch, Height, Width
      intersection = (probs * targets_onehot).sum(dims)
      union = probs.sum(dims) + targets_onehot.sum(dims)
      dice = (2. * intersection + self.smooth) / (union + self.smooth)
      dice_loss = 1 - dice
      if self.weight is not None:
          return (self.weight.to(logits.device) * dice_loss).mean()
      return dice_loss.mean()

class ComboLoss(nn.Module):
    def __init__(self, weight=None, dice_weight=0.5, ce_weight=0.5, dice_smooth=1e-6):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight)
        self.dice_loss = DiceLoss(weight=weight, smooth=dice_smooth)
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, inputs, targets):
        # CrossEntropy expects [B, C, H, W], targets = [B, H, W]
        ce_loss = self.ce(inputs, targets)
        
        # Compute Dice loss via DiceLoss class
        dice_loss = self.dice_loss(inputs, targets)
        
        # Combine both losses
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss