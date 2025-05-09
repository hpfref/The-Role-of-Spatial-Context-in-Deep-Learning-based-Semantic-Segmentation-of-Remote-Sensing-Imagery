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

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0.0, mode='min'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.mode = mode

    def __call__(self, metric_value):
        score = -metric_value if self.mode == 'min' else metric_value

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

class PolyLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, max_epochs, power=0.9, last_epoch=-1):
        self.max_epochs = max_epochs
        self.power = power
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            base_lr * (1 - self.last_epoch / self.max_epochs) ** self.power
            for base_lr in self.base_lrs
        ]