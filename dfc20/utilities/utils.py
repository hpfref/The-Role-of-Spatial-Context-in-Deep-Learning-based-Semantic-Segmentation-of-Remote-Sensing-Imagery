import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Sampler
from collections import defaultdict, Counter
import random
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

class WarmUpLR:
    def __init__(self, optimizer, warmup_epochs, base_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.current_epoch = 0

    def step(self):
        if self.current_epoch < self.warmup_epochs:
            lr = self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        self.current_epoch += 1
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
    
class ComboLossFocal(nn.Module):
    def __init__(self, weight=None, dice_weight=0.5, focal_weight=0.5, dice_smooth=1e-6,
                 alpha=0.25, gamma=2.0):
        super().__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma, weight=weight, reduction='mean')
        self.dice_loss = DiceLoss(weight=weight, smooth=dice_smooth)
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, inputs, targets):
        focal_loss = self.focal(inputs, targets)
        dice_loss = self.dice_loss(inputs, targets)
        return self.focal_weight * focal_loss + self.dice_weight * dice_loss
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, weight=None, reduction='mean'):
        """
        Focal Loss for multi-class segmentation.
        Args:
            alpha (float): Class balancing factor.
            gamma (float): Focusing parameter for hard examples.
            weight (Tensor, optional): Class weights [num_classes].
            reduction (str): 'mean', 'sum', or 'none'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Tensor of shape [B, C, H, W] — raw logits
            targets: Tensor of shape [B, H, W] — class indices
        """
        # Reshape for computation
        B, C, H, W = inputs.shape
        inputs = inputs.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]
        targets = targets.view(-1)  # [B*H*W]

        # Cross-entropy per pixel
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')  # [B*H*W]

        # Prob of correct class
        pt = torch.exp(-ce_loss)

        # Focal loss formula
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss.view(B, H, W)  # reshape to [B, H, W] if needed

class LovaszSoftmaxLoss(nn.Module):
    def __init__(self):
        super(LovaszSoftmaxLoss, self).__init__()

    def forward(self, logits, labels):
        """
        logits: [B, C, H, W] - unnormalized scores
        labels: [B, H, W] - ground truth labels (0 to C-1)
        """
        probs = torch.softmax(logits, dim=1)  # [B, C, H, W]

        # reshape to [P, C] and [P]
        B, C, H, W = probs.shape
        probs_flat = probs.permute(0, 2, 3, 1).reshape(-1, C)
        labels_flat = labels.view(-1)

        return self.lovasz_softmax_flat(probs_flat, labels_flat, C)

    def lovasz_softmax_flat(self, probs, labels, classes):
        losses = []
        for c in range(classes):
            fg = (labels == c).float()
            if fg.sum() == 0:
                continue
            errors = (fg - probs[:, c]).abs()
            errors_sorted, perm = torch.sort(errors, descending=True)
            fg_sorted = fg[perm]
            grad = self.lovasz_grad(fg_sorted)
            losses.append(torch.dot(errors_sorted, grad))
        return torch.mean(torch.stack(losses)) if losses else torch.tensor(0., device=probs.device)

    def lovasz_grad(self, gt_sorted):
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.cumsum(0)
        union = gts + (1 - gt_sorted).cumsum(0)
        jaccard = 1. - intersection / union
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
        return jaccard

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
    

class BatchBalancingSampler(Sampler):
    def __init__(self, majority_classes, batch_size, min_samples_per_class=1, seed=42):
        """
        Args:
            majority_classes: List[int], majority class label for each sample
            batch_size: int, total number of samples per batch
            min_samples_per_class: int, minimum number of samples per class in each batch (if available)
            seed: int, random seed for reproducibility
        """
        self.majority_classes = majority_classes
        self.batch_size = batch_size
        self.min_samples_per_class = min_samples_per_class
        self.seed = seed
        self.num_samples = len(majority_classes)

        self.class_to_indices = defaultdict(list)
        for idx, cls in enumerate(majority_classes):
            self.class_to_indices[cls].append(idx)

        self.all_classes = list(self.class_to_indices.keys())
        self.all_indices = list(range(len(majority_classes)))
        random.seed(seed)

    def __iter__(self):
        used_indices = set()
        batches = []

        # Build class iterators
        class_iters = {}
        for cls, indices in self.class_to_indices.items():
            shuffled = indices.copy()
            random.shuffle(shuffled)
            class_iters[cls] = iter(shuffled)

        while len(used_indices) < self.num_samples:
            batch = []

            # Sample min_samples_per_class from as many classes as possible
            random.shuffle(self.all_classes)
            for cls in self.all_classes:
                cls_batch = 0
                while cls_batch < self.min_samples_per_class and len(batch) < self.batch_size:
                    try:
                        idx = next(class_iters[cls])
                        if idx not in used_indices:
                            batch.append(idx)
                            used_indices.add(idx)
                            cls_batch += 1
                    except StopIteration:
                        break

            # Fill the rest of the batch randomly
            remaining = list(set(self.all_indices) - used_indices)
            random.shuffle(remaining)
            for idx in remaining:
                if len(batch) >= self.batch_size:
                    break
                batch.append(idx)
                used_indices.add(idx)

            batches.append(batch)

        return iter([i for batch in batches for i in batch])

    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size
    
def oversample_indices(majority_classes, target_count_per_class=None, max_oversample_factor=10):
    class_counts = Counter(majority_classes)
    all_classes = list(class_counts.keys())

    # Default: balance to the max class count
    max_count = max(class_counts.values())
    if target_count_per_class is None:
        target_count_per_class = {cls: min(max_count, class_counts[cls] * max_oversample_factor)
                                  for cls in all_classes}

    new_indices = []
    class_to_indices = defaultdict(list)
    for idx, cls in enumerate(majority_classes):
        class_to_indices[cls].append(idx)

    for cls in all_classes:
        indices = class_to_indices[cls]
        repeat_factor = target_count_per_class[cls] // len(indices)
        remainder = target_count_per_class[cls] % len(indices)
        new_indices.extend(indices * repeat_factor + random.sample(indices, remainder))

    return new_indices

#### visuals

def predict_patch(model, image):
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # No gradients needed during inference
        # Convert to tensor if it's not already
        if isinstance(image, np.ndarray):
            image = torch.tensor(image, dtype=torch.float32)
        # Check the shape and ensure it's in the correct format [batch_size, channels, height, width]
        if image.ndim == 3:  # If it's a single image (HxWxC)
            image = image.unsqueeze(0)  # Add batch dimension
        # Forward pass through the model
        output = model(image)  # Assuming the model's output shape is [batch_size, num_classes, H, W]
        # Apply argmax to get the predicted class for each pixel
        pred = torch.argmax(output, dim=1).squeeze(0)  # Get the class with max score for each pixel
        # Convert to numpy array for visualization
        pred = pred.cpu().numpy()

    return pred

def compute_confusion_matrix(y_true, y_pred, num_classes):
    """
    Compute confusion matrix normalized globally (by total number of valid pixels),
    excluding class 0 (unclassified).
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    # Ignore unclassified class (0)
    valid = y_true != 0
    y_true = y_true[valid]
    y_pred = y_pred[valid]

    labels = np.arange(1, num_classes)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    total = cm.sum()
    cm_percent = (cm.astype(np.float32) / total) * 100
    return cm_percent


def plot_confusion_matrix(cm, class_info):
    class_ids = [cid for cid in sorted(class_info.keys()) if cid != 0]
    class_names = [class_info[cid][0] for cid in class_ids]
    class_colors_rgb = [class_info[cid][1] for cid in class_ids]
    class_colors = [(r / 255, g / 255, b / 255) for (r, g, b) in class_colors_rgb]

    cm_percent = 100 * cm / cm.sum()

    num_classes = len(class_names)
    fig_height = max(6, 9)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    sns.heatmap(cm_percent, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=False, yticklabels=False,
                square=True, linewidths=0.5,
                cbar_kws={"label": "% of total pixels"},
                ax=ax)

    # Fix colorbar height to match heatmap
    cbar = ax.collections[0].colorbar
    heatmap_pos = ax.get_position()
    cbar_ax = cbar.ax
    cbar_pos = cbar_ax.get_position()
    cbar_ax.set_position([cbar_pos.x0, heatmap_pos.y0, cbar_pos.width, heatmap_pos.height])

    ax.set_xlim(-1.5, num_classes)
    ax.set_ylim(num_classes, 0)

    # Draw colored rectangles + class names on left
    for ytick_pos, (name, color) in enumerate(zip(class_names, class_colors)):
        ax.add_patch(mpatches.Rectangle(
            (-1.3, ytick_pos + 0.25), 0.3, 0.3,
            color=color, transform=ax.transData, clip_on=False))
        ax.text(-1.5, ytick_pos + 0.4, name,
                ha='right', va='center', fontsize=10)

    # Draw bottom color rectangles for predicted classes
    for xtick_pos, color in enumerate(class_colors):
        ax.add_patch(mpatches.Rectangle(
            (xtick_pos + 0.35, num_classes + 0.15), 0.3, 0.3,
            color=color, transform=ax.transData, clip_on=False))

    # Label: Predicted Class
    ax.set_xlabel("Predicted Class", fontsize=12, fontweight='bold', labelpad=30)

    # Label: True Class (positioned close to matrix)
    ax_pos = ax.get_position()
    fig.text(ax_pos.x1 + 0.09, ax_pos.y0 + ax_pos.height / 2,
             "True Class", va='center', ha='left',
             fontsize=12, fontweight='bold', rotation=270)

    ax.set_title("Confusion Matrix (% of total pixels)", fontsize=14, pad=20)
    ax.tick_params(left=False, bottom=False)
    plt.tight_layout()
    plt.show()