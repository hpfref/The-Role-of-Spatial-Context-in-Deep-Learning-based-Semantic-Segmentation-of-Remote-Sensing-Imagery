import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class BottleneckMajorityClassifier(nn.Module):
    def __init__(self, encoder, bottleneck_channels, num_classes):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Conv2d(bottleneck_channels, num_classes, kernel_size=1),
            nn.AdaptiveAvgPool2d(1),  # (B, num_classes, 1, 1)
            nn.Flatten(),             # (B, num_classes)
            nn.Softmax(dim=1)         # Probabilities over classes
        )

    def forward(self, x):
        with torch.no_grad():  # Freeze encoder unless fine-tuning
            x = self.encoder(x)
        return self.classifier(x)
    
class BottleneckOnlyEncoder(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x):
        return self.encoder(x)[-1]  # Just x5