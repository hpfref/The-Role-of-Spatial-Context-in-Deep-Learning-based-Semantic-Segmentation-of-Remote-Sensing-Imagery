import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MajorityClassifier(nn.Module):
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
    
class BottleneckEncoder(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x):
        return self.encoder(x)[-1]  # Just x3
    
class AutoSizedMajorityClassifier(nn.Module):
    def __init__(self, encoder, feature_level, num_classes, target_head_params=10000):
        """
        Args:
            encoder: feature extractor (returns list of features)
            feature_level: which encoder feature to use (e.g., -1 for bottleneck, -2 for previous)
            num_classes: number of output classes
            target_head_params: desired total parameter count for classification head
        """
        super().__init__()
        self.encoder = encoder
        self.feature_level = feature_level
        self.num_classes = num_classes

        # Get input channels from desired encoder output
        with torch.no_grad():
            dummy_input = torch.randn(1, 4, 256, 256)  # adapt input shape if needed
            features = self.encoder(dummy_input)
            selected_feature = features[feature_level]
            in_channels = selected_feature.shape[1]

        # Compute optimal hidden_dim to stay close to target_head_params
        hidden_dim = find_hidden_dim(in_channels, num_classes, target_head_params)
        self.hidden_dim = hidden_dim

        self.channel_adapter = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),  # (B, hidden_dim)
            nn.Linear(hidden_dim, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        with torch.no_grad():
            features = self.encoder(x)
        x = features[self.feature_level]
        x = self.channel_adapter(x)
        return self.classifier(x)
    
def find_hidden_dim(in_channels, num_classes, target_params):
    # Find hidden_dim -> total_params = in_channels*hidden_dim + hidden_dim*num_classes ~= target_params
    low, high = 1, 10000
    while low < high:
        mid = (low + high) // 2
        total = in_channels * mid + mid * num_classes
        if total > target_params:
            high = mid
        else:
            low = mid + 1
    return low