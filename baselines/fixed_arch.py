import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class VGGBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.GroupNorm(min(32, out_channels), out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class FixedVGG(nn.Module):
    """A fixed VGG-style architecture for baseline comparison.
    
    This architecture mimics the structure of TSRNetwork, but uses standard
    static convolutional layers instead of adaptive TSR blocks.
    """
    def __init__(self, channels: List[int], in_channels: int = 3, num_classes: int = 10):
        super().__init__()
        self.channels = channels
        
        blocks = []
        current_channels = in_channels
        
        for out_channels in channels:
            blocks.append(VGGBlock(current_channels, out_channels))
            current_channels = out_channels
            
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(current_channels * 16, current_channels * 4),
            nn.ReLU(inplace=True),
            nn.Linear(current_channels * 4, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

def get_baseline_models(num_classes: int = 10) -> dict:
    """Returns a dictionary of static baseline models of varying sizes
    to plot the Pareto frontier.
    """
    return {
        "vgg_tiny": FixedVGG([8, 8, 16], num_classes=num_classes),
        "vgg_small": FixedVGG([16, 16, 32], num_classes=num_classes),
        "vgg_medium": FixedVGG([32, 32, 64], num_classes=num_classes),
        "vgg_large": FixedVGG([64, 64, 128], num_classes=num_classes),
        "vgg_xlarge": FixedVGG([64, 64, 128, 128, 256, 256, 256], num_classes=num_classes),
        "vgg_16_style": FixedVGG([64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512], num_classes=num_classes),
        "vgg_19_style": FixedVGG([64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512], num_classes=num_classes),
    }
