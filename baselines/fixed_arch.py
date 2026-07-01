import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

def _make_gn(channels: int) -> nn.GroupNorm:
    num_groups = next(g for g in range(min(32, channels), 0, -1) if channels % g == 0)
    return nn.GroupNorm(num_groups, channels)


class VGGBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = _make_gn(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ResBasicBlock(nn.Module):
    """Standard CIFAR-style residual unit: identity shortcut within a stage,
    1x1-projection + stride-2 shortcut at stage transitions. Uses GroupNorm
    (not BatchNorm) so the only difference from VGGBlock is the shortcut itself
    — an apples-to-apples test of whether residual connections raise the ceiling.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.gn1 = _make_gn(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.gn2 = _make_gn(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                _make_gn(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        return self.relu(out + identity)


class FixedResNet(nn.Module):
    """CIFAR-style ResNet (He et al. 2016), GroupNorm variant, for the sanity check:
    does adding real residual shortcuts raise the ~90% plain-VGG ceiling under this
    exact recipe (same optimizer/schedule/augmentation/head as TSR and FixedVGG)?

    Default stage_channels=[16,32,64], blocks_per_stage=3 is the classic ResNet-20
    (6n+2 layers, n=3), adapted with a GAP head to match TSRNetwork's current head.
    """

    def __init__(
        self,
        stage_channels: List[int] = None,
        blocks_per_stage: int = 3,
        in_channels: int = 3,
        num_classes: int = 10,
    ):
        super().__init__()
        if stage_channels is None:
            stage_channels = [16, 32, 64]
        self.stage_channels = stage_channels

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stage_channels[0], 3, padding=1, bias=False),
            _make_gn(stage_channels[0]),
            nn.ReLU(inplace=True),
        )

        layers = []
        prev = stage_channels[0]
        for i, ch in enumerate(stage_channels):
            for b in range(blocks_per_stage):
                stride = 2 if (b == 0 and i > 0) else 1
                layers.append(ResBasicBlock(prev, ch, stride=stride))
                prev = ch
        self.layers = nn.Sequential(*layers)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(prev, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layers(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)

class SkipProjection(nn.Module):
    """Permanent (non-gated) residual projection for static baselines.

    Used to reconstruct TSR's *discovered* skip topology in a control net that
    has the same shape but none of the plasticity machinery — no learnable gate,
    no born-alive/newborn-protection dynamics. Identity if channels match,
    otherwise a 1x1 conv; spatially resized via adaptive_avg_pool if needed
    (mirrors GatedConnection's projection logic minus the gate).
    """

    def __init__(self, src_channels: int, dst_channels: int):
        super().__init__()
        if src_channels != dst_channels:
            self.proj: nn.Module = nn.Conv2d(src_channels, dst_channels, kernel_size=1, bias=False)
        else:
            self.proj = nn.Identity()

    def forward(self, src: torch.Tensor, dst_spatial: Tuple[int, int]) -> torch.Tensor:
        h = self.proj(src)
        if h.shape[-2:] != dst_spatial:
            h = F.adaptive_avg_pool2d(h, dst_spatial)
        return h


class FixedVGG(nn.Module):
    """A fixed VGG-style architecture for baseline comparison.

    Mimics TSRNetwork's structure exactly — same block loop, pool_positions,
    optional skip connections, and GAP head — but uses standard static conv
    layers (no gating) so the *only* structural difference from TSR is that
    this network never grows: same final shape, trained from scratch.

    Args:
        channels: Per-block output channel counts (mirrors TSRNetwork's discovered
            or seed channel list — index i is block i's out_channels).
        pool_positions: Indices of blocks after which to 2x2 maxpool. Defaults to
            TSRNetwork's default (every 2nd block) so vanilla baselines (vgg_tiny
            etc.) match TSR's downsampling schedule too.
        skip_connections: Optional list of (src_block_idx, dst_block_idx) pairs —
            TSR's discovered residual topology, reconstructed here as permanent
            (non-gated) SkipProjection edges. None = plain VGG, no skips.
    """

    def __init__(
        self,
        channels: List[int],
        in_channels: int = 3,
        num_classes: int = 10,
        classifier_hidden: Optional[int] = None,
        pool_positions: Optional[List[int]] = None,
        skip_connections: Optional[List[Tuple[int, int]]] = None,
    ):
        super().__init__()
        self.channels = channels

        blocks = []
        current_channels = in_channels
        for out_channels in channels:
            blocks.append(VGGBlock(current_channels, out_channels))
            current_channels = out_channels
        self.blocks = nn.ModuleList(blocks)

        if pool_positions is None:
            pool_positions = [i for i in range(1, len(channels), 2)]
            if len(channels) >= 2 and not pool_positions:
                pool_positions = [len(channels) - 1]
        self.pool_positions = set(pool_positions)

        self.skip_connections = nn.ModuleDict()
        for src_idx, dst_idx in (skip_connections or []):
            key = f"{src_idx}__{dst_idx}"
            self.skip_connections[key] = SkipProjection(channels[src_idx], channels[dst_idx])

        # GAP head — matches TSRNetwork so the classifier bridge is identical
        # (no ×16 blowup, no confound from differing pooling schemes).
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        if classifier_hidden is None:
            classifier_hidden = max(current_channels * 2, 32)  # matches TSRNetwork's default
        self.classifier = nn.Sequential(
            nn.Linear(current_channels, classifier_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(classifier_hidden, num_classes)
        )

    def forward(self, x):
        block_outputs = []
        for i, block in enumerate(self.blocks):
            h = block(x)

            for key, proj in self.skip_connections.items():
                src_idx, dst_idx = (int(v) for v in key.split("__"))
                if dst_idx == i:
                    h = h + proj(block_outputs[src_idx], h.shape[-2:])

            block_outputs.append(h)
            x = F.max_pool2d(h, 2) if i in self.pool_positions else h

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
        "vgg_16_style": FixedVGG(VGG_CONFIGS["vgg16"], num_classes=num_classes),
        "vgg_19_style": FixedVGG(VGG_CONFIGS["vgg19"], num_classes=num_classes),
    }


# Standard VGG channel configurations adapted for CIFAR-10.
# These match the original VGG paper channel widths but use AdaptiveAvgPool
# to handle CIFAR-10's 32x32 input (no need for the full 224x224 spatial stack).
VGG_CONFIGS: dict = {
    "vgg16": [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512],
    "vgg19": [64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512],
}

