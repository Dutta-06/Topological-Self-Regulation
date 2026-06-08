"""
TSRGroupNorm: Width-invariant normalization for TSR layers.

Standard BatchNorm breaks when layer width changes because its running
statistics are fixed-size. GroupNorm divides channels into groups and
normalizes within each group — the number of groups can be adjusted
when channels are added/removed, and no running statistics are stored.

This wrapper automatically handles group count adjustment during
structural changes.
"""

import torch
import torch.nn as nn


class TSRGroupNorm(nn.Module):
    """GroupNorm wrapper that adapts to channel count changes.

    Uses a target group size rather than a fixed number of groups.
    When channels are added/removed, the number of groups adjusts to
    maintain approximately the target group size.

    Args:
        num_channels: Initial number of channels to normalize.
        target_group_size: Target channels per group. Default: 8.
            Actual group size may differ slightly to ensure num_channels
            is divisible by num_groups.
        eps: Epsilon for numerical stability. Default: 1e-5.
        affine: If True, learnable affine parameters. Default: True.
    """

    def __init__(
        self,
        num_channels: int,
        target_group_size: int = 8,
        eps: float = 1e-5,
        affine: bool = True,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.target_group_size = target_group_size
        self.eps = eps
        self.affine = affine

        num_groups = self._compute_num_groups(num_channels)
        self.norm = nn.GroupNorm(
            num_groups=num_groups,
            num_channels=num_channels,
            eps=eps,
            affine=affine,
        )

    def _compute_num_groups(self, num_channels: int) -> int:
        """Compute number of groups to approximate target group size.

        Finds the largest divisor of num_channels that is ≤ the ideal
        number of groups (num_channels / target_group_size), with a
        minimum of 1 group.
        """
        if num_channels <= 0:
            return 1

        ideal_groups = max(1, num_channels // self.target_group_size)

        # Find the largest divisor of num_channels that is ≤ ideal_groups
        best = 1
        for g in range(1, ideal_groups + 1):
            if num_channels % g == 0:
                best = g
        return best

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through GroupNorm.

        Args:
            x: Input tensor. Shape (batch, channels, ...) for conv,
               or (batch, channels) for linear (unsqueezed internally).

        Returns:
            Normalized tensor of same shape.
        """
        if x.dim() == 2:
            # Linear layer output: (batch, features) → (batch, features, 1) → norm → squeeze
            x = x.unsqueeze(-1)
            x = self.norm(x)
            return x.squeeze(-1)
        return self.norm(x)

    def resize(self, new_num_channels: int) -> None:
        """Rebuild the GroupNorm for a new channel count.

        Preserves affine parameters for surviving channels where possible.

        Args:
            new_num_channels: New number of channels.
        """
        old_channels = self.num_channels
        new_groups = self._compute_num_groups(new_num_channels)

        old_weight = self.norm.weight.data if self.affine else None
        old_bias = self.norm.bias.data if self.affine else None
        device = old_weight.device if self.affine else None

        self.norm = nn.GroupNorm(
            num_groups=new_groups,
            num_channels=new_num_channels,
            eps=self.eps,
            affine=self.affine,
        )
        
        if device is not None:
            self.norm = self.norm.to(device)

        # Copy surviving affine params
        if self.affine and old_weight is not None:
            copy_n = min(old_channels, new_num_channels)
            self.norm.weight.data[:copy_n] = old_weight[:copy_n]
            self.norm.bias.data[:copy_n] = old_bias[:copy_n]

        self.num_channels = new_num_channels

    def extra_repr(self) -> str:
        return (
            f"num_channels={self.num_channels}, "
            f"groups={self.norm.num_groups}, "
            f"target_group_size={self.target_group_size}, "
            f"affine={self.affine}"
        )
