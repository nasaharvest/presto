import math
from typing import Optional, Union

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from .utils import device


def param_groups_weight_decay(model: nn.Module, weight_decay=1e-5, no_weight_decay_list=()):
    # https://github.com/huggingface/pytorch-image-models/blob/main/timm/optim/optim_factory.py
    no_weight_decay_list = set(no_weight_decay_list)
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.ndim <= 1 or name.endswith(".bias") or name in no_weight_decay_list:
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]


def adjust_learning_rate(optimizer, epoch, warmup_epochs, total_epochs, max_lr, min_lr):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < warmup_epochs:
        lr = max_lr * epoch / warmup_epochs
    else:
        lr = min_lr + (max_lr - min_lr) * 0.5 * (
            1.0 + math.cos(math.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs))
        )
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            # This is only used during finetuning, and not yet
            # implemented in our codebase
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


class LossWrapper(nn.Module):
    def __init__(self, loss: nn.Module):
        super().__init__()
        self.loss = loss

    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        assert len(pred) == len(true)
        if len(pred) == 0:
            # len(pred) == 0 -> no inputs are masked, so no
            # inputs are passed to the loss
            return torch.tensor(0).float().to(device)
        return self.loss(pred, true)


class BCELossWithSmoothing(nn.BCELoss):
    def __init__(
        self, smoothing: float = 0.1, weight=None, size_average=None, reduce=None, reduction="mean"
    ):
        super().__init__(
            weight=weight, size_average=size_average, reduce=reduce, reduction=reduction
        )
        assert smoothing < 1
        assert smoothing >= 0
        self.smoothing = smoothing

    def forward(self, input, target):
        return super().forward(
            input, torch.clamp(target, min=self.smoothing, max=(1 - self.smoothing))
        )


class Seq2Seq(nn.Module):
    encoder: nn.Module
    decoder: nn.Module

    def forward(
        self,
        x: torch.Tensor,
        dynamic_world: torch.Tensor,
        latlons: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        month: Union[torch.Tensor, int] = 0,
    ):
        raise NotImplementedError


class FinetuningHead(nn.Module):
    def __init__(self, hidden_size: int, num_outputs: int, regression: bool) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.num_outputs = num_outputs
        self.regression = regression
        self.linear = nn.Linear(hidden_size, num_outputs)

    def forward(self, x: torch.Tensor):
        x = self.linear(x)
        if (not self.regression) & (self.num_outputs == 1):
            x = torch.sigmoid(x)
        return x


class FineTuningModel(nn.Module):
    encoder: nn.Module

    def forward(
        self,
        x: torch.Tensor,
        dynamic_world: torch.Tensor,
        latlons: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        month: Union[torch.Tensor, int] = 0,
    ) -> torch.Tensor:
        raise NotImplementedError


class Mosaiks1d(nn.Module):
    def __init__(
        self, in_channels: int, k: int, kernel_size: int, patches: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels, out_channels=k, kernel_size=kernel_size, bias=False
        )
        if patches is not None:
            assert patches.shape == self.conv.weight.shape
            self.conv.weight = nn.Parameter(patches)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.conv(rearrange(x, "batch timestep channel -> batch channel timestep"))
        return F.relu(x).mean(dim=-1)

    def encoder(self, x, dynamic_world, mask, latlons, month) -> torch.Tensor:
        # ensures the model works seamlessly with the eval tasks
        return self(x)
