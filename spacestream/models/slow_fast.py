import torch
import torch.nn as nn

from .fast_resnet3d import *
from .slow_resnet3d import *

__all__ = ["SlowFastNetwork", "slow_fast_resnet18", "slow_fast_resnet50"]


class SlowFastNetwork(nn.Module):
    """
    Construction of the SlowFast architecture of Feichtenhofer et al., 2019.
    https://arxiv.org/pdf/1812.03982.pdf

    By default, we do not include the final fully-connected layer for classification
    The fully-connected layer is incorporated in the loss function (see
    `cross_entropy_loss.py')

    Arguments:
        slow_pathway : (Callable) function that returns the slow pathway network.
                       Default: resnet50_slow
        fast_pathway : (Callable) function that returns the fast pathway network.
                       Default: resnet50_fast
        alpha        : (int) The fast pathway has alpha times less channels that the
                       slow pathway. Default: 8
        f2s_mul      : (int) This is used to determine the number of output channels
                       when performing the time-strided convolutions for the lateral
                       connections between the fast and slow pathways. Default: 2
        in_channels  : (int) number of input channels of each video frame. Default: 3
                       Default: 400 (400 classes in Kinetics-400)
    """

    def __init__(
        self,
        slow_pathway=resnet50_slow,
        fast_pathway=resnet50_fast,
        alpha=8,
        f2s_mul=2,
        **fast_to_slow_fusion_kwargs,
    ):
        super(SlowFastNetwork, self).__init__()

        self.alpha = alpha
        self.f2s_mul = f2s_mul

        self.fast_to_slow_fusion_kwargs = fast_to_slow_fusion_kwargs
        self.slow_pathway = slow_pathway
        self.fast_pathway = fast_pathway
        self.slow, self.fast = self._create_slow_fast_pathways()

    def _create_slow_fast_pathways(self):
        # Slow pathway
        slow = self.slow_pathway(
            alpha=self.alpha,
            f2s_mul=self.f2s_mul,
        )

        # Fast pathway
        fast = self.fast_pathway(
            alpha=self.alpha,
            f2s_mul=self.f2s_mul,
            **self.fast_to_slow_fusion_kwargs,
        )

        return slow, fast

    def forward(self, x):
        # x is a length-2 list of tensors
        x_slow, x_fast = x

        x_fast, laterals = self.fast(x_fast)
        x_slow = self.slow([x_slow, laterals])

        x = torch.cat([x_slow, x_fast], dim=1)

        return x


def slow_fast_resnet18(**kwargs):
    """
    Creates the SlowFast architecture using the ResNet-50 backbone.
    """
    alpha = kwargs["alpha"]
    f2s_mul = kwargs["f2s_mul"]
    kwargs.pop("alpha")
    kwargs.pop("f2s_mul")

    model = SlowFastNetwork(
        slow_pathway=resnet18_slow,
        fast_pathway=resnet18_fast,
        alpha=alpha,
        f2s_mul=f2s_mul,
        **kwargs,
    )

    return model


def slow_fast_resnet50(**kwargs):
    """
    Creates the SlowFast architecture using the ResNet-50 backbone.
    """
    alpha = kwargs["alpha"]
    f2s_mul = kwargs["f2s_mul"]
    kwargs.pop("alpha")
    kwargs.pop("f2s_mul")

    model = SlowFastNetwork(
        slow_pathway=resnet50_slow,
        fast_pathway=resnet50_fast,
        alpha=alpha,
        f2s_mul=f2s_mul,
        **kwargs,
    )

    return model
