import torch.nn as nn
import torch.nn.functional as F

from .resnet3d import BasicBlock3D, Bottleneck3D, ResNet3D

__all__ = ["FastNet", "resnet18_fast", "resnet50_fast"]


class FastNet(ResNet3D):
    def __init__(
        self,
        block,
        layers,
        fusion_kernel_size=(5, 1, 1),
        fusion_stride=(8, 1, 1),
        fusion_padding=(2, 0, 0),
        **kwargs
    ):
        super(FastNet, self).__init__(block, layers, **kwargs)
        expansion = block.expansion

        self.l_maxpool = nn.Conv3d(
            64 // self.alpha,
            64 // self.alpha * self.f2s_mul,
            kernel_size=fusion_kernel_size,
            stride=fusion_stride,
            bias=False,
            padding=fusion_padding,
        )
        self.l_layer1 = nn.Conv3d(
            expansion * 64 // self.alpha,
            expansion * 64 // self.alpha * self.f2s_mul,
            kernel_size=fusion_kernel_size,
            stride=fusion_stride,
            bias=False,
            padding=fusion_padding,
        )
        self.l_layer2 = nn.Conv3d(
            expansion * 128 // self.alpha,
            expansion * 128 // self.alpha * self.f2s_mul,
            kernel_size=fusion_kernel_size,
            stride=fusion_stride,
            bias=False,
            padding=fusion_padding,
        )
        self.l_layer3 = nn.Conv3d(
            expansion * 256 // self.alpha,
            expansion * 256 // self.alpha * self.f2s_mul,
            kernel_size=fusion_kernel_size,
            stride=fusion_stride,
            bias=False,
            padding=fusion_padding,
        )
        self.l_layer4 = nn.Conv3d(
            expansion * 512 // self.alpha,
            expansion * 512 // self.alpha * self.f2s_mul,
            kernel_size=fusion_kernel_size,
            stride=fusion_stride,
            bias=False,
            padding=fusion_padding,
        )

        self.init_params()

    def forward(self, x):
        laterals = []

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        laterals.append(self.l_maxpool(x))

        x = self.layer1(x)
        laterals.append(self.l_layer1(x))

        x = self.layer2(x)
        laterals.append(self.l_layer2(x))

        x = self.layer3(x)
        laterals.append(self.l_layer3(x))

        x = self.layer4(x)
        laterals.append(self.l_layer4(x))

        x = self.avgpool(x)
        x = x.view(-1, x.size(1))

        return x, laterals


def resnet18_fast(**kwargs):
    """
    Constructs the fast pathway of the two-stream SlowFast architecture.
    """
    model = FastNet(BasicBlock3D, [2, 2, 2, 2], slow=False, **kwargs)

    return model


def resnet50_fast(**kwargs):
    """
    Constructs the fast pathway of the two-stream SlowFast architecture.
    """
    model = FastNet(Bottleneck3D, [3, 4, 6, 3], slow=False, **kwargs)

    return model
