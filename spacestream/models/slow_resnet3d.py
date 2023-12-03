import torch
import torch.nn.functional as F

from .resnet3d import BasicBlock3D, Bottleneck3D, ResNet3D

__all__ = ["SlowNet", "resnet18_slow", "resnet50_slow"]


class SlowNet(ResNet3D):
    def __init__(self, block, layers, **kwargs):
        super(SlowNet, self).__init__(block, layers, **kwargs)
        self.init_params()

    def forward(self, x):
        x, laterals = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = torch.cat([x, laterals[0]], dim=1)
        x = self.layer1(x)

        x = torch.cat([x, laterals[1]], dim=1)
        x = self.layer2(x)

        x = torch.cat([x, laterals[2]], dim=1)
        x = self.layer3(x)

        x = torch.cat([x, laterals[3]], dim=1)
        x = self.layer4(x)

        x = torch.cat([x, laterals[4]], dim=1)
        x = self.avgpool(x)
        x = x.view(-1, x.size(1))

        return x


def resnet18_slow(**kwargs):
    """
    Constructs the slow pathway of the two-stream SlowFast architecture.
    """
    model = SlowNet(BasicBlock3D, [2, 2, 2, 2], slow=True, **kwargs)

    return model


def resnet50_slow(**kwargs):
    """
    Constructs the slow pathway of the two-stream SlowFast architecture.
    """
    model = SlowNet(Bottleneck3D, [3, 4, 6, 3], slow=True, **kwargs)

    return model
