import torch.nn as nn
import torch.nn.init as nn_init

__all__ = ["ResNet3D"]


# =============================
# 3D ResNet
# =============================


class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, head_conv=1):
        super(BasicBlock3D, self).__init__()

        if head_conv == 1:
            self.conv1 = nn.Conv3d(
                inplanes,
                planes,
                kernel_size=(1, 3, 3),
                bias=False,
                padding=(0, 1, 1),
                stride=(1, stride, stride),
            )
            self.bn1 = nn.BatchNorm3d(planes)
        elif head_conv == 3:
            self.conv1 = nn.Conv3d(
                inplanes,
                planes,
                kernel_size=(3, 3, 3),
                bias=False,
                padding=(1, 1, 1),
                stride=(1, stride, stride),
            )
            self.bn1 = nn.BatchNorm3d(planes)
        else:
            raise ValueError("Unsupported head_conv!")

        self.conv2 = nn.Conv3d(
            planes,
            planes,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(1, 1, 1),
            bias=False,
        )
        self.bn2 = nn.BatchNorm3d(planes)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck3D(nn.Module):
    expansion = 4

    def __init__(
        self, inplanes, planes, stride=1, downsample=None, bias=False, head_conv=1
    ):
        super(Bottleneck3D, self).__init__()

        if head_conv == 1:
            self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm3d(planes)
        elif head_conv == 3:
            self.conv1 = nn.Conv3d(
                inplanes, planes, kernel_size=(3, 1, 1), bias=False, padding=(1, 0, 0)
            )
            self.bn1 = nn.BatchNorm3d(planes)
        else:
            raise ValueError("Unsupported head_conv!")

        self.conv2 = nn.Conv3d(
            planes,
            planes,
            kernel_size=(1, 3, 3),
            stride=(1, stride, stride),
            padding=(0, 1, 1),
            bias=bias,
        )
        self.bn2 = nn.BatchNorm3d(planes)

        self.conv3 = nn.Conv3d(
            planes, planes * self.expansion, kernel_size=1, bias=bias
        )
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet3D(nn.Module):
    def __init__(self, block, layers, **kwargs):
        super(ResNet3D, self).__init__()

        self.alpha = kwargs["alpha"]
        self.slow = kwargs["slow"]  # slow->True else fast->False
        self.f2s_mul = kwargs["f2s_mul"]

        # Due to the lateral connections between the fast->slow pathway (where we
        # use time-strided convolution), we need to add more channels.
        self.inplanes = (
            (64 + 64 // self.alpha * self.f2s_mul) if self.slow else 64 // self.alpha
        )
        self.conv1 = nn.Conv3d(
            3,  # RGB channels
            64 // (1 if self.slow else self.alpha),
            kernel_size=(1 if self.slow else 5, 7, 7),
            stride=(1, 2, 2),
            padding=(0 if self.slow else 2, 3, 3),
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(64 // (1 if self.slow else self.alpha))
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(
            kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)
        )
        self.layer1 = self._make_layer(
            block,
            64 // (1 if self.slow else self.alpha),
            layers[0],
            head_conv=1 if self.slow else 3,
        )
        self.layer2 = self._make_layer(
            block,
            128 // (1 if self.slow else self.alpha),
            layers[1],
            stride=2,
            head_conv=1 if self.slow else 3,
        )
        self.layer3 = self._make_layer(
            block,
            256 // (1 if self.slow else self.alpha),
            layers[2],
            stride=2,
            head_conv=3,
        )
        self.layer4 = self._make_layer(
            block,
            512 // (1 if self.slow else self.alpha),
            layers[3],
            stride=2,
            head_conv=3,
        )

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn_init.kaiming_normal_(m.weight)
                if m.bias:
                    nn_init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn_init.constant_(m.weight, 1)
                nn_init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, head_conv=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=(1, 1, 1),
                    stride=(1, stride, stride),
                    bias=False,
                ),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = list()
        layers.append(
            block(self.inplanes, planes, stride, downsample, head_conv=head_conv)
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, head_conv=head_conv))

        if self.slow:
            self.inplanes += block.expansion * planes // self.alpha * self.f2s_mul

        return nn.Sequential(*layers)

    def forward(self, x):
        raise NotImplementedError("Use each pathway network's forward function")
