import torch
import torch.nn as nn


def conv3x3(in_channels, out_channels, stride=1):
    """
    3x3 卷积，padding=1 保持特征图尺寸
    """
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
    )


def conv1x1(in_channels, out_channels, stride=1):
    """
    1x1 卷积，用于 shortcut 维度匹配
    """
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=1, stride=stride, bias=False
    )


class BasicBlock(nn.Module):
    """
    ResNet-18 / ResNet-34 使用的基础残差块

    结构：
        x -> Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN
        shortcut(x) 加到主分支
        -> ReLU
    """

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(out_channels, out_channels, stride=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None

        # 当尺寸或通道数不一致时，需要用 1x1 卷积调整 shortcut
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.downsample = nn.Sequential(
                conv1x1(in_channels, out_channels * self.expansion, stride),
                nn.BatchNorm2d(out_channels * self.expansion),
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    手写 ResNet 主体

    ResNet-18:
        layers = [2, 2, 2, 2]

    ResNet-34:
        layers = [3, 4, 6, 3]
    """

    def __init__(self, block, layers, num_classes=1000, in_channels=3):
        super().__init__()

        self.in_channels = 64

        # 输入一般是 [B, 3, 224, 224]
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._init_weights()

    def _make_layer(self, block, out_channels, num_blocks, stride):
        layers = []

        # 每个 stage 的第一个 block 可能需要降采样
        layers.append(
            block(
                in_channels=self.in_channels, out_channels=out_channels, stride=stride
            )
        )

        self.in_channels = out_channels * block.expansion

        # 后续 block 不改变特征图尺寸
        for _ in range(1, num_blocks):
            layers.append(
                block(in_channels=self.in_channels, out_channels=out_channels, stride=1)
            )

        return nn.Sequential(*layers)

    def _init_weights(self):
        """
        Kaiming 初始化，适合 ReLU 网络
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # residual stages
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # classifier
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def resnet18(num_classes=1000, in_channels=3):
    return ResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        num_classes=num_classes,
        in_channels=in_channels,
    )


def resnet34(num_classes=1000, in_channels=3):
    return ResNet(
        block=BasicBlock,
        layers=[3, 4, 6, 3],
        num_classes=num_classes,
        in_channels=in_channels,
    )


if __name__ == "__main__":
    model = resnet18(num_classes=101)

    x = torch.randn(4, 3, 224, 224)
    y = model(x)

    print(model)
    print("input shape:", x.shape)
    print("output shape:", y.shape)
