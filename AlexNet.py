import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()

        # 特征提取部分：卷积层 + ReLU + 最大池化
        self.features = nn.Sequential(
            # 输入: [B, 3, 224, 224]
            # 输出: [B, 64, 55, 55]
            nn.Conv2d(
                in_channels=3, out_channels=64, kernel_size=11, stride=4, padding=2
            ),
            nn.ReLU(inplace=True),
            # 输出: [B, 64, 27, 27]
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 输出: [B, 192, 27, 27]
            nn.Conv2d(
                in_channels=64, out_channels=192, kernel_size=5, stride=1, padding=2
            ),
            nn.ReLU(inplace=True),
            # 输出: [B, 192, 13, 13]
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 输出: [B, 384, 13, 13]
            nn.Conv2d(
                in_channels=192, out_channels=384, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(inplace=True),
            # 输出: [B, 256, 13, 13]
            nn.Conv2d(
                in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(inplace=True),
            # 输出: [B, 256, 13, 13]
            nn.Conv2d(
                in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(inplace=True),
            # 输出: [B, 256, 6, 6]
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # 自适应池化，保证输出尺寸为 [B, 256, 6, 6]
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        # 分类器部分：全连接层
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            # 256 * 6 * 6 = 9216
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):  # [B, 3, 224, 224]
        x = self.features(x)

        x = self.avgpool(x)

        # 展平: [B, 256, 6, 6] -> [B, 9216]
        x = torch.flatten(x, start_dim=1)

        x = self.classifier(x)

        return x


if __name__ == "__main__":
    model = AlexNet(num_classes=1000)

    x = torch.randn(2, 3, 224, 224)

    y = model(x)

    print("输出形状:", y.shape)
