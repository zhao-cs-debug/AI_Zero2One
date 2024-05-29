import torch
import torch.nn as nn

class DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super().__init__()
        self.drop_rate = drop_rate
        self.dense_layer = nn.Sequential(
            nn.BatchNorm2d(num_input_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=num_input_features,
                out_channels=bn_size * growth_rate,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(bn_size * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=bn_size * growth_rate,
                out_channels=growth_rate,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
        )
        self.dropout = nn.Dropout(p=self.drop_rate)

    def forward(self, x):
        y = self.dense_layer(x)
        y = self.dropout(y)
        return torch.cat([x, y], 1)


class DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, growth_rate, bn_size, drop_rate):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(
                DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            )
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class TransitionLayer(nn.Module):
    def __init__(self, num_input_features, num_output_features):
        super().__init__()
        self.transition_layer = nn.Sequential(
            nn.BatchNorm2d(num_input_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=num_input_features,
                out_channels=num_output_features,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.transition_layer(x)


class DenseNet(nn.Module):
    def __init__(
        self,
        init_channels=24,
        growth_rate=4,
        blocks=[6, 6, 6],
        bn_size=4,
        drop_rate=0,
        num_classes=10,
    ):
        super().__init__()
        self.features = nn.ModuleList()
        self.features.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=1,
                    out_channels=init_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True),
            )
        )
        num_features = init_channels
        for i, num_layers in enumerate(blocks):
            block = DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
            )
            self.features.append(block)
            num_features += num_layers * growth_rate

            if i != len(blocks) - 1:
                trans = TransitionLayer(
                    num_input_features=num_features,
                    num_output_features=num_features // 2,
                )
                self.features.append(trans)
                num_features = num_features // 2

        self.norm = nn.BatchNorm2d(num_features)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        for block in self.features:
            x = block(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    img = torch.rand(2, 1, 28, 28)

    model = DenseNet(init_channels=10, growth_rate=4, blocks=[6, 6, 6], num_classes=10)
    print(model)

    logits = model(img)
    print(logits.shape)
