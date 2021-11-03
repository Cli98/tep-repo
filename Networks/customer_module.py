import torch.nn as nn


class stack_conv_layer(nn.Module):
    def __init__(self, filter):
        super(stack_conv_layer, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(filter[0], filter[1], kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(filter[1], filter[2], kernel_size=3, padding=1, bias=True)
        self.batchnorm1 = nn.BatchNorm2d(filter[1])
        self.batchnorm2 = nn.BatchNorm2d(filter[2])

    def forward(self, feature_map):
        out1 = self.conv1(feature_map)
        bn1 = self.batchnorm1(out1)
        res1 = self.activation(bn1)

        out2 = self.conv2(res1)
        bn2 = self.batchnorm2(out2)
        res2 = self.activation(bn2)
        return res2

    """
    This is the official way to stack layer. Let's see if we can use this.
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    """


class upsample_layer(nn.Module):
    def __init__(self, filter):
        super(upsample_layer, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(filter[0], filter[1], kernel_size=3, padding=1, bias=True)
        self.batchnorm1 = nn.BatchNorm2d(filter[1])

    def forward(self, feature_map):
        up = self.upsample(feature_map)
        con = self.conv1(up)
        bn = self.batchnorm1(con)
        output = self.activation(bn)
        return output
