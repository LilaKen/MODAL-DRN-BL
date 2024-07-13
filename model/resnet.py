import torch.nn as nn
import math

BatchNorm = nn.BatchNorm2d


def conv3x3(in_planes, out_planes, stride=(1, 1), padding=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(3, 3), stride=stride, padding=padding, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, residual=True):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm(planes)
        self.downsample = downsample
        self.stride = stride
        self.residual = residual

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        if self.residual:
            out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=(1, 1), downsample=None, residual=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=(1, 1), bias=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=(3, 3), stride=stride, bias=False)
        self.bn2 = BatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=(1, 1), bias=False)
        self.bn3 = BatchNorm(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.residual = residual

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


class RES(nn.Module):

    def __init__(self, block, layers, num_classes=1, channels=(16, 32, 64, 128, 256, 512, 512, 512), out_map=False, out_middle=False, pool_size=28, arch='D'):
        super(RES, self).__init__()
        self.inplanes = channels[0]
        self.out_map = out_map
        self.out_dim = channels[-1]
        self.out_middle = out_middle
        self.arch = arch

        if arch == 'C':
            self.conv1 = nn.Conv2d(1, channels[0], kernel_size=(7, 7), stride=(1, 1), padding=3, bias=False)
            self.bn1 = BatchNorm(channels[0])
            self.relu = nn.ReLU(inplace=True)

            self.layer1 = self._make_layer(BasicBlock, channels[0], layers[0], stride=1)
            self.layer2 = self._make_layer(BasicBlock, channels[1], layers[1], stride=2)
        elif arch == 'D':
            self.layer0 = nn.Sequential(
                nn.Conv2d(1, channels[0], kernel_size=(7, 7), stride=(1, 1), padding=3, bias=False),
                BatchNorm(channels[0]),
                nn.ReLU(inplace=True)
            )

            self.layer1 = self._make_conv_layers(channels[0], layers[0], stride=1)
            self.layer2 = self._make_conv_layers(channels[1], layers[1], stride=2)

        self.layer3 = self._make_layer(block, channels[2], layers[2], stride=1)
        self.layer4 = self._make_layer(block, channels[3], layers[3], stride=1)
        self.layer5 = self._make_layer(block, channels[4], layers[4], new_level=False)
        self.layer6 = None if layers[5] == 0 else self._make_layer(block, channels[5], layers[5], new_level=False)

        if arch == 'C':
            self.layer7 = None if layers[6] == 0 else self._make_layer(BasicBlock, channels[6], layers[6], new_level=False, residual=False)
            self.layer8 = None if layers[7] == 0 else self._make_layer(BasicBlock, channels[7], layers[7], new_level=False, residual=False)
        elif arch == 'D':
            self.layer7 = None if layers[6] == 0 else self._make_conv_layers(channels[6], layers[6])
            self.layer8 = None if layers[7] == 0 else self._make_conv_layers(channels[7], layers[7])

        if num_classes > 0:
            self.avgpool = nn.AdaptiveAvgPool2d(pool_size)
            self.fc = nn.Conv2d(self.out_dim, num_classes, kernel_size=(11, 11), stride=(2, 2), padding=1, bias=True)
            self.flat = nn.Flatten()
            self.fc2 = nn.Linear(in_features=100, out_features=10, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=(1, 1), new_level=True, residual=True):
        """Create a layer of blocks"""
        downsample = None
        if stride != (1, 1) or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=(1, 1), stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample, residual=residual)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, residual=residual))

        return nn.Sequential(*layers)

    def _make_conv_layers(self, channels, convs, stride=(1, 1)):
        """Create a layer of convolution layers"""
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(self.inplanes, channels, kernel_size=(3, 3), stride=stride if i == 0 else 1, bias=False),
                BatchNorm(channels),
                nn.ReLU(inplace=True)])
            self.inplanes = channels
        return nn.Sequential(*modules)

    def forward(self, x):
        y = list()

        if self.arch == 'C':
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        elif self.arch == 'D':
            x = self.layer0(x)

        x = self.layer1(x)
        y.append(x)
        x = self.layer2(x)
        y.append(x)

        x = self.layer3(x)
        y.append(x)

        x = self.layer4(x)
        y.append(x)

        x = self.layer5(x)
        y.append(x)

        if self.layer6 is not None:
            x = self.layer6(x)
            y.append(x)

        if self.layer7 is not None:
            x = self.layer7(x)
            y.append(x)

        if self.layer8 is not None:
            x = self.layer8(x)
            y.append(x)

        if self.out_map:
            x = self.fc(x)
        else:
            x = self.avgpool(x)
            x = self.fc(x)
            x = self.flat(x)
            x = self.fc2(x)
            x = x.view(x.size(0), -1)

        if self.out_middle:
            return x, y
        else:
            return x


class RES_A(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        self.inplanes = 64
        super(RES_A, self).__init__()
        self.out_dim = 512 * 4
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=3)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d(28)
        self.fc = nn.Linear(1605632, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=(1, 1)):
        """Create a layer of blocks"""
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=(1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def res_a_50(**kwargs):
    return RES_A(Bottleneck, [3, 4, 6, 3], **kwargs)


def res_c_26(**kwargs):
    return RES(BasicBlock, [1, 1, 2, 2, 2, 2, 1, 1], arch='C', **kwargs)


def res_c_42(**kwargs):
    return RES(BasicBlock, [1, 1, 3, 4, 6, 3, 1, 1], arch='C', **kwargs)


def res_c_58(**kwargs):
    return RES(Bottleneck, [1, 1, 3, 4, 6, 3, 1, 1], arch='C', **kwargs)


def res_d_22(**kwargs):
    return RES(BasicBlock, [1, 1, 2, 2, 2, 2, 1, 1], arch='D', **kwargs)


def res_d_24(**kwargs):
    return RES(BasicBlock, [1, 1, 2, 2, 2, 2, 2, 2], arch='D', **kwargs)


def res_d_38(**kwargs):
    return RES(BasicBlock, [1, 1, 3, 4, 6, 3, 1, 1], arch='D', **kwargs)


def res_d_40(**kwargs):
    return RES(BasicBlock, [1, 1, 3, 4, 6, 3, 2, 2], arch='D', **kwargs)


def res_d_54(**kwargs):
    return RES(Bottleneck, [1, 1, 3, 4, 6, 3, 1, 1], arch='D', **kwargs)


def res_d_56(**kwargs):
    return RES(Bottleneck, [1, 1, 3, 4, 6, 3, 2, 2], arch='D', **kwargs)


def res_d_105(**kwargs):
    return RES(Bottleneck, [1, 1, 3, 4, 23, 3, 1, 1], arch='D', **kwargs)


def res_d_107(**kwargs):
    return RES(Bottleneck, [1, 1, 3, 4, 23, 3, 2, 2], arch='D', **kwargs)
