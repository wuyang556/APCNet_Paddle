# coding = utf-8
# python 3.7.3
# created by wuyang on 2020/3/23
import os
import paddle
import paddle.fluid as fluid
from ..nn import ReLU, Dropout2d


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return fluid.dygraph.Conv2D(in_planes, out_planes, filter_size=3, stride=stride,
                     padding=dilation, groups=groups, bias_attr=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return fluid.dygraph.Conv2D(in_planes, out_planes, filter_size=1, stride=stride, bias_attr=False)


class BasicBlock(fluid.dygraph.Layer):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = fluid.dygraph.BatchNorm
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(fluid.dygraph.Layer):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, previous_dilation=1, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = fluid.dygraph.BatchNorm
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(fluid.dygraph.Layer):

    def __init__(self, block, layers, num_classes=1000, dilated=False, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, output_size=8):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = fluid.dygraph.BatchNorm
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = fluid.dygraph.Conv2D(3, self.inplanes, filter_size=7, stride=2, padding=3,
                               bias_attr=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = ReLU()
        self.maxpool = fluid.dygraph.Pool2D(pool_size=3, pool_stride=2, pool_padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        dilation_rate = 2
        if dilated and output_size <= 8:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
                                           dilation=dilation_rate)
            dilation_rate *= 2
        else:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                           dilate=replace_stride_with_dilation[1])
        if dilated and output_size <= 16:
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                           dilation=dilation_rate)
        else:
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                           dilate=replace_stride_with_dilation[2])
        self.avgpool = fluid.dygraph.Pool2D(pool_size=1, pool_type="avg", global_pooling=True)
        self.fc = fluid.dygraph.Linear(512 * block.expansion, num_classes)


    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, dilation=1):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = fluid.dygraph.container.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        if dilation == 1 or dilation == 2:
            layers.append(block(self.inplanes, planes, stride, dilation=1,
                                downsample=downsample, previous_dilation=dilation, norm_layer=norm_layer))
        elif dilation == 4:
            layers.append(block(self.inplanes, planes, stride, dilation=2,
                                downsample=downsample, previous_dilation=dilation, norm_layer=norm_layer))
        else:
            raise RuntimeError("=> unknown dilation size: {}".format(dilation))
        # layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
        #                     self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=dilation, previous_dilation=dilation,
                                norm_layer=norm_layer))

        return fluid.dygraph.container.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        x = self.avgpool(x)
        x = fluid.layers.flatten(x, 1)
        x = self.fc(x)

        return c1, c2, c3, c4

def _resnet(arch, block, layers, pretrained, model_path,  **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = fluid.dygraph.load_dygraph(model_path)
        model.set_dict(state_dict[0])
    return model


def resnet18(pretrained=False, root="~/work/PdSeg/models/backbone", **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model_path = os.path.join(os.path.expanduser(root), "resnet18")
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, model_path,
                   **kwargs)


def resnet34(pretrained=False, root="~/work/PdSeg/models/backbone", **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model_path = os.path.join(os.path.expanduser(root), "resnet34")
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, model_path,
                   **kwargs)


def resnet50(pretrained=False, root="~/work/PdSeg/models/backbone", **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model_path = os.path.join(os.path.expanduser(root), "resnet50_th")
    model_path = r"E:\Code\Python\PaddleSeg\PdSeg\models\backbone\resnet50_th"
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, model_path,  dilated=False,
                   **kwargs)


def resnet101(pretrained=False, root="~/work/PdSeg/models/backbone", **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model_path = os.path.join(os.path.expanduser(root), "resnet101_th")
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, model_path, dilated=True,
                   **kwargs)


def resnet152(pretrained=False, root="~/work/PdSeg/models/backbone", **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model_path = os.path.join(os.path.expanduser(root), "resnet152_th")
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, model_path, dilated=True,
                   **kwargs)


def resnext50_32x4d(pretrained=False, root="~/work/PdSeg/models/backbone", **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    model_path = os.path.join(os.path.expanduser(root), "resnet18")
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, model_path, **kwargs)


def resnext101_32x8d(pretrained=False, root="~/work/PdSeg/models/backbone", **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    model_path = os.path.join(os.path.expanduser(root), "resnet18")
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, model_path, **kwargs)


def wide_resnet50_2(pretrained=False, root="~/work/PdSeg/models/backbone", **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    model_path = os.path.join(os.path.expanduser(root), "resnet18")
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, model_path, **kwargs)


def wide_resnet101_2(pretrained=False, root="~/work/PdSeg/models/backbone", **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    model_path = os.path.join(os.path.expanduser(root), "resnet18")
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, model_path, **kwargs)


if __name__ == '__main__':
    import torch
    with fluid.dygraph.guard():
        model = resnet50()
        state_dict = model.state_dict()
        print(state_dict.keys())
        print(len(state_dict.keys()))

        model_torch_path = r"C:\Users\wuyang\.cache\torch\checkpoints\resnet50-19c8e357.pth"
        model_torch = torch.load(model_torch_path, map_location=torch.device("cpu"))
        print(model_torch.keys())
        print(len(model_torch.keys()))

        new_state_dict = {}
        for key in state_dict.keys():
            if key in model_torch.keys():
                if "fc" in key:
                    print(state_dict[key].shape, model_torch[key].detach().numpy().transpose().shape)
                    new_state_dict[key] = fluid.dygraph.to_variable(model_torch[key].detach().numpy().transpose().astype("float32"))
                else:
                    print(state_dict[key].shape, model_torch[key].detach().numpy().shape)
                    new_state_dict[key] = fluid.dygraph.to_variable(model_torch[key].detach().numpy().astype("float32"))
            else:
                if "_mean" in key:
                    torch_key = key.replace("_mean", "running_mean")
                    print(state_dict[key].shape, model_torch[torch_key].detach().numpy().shape)
                    new_state_dict[key] = fluid.dygraph.to_variable(model_torch[torch_key].detach().numpy().astype("float32"))
                if "_variance" in key:
                    torch_key = key.replace("_variance", "running_var")
                    print(state_dict[key].shape, model_torch[torch_key].detach().numpy().shape)
                    new_state_dict[key] = fluid.dygraph.to_variable(model_torch[torch_key].detach().numpy().astype("float32"))

        print(len(new_state_dict.keys()))

        model.set_dict(new_state_dict)
        fluid.dygraph.save_dygraph(model.state_dict(), "./resnet50")

        import torchvision.models.alexnet
