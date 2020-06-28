# coding = utf-8
# python 3.7.3
# created by wuyang on 2020/3/24

import paddle.fluid as fluid


class PixelShuffle(fluid.dygraph.Layer):
    def __init__(self, upscale_factor):
        super(PixelShuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        return fluid.layers.pixel_shuffle(x, upscale_factor=self.upscale_factor)


class Dropout2d(fluid.dygraph.Layer):
    def __init__(self, p=0.5):
        super(Dropout2d, self).__init__()
        self.p = p

    def forward(self, x):
        # if self.train():
        #     return fluid.layers.dropout(x, dropout_prob=self.p)
        # else:
        #     return x
        return fluid.layers.dropout(x, dropout_prob=self.p)


class Softmax(fluid.dygraph.Layer):
    def __init__(self, axis=-1):
        super(Softmax, self).__init__()
        self.axis = axis

    def forward(self, x):
        return fluid.layers.softmax(x, axis=self.axis)


class Sigmoid(fluid.dygraph.Layer):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x):
        return fluid.layers.sigmoid(x)


class ReLU(fluid.dygraph.Layer):
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        return fluid.layers.relu(x)


class ReLU6(fluid.dygraph.Layer):
    def __init__(self):
        super(ReLU6, self).__init__()

    def forward(self, x):
        return fluid.layers.relu6(x)


class AdaptiveAvgPool2d(fluid.dygraph.Layer):
    def __init__(self, pool_size, pool_type="avg"):
        super(AdaptiveAvgPool2d, self).__init__()
        self.pool_size = pool_size
        self.pool_type = pool_type

    def forward(self, x):
        return fluid.layers.adaptive_pool2d(x, pool_size=self.pool_size, pool_type=self.pool_type)


class AdaptiveAvgPool3d(fluid.dygraph.Layer):
    def __init__(self, pool_size, pool_type="avg"):
        super(AdaptiveAvgPool3d, self).__init__()
        self.pool_size = pool_size
        self.pool_type = pool_type

    def forward(self, x):
        return fluid.layers.adaptive_pool3d(x, pool_size=self.pool_size, pool_type=self.pool_type)


class SeparableConv2d(fluid.dygraph.Layer):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = fluid.dygraph.Conv2D(inplanes, inplanes, kernel_size, stride, padding, dilation, groups=inplanes, bias_attr=bias)
        self.bn = fluid.dygraph.BatchNorm(inplanes)
        self.pointwise = fluid.dygraph.Conv2D(inplanes, planes, 1, 1, 0, 1, 1, bias_attr=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


class JPU(fluid.dygraph.Layer):
    def __init__(self, in_channels, width=512, norm_layer=None):
        super(JPU, self).__init__()

        self.conv5 = fluid.dygraph.container.Sequential(
            fluid.dygraph.Conv2D(in_channels[-1], width, 3, padding=1, bias_attr=False),
            norm_layer(width),
            ReLU())
        self.conv4 = fluid.dygraph.container.Sequential(
            fluid.dygraph.Conv2D(in_channels[-2], width, 3, padding=1, bias_attr=False),
            norm_layer(width),
            ReLU())
        self.conv3 = fluid.dygraph.container.Sequential(
            fluid.dygraph.Conv2D(in_channels[-3], width, 3, padding=1, bias_attr=False),
            norm_layer(width),
            ReLU())

        self.dilation1 = fluid.dygraph.container.Sequential(SeparableConv2d(3*width, width, kernel_size=3, padding=1, dilation=1, bias=False),
                                       norm_layer(width),
                                       ReLU())
        self.dilation2 = fluid.dygraph.container.Sequential(SeparableConv2d(3*width, width, kernel_size=3, padding=2, dilation=2, bias=False),
                                       norm_layer(width),
                                       ReLU())
        self.dilation3 = fluid.dygraph.container.Sequential(SeparableConv2d(3*width, width, kernel_size=3, padding=4, dilation=4, bias=False),
                                       norm_layer(width),
                                       ReLU())
        self.dilation4 = fluid.dygraph.container.Sequential(SeparableConv2d(3*width, width, kernel_size=3, padding=8, dilation=8, bias=False),
                                       norm_layer(width),
                                       ReLU())

    def forward(self, *inputs):
        feats = [self.conv5(inputs[-1]), self.conv4(inputs[-2]), self.conv3(inputs[-3])]
        _, _, h, w = feats[-1].shape
        feats[-2] = fluid.layers.resize_bilinear(feats[-2], out_shape=(h, w), align_corners=True, align_mode=True)
        feats[-3] = fluid.layers.resize_bilinear(feats[-3], out_shape=(h, w), align_corners=True, align_mode=True)
        feat = fluid.layers.concat(feats, axis=1)
        feat = fluid.layers.concat([self.dilation1(feat), self.dilation2(feat), self.dilation3(feat), self.dilation4(feat)], axis=1)

        return inputs[0], inputs[1], inputs[2], feat


class PyramidPooling(fluid.dygraph.Layer):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """
    def __init__(self, in_channels, norm_layer, up_kwargs):
        super(PyramidPooling, self).__init__()
        self.pool1 = AdaptiveAvgPool2d(1)
        self.pool2 = AdaptiveAvgPool2d(2)
        self.pool3 = AdaptiveAvgPool2d(3)
        self.pool4 = AdaptiveAvgPool2d(6)

        out_channels = int(in_channels/4)
        self.conv1 = fluid.dygraph.container.Sequential(
            fluid.dygraph.Conv2D(in_channels, out_channels, 1, bias_attr=False),
            norm_layer(out_channels),
            ReLU())
        self.conv2 = fluid.dygraph.container.Sequential(
            fluid.dygraph.Conv2D(in_channels, out_channels, 1, bias_attr=False),
            norm_layer(out_channels),
            ReLU())
        self.conv3 = fluid.dygraph.container.Sequential(
            fluid.dygraph.Conv2D(in_channels, out_channels, 1, bias_attr=False),
            norm_layer(out_channels),
            ReLU())
        self.conv4 = fluid.dygraph.container.Sequential(
            fluid.dygraph.Conv2D(in_channels, out_channels, 1, bias_attr=False),
            norm_layer(out_channels),
            ReLU())
        # bilinear upsample options
        self._up_kwargs = up_kwargs

    def forward(self, x):
        _, _, h, w = x.shape
        # feat1 = F.interpolate(self.conv1(self.pool1(x)), (h, w), **self._up_kwargs)
        feat1 = fluid.layers.resize_bilinear(self.conv1(self.pool1(x)), out_shape=(h, w))
        # feat2 = F.interpolate(self.conv2(self.pool2(x)), (h, w), **self._up_kwargs)
        feat2 = fluid.layers.resize_bilinear(self.conv2(self.pool2(x)), out_shape=(h, w))
        # feat3 = F.interpolate(self.conv3(self.pool3(x)), (h, w), **self._up_kwargs)
        feat3 = fluid.layers.resize_bilinear(self.conv3(self.pool3(x)), out_shape=(h, w))
        # feat4 = F.interpolate(self.conv4(self.pool4(x)), (h, w), **self._up_kwargs)
        feat4 = fluid.layers.resize_bilinear(self.conv4(self.pool4(x)), out_shape=(h, w))
        return fluid.layers.concat([x, feat1, feat2, feat3, feat4], 1)


if __name__ == '__main__':
    with fluid.dygraph.guard():
        import numpy as np
        in_np1 = np.random.rand(4, 512, 16, 16).astype("float32")
        in_var1 = fluid.dygraph.to_variable(in_np1)
        in_np2 = np.random.rand(4, 1024, 32, 32).astype("float32")
        in_var2 = fluid.dygraph.to_variable(in_np2)
        in_np3 = np.random.rand(4, 2048, 64, 64).astype("float32")
        in_var3 = fluid.dygraph.to_variable(in_np3)
        jpu = JPU([512, 1024, 2048], norm_layer=fluid.dygraph.BatchNorm)
        out = jpu(*[in_var1, in_var2, in_var3])
        # sepconv = SeparableConv2d(1024, 512)
        # out = sepconv(in_var2)
        # out = fluid.layers.resize_bilinear(input=in_var2, out_shape=(64, 64), align_corners=True, align_mode=True)
        print(out[-1].shape)
