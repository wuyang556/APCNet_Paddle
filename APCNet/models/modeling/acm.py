# coding = utf-8
# python 3.7.3
# created by wuyang on 2020/6/6
import paddle.fluid as fluid


class AdaptivePool2D(fluid.dygraph.Layer):
    def __init__(self, pool_size, pool_type):
        super(AdaptivePool2D, self).__init__()
        self.pool_size = pool_size
        self.pool_type = pool_type

    def forward(self, x):
        x = fluid.layers.adaptive_pool2d(x, pool_size=self.pool_size, pool_type=self.pool_type)
        return x


class ACM(fluid.dygraph.Layer):
    """Adaptive Context Module"""
    def __init__(self, in_channel, s):
        super(ACM, self).__init__()
        self.gla_conv1 = fluid.dygraph.Conv2D(num_channels=in_channel, num_filters=512, filter_size=1, padding=0, bias_attr=False)
        # self.gla_gi = fluid.dygraph.Pool2D(pool_type="avg", pool_size=1, global_pooling=True)
        self.gla_gi = AdaptivePool2D(pool_size=1, pool_type="avg")
        self.gla_conv2 = fluid.dygraph.Conv2D(num_channels=512, num_filters=s**2, filter_size=1, padding=0, bias_attr=False)

        # self.ap = fluid.dygraph.Pool2D(pool_type="avg", pool_size=s, global_pooling=True)
        self.ap = AdaptivePool2D(pool_size=s, pool_type="avg")
        self.conv = fluid.dygraph.Conv2D(num_channels=in_channel, num_filters=512, filter_size=1, padding=0, bias_attr=False)

    def forward(self, x):
        x_gla = self.gla_conv1(x)
        x_gla_gi = self.gla_gi(x_gla)
        x_gla_gi_ = x_gla + x_gla_gi
        x_gla_2 = self.gla_conv2(x_gla_gi_)
        n, s, h1, w1 = x_gla_2.shape
        am = fluid.layers.reshape(x_gla_2, shape=(n, s, h1*w1))
        am = fluid.layers.transpose(am, perm=[0, 2, 1])

        ap = self.ap(x)
        conv = self.conv(ap)
        n, c, h2, w2 = conv.shape

        conv = fluid.layers.reshape(conv, shape=(n, c, h2*w2))
        conv = fluid.layers.transpose(conv, perm=[0, 2, 1])

        mp = fluid.layers.matmul(am, conv)
        mp = fluid.layers.reshape(fluid.layers.transpose(mp, perm=[0, 2, 1]), shape=(n, c, h1, w1))

        return mp + x_gla


if __name__ == '__main__':
    import numpy as np
    data = np.ones(shape=(4, 2048, 32, 32), dtype="float32")
    with fluid.dygraph.guard():
        acm = ACM(in_channel=2048, s=5)
        data = fluid.dygraph.to_variable(data)
        out = acm(data)
        print(data.shape)
