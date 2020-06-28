# coding = utf-8
# python 3.7.3
# created by wuyang on 2020/6/6
import paddle.fluid as fluid
from .backbone import resnet50, resnet101, resnet152
from .nn import ReLU, Dropout2d
from .modeling import ACM


class APCNet(fluid.dygraph.Layer):
    """
    Adaptive Pyramid Context Network
    """
    def __init__(self,  backbone, pretrained=False, root=None, nclass=21):
        super(APCNet, self).__init__()
        if backbone == "resnet50":
            self.backbone = resnet50(pretrained, root)
        elif backbone == "resnet101":
            self.backbone = resnet101(pretrained, root)
        elif backbone == "resnet152":
            self.backbone = resnet152(pretrained, root)

        self.head = APCNetHead(2048, nclass, norm_layer=fluid.dygraph.Layer)

    def forward(self, x):
        _, _, h, w = x.shape
        _, _, c3, c4 = self.backbone(x)
        outputs = []
        x = self.head(c4)
        x = fluid.layers.resize_bilinear(x, out_shape=(h, w))
        outputs.append(x)
        return tuple(outputs)


class APCNetHead(fluid.dygraph.Layer):
    def __init__(self, in_channels, out_channels, s=(1, 2, 3, 6), norm_layer=fluid.dygraph.BatchNorm):
        super(APCNetHead, self).__init__()
        self.acm = fluid.dygraph.LayerList()
        for S in s:
            self.acm.append(
                fluid.dygraph.Sequential(
                    ACM(in_channels, S),
                    norm_layer(512),
                    ReLU()
                ))

        inter_channels = in_channels // 4
        self.cls = fluid.dygraph.Sequential(
            fluid.dygraph.Conv2D(in_channels+len(s)*512, inter_channels,3, padding=1, bias_attr=False),
            norm_layer(inter_channels),
            ReLU(),
            Dropout2d(0.1),
            fluid.dygraph.Conv2D(inter_channels, out_channels, 1)
        )

    def forward(self, x):
        feats = [x]
        for acm in self.acm:
            feats.append(acm(x))
        feats = fluid.layers.concat(feats, axis=1)
        return self.cls(feats)
