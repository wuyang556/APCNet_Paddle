# coding = utf-8
# python 3.7.3
# created by wuyang on 2020/3/17
import paddle
import paddle.fluid as fluid
import numpy as np


class SegmentationLosses(object):
    """
    2D Cross Entropy Loss with Auxilary Loss
    """
    def __init__(self, aux, aux_weight, nclass=-1, weight=None):
        super(SegmentationLosses, self).__init__()
        self.aux = aux
        self.aux_weight = aux_weight
        self.nclass = nclass
        self.weight = weight

    def __call__(self, *inputs):
        if not self.aux:

            pred, target = tuple(inputs)
            return loss(pred, target, self.nclass)[0]
        else:
            pred1, pred2, target = tuple(inputs)
            loss1 = loss(pred1, target, self.nclass)[0]
            loss2 = loss(pred2, target, self.nclass)[0]
            return loss1 + self.aux_weight * loss2


def loss(logit, label, num_classes):
    label_nignore = fluid.layers.less_than(
        label.astype('float32'),
        fluid.layers.assign(np.array([num_classes], 'float32')),
        force_cpu=False).astype('float32')
    logit = fluid.layers.transpose(logit, [0, 2, 3, 1])
    logit = fluid.layers.reshape(logit, [-1, num_classes])
    label = fluid.layers.reshape(label, [-1, 1])
    label = fluid.layers.cast(label, 'int64')
    label_nignore = fluid.layers.reshape(label_nignore, [-1, 1])
    logit = fluid.layers.softmax(logit, use_cudnn=False)
    loss = fluid.layers.cross_entropy(logit, label, ignore_index=255)
    label_nignore.stop_gradient = True
    label.stop_gradient = True
    return loss, label_nignore


if __name__ == '__main__':
    from FCN32s.models import FCN32s

    with fluid.dygraph.guard():
        model = FCN32s(n_class=19)
        import numpy as np
        pred1 = np.random.rand(4, 19, 224, 224).astype("float32")
        pred2 = np.random.rand(4, 19, 224, 224).astype("float32")
        target = np.ones(shape=(4, 224, 224)).astype("int64")
        pred1 = fluid.dygraph.to_variable(pred1)
        pred2 = fluid.dygraph.to_variable(pred2)
        target = fluid.dygraph.to_variable(target)
        inputs = fluid.dygraph.to_variable(np.random.rand(4, 3, 224, 224).astype("float32"))
        out = model(inputs)[0]
        print(out.shape)
        print(out.dtype)
        ce = SegmentationLosses(aux=False, aux_weight=0.4, nclass=19)
        loss = ce(pred1, target)
        avg_loss = fluid.layers.mean(loss)
        avg_loss.backward()

        print(avg_loss.numpy())

