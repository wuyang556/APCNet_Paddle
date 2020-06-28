# coding = utf-8
# python 3.7.3
# created by wuyang on 2020/3/17
import threading
import numpy as np
import paddle
import paddle.fluid as fluid


class SegmentationMetric(object):
    """
    Computes pixAcc and mIoU metric scores
    """
    def __init__(self, nclass):
        self.nclass = nclass
        self.lock = threading.Lock()
        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0
        self.reset()

    # def evaluate_worker(self, label, pred):
    #     correct, labeled = batch_pix_accuracy(pred, label)
    #     inter, union = batch_intersection_union(pred, label, self.nclass)
    #     with self.lock:
    #         self.total_correct += correct
    #         self.total_label += labeled
    #         self.total_inter += inter
    #         self.total_union += union
    #     return

    def update(self, labels, preds):
        correct, labeled = batch_pix_accuracy(preds, labels)
        inter, union = batch_intersection_union(preds, labels, self.nclass)
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union

        # def evaluate_worker(label, pred):
        #     correct, labeled = batch_pix_accuracy(pred, label)
        #     inter, union = batch_intersection_union(pred, label, self.nclass)
        #     with self.lock:
        #         self.total_correct += correct
        #         self.total_label += labeled
        #         self.total_inter += inter
        #         self.total_union += union
        #     return
        #
        # evaluate_worker(labels, preds)
        # if isinstance(preds, paddle.fluid.Tensor):
        #     evaluate_worker(labels, preds)
        # elif isinstance(preds, (list, tuple)):
        #     threads = [threading.Thread(target=evaluate_worker,
        #                                 args=(self, label, pred)
        #                                 )
        #                for (label, pred) in zip(labels, preds)
        #     ]
        #     for thread in threads:
        #         thread.start()
        #     for thread in threads:
        #         thread.join()
        # else:
        #     raise NotImplemented

    def get(self):
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return pixAcc, mIoU

    def reset(self):
        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0
        return


def batch_pix_accuracy(output, target):
    """
    Batch Pixel Accuracy.
    :param output: input 4D tensor
    :param target: label 3D tensor
    :return:
    """
    predict = fluid.layers.argmax(output, axis=1)

    predict = predict.numpy().astype("int64") + 1
    target = target.numpy().astype("int64") + 1

    pixel_labeled = np.sum(target > 0)
    pixel_correct = np.sum((predict == target)*(target > 0))
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than labeled"

    return pixel_correct, pixel_labeled


def batch_intersection_union(output, target, nclass):
    """
    Batch Intersection of Union.
    :param ouput: input 4D tensor
    :param target: label 3D tensor
    :param nclass: number of categories (int)
    :return:
    """
    predict = output.numpy()
    predict = np.argmax(predict, axis=1)
    predict = fluid.layers.argmax(output, axis=1)
    mini = 1
    maxi = nclass
    nbins = nclass
    predict = predict.numpy().astype("int64") + 1
    target = target.numpy().astype("int64") + 1

    predict = predict * (target > 0).astype(predict.dtype)
    intersection = predict * (predict == target)
    # areas of intersection and union
    area_inter, _ = np.histogram(intersection, bins=nbins, range=(mini, maxi))
    area_pred, _ = np.histogram(predict, bins=nbins, range=(mini, maxi))
    area_lab, _ = np.histogram(target, bins=nbins, range=(mini, maxi))
    area_union = area_pred + area_lab - area_inter
    assert (area_inter <= area_union).all(), "Intersection area should be smaller than Union area"

    return area_inter, area_union


if __name__ == '__main__':
    in_np = np.random.rand(4, 21, 512, 512).astype("float32")
    label_np = np.ones(shape=(4, 512, 512)).astype("int32")
    in_np1 = np.random.rand(4, 21, 512, 512).astype("float32")
    with fluid.dygraph.guard():
        metric = SegmentationMetric(21)
        in_var = fluid.dygraph.to_variable(in_np)
        in_var1 = fluid.dygraph.to_variable(in_np1)
        label_var = fluid.dygraph.to_variable(label_np)
        metric.update(label_var, in_var)
        pixAcc, mIoU = metric.get()
        inter, union = batch_intersection_union(in_var, label_var, 21)
        correct, labeled = batch_pix_accuracy(in_var, label_var)

        print(inter)
        print(union)
        print(correct)
        print(labeled)
        print(pixAcc, mIoU)
        metric.update(label_var, in_var1)
        pixAcc, mIoU = metric.get()
        print(pixAcc, mIoU)


