# coding = utf-8
# python 3.7.3
# created by wuyang on 2020/3/29
import os
import matplotlib.pyplot as plt


class RecordViz(object):
    """Visualize the results"""
    def __init__(self, args):
        record_folder = os.path.join(args.logger_folder, "record")
        if not os.path.exists(record_folder):
            os.makedirs(record_folder)
        self.filename = os.path.join(record_folder, f"{args.model}_{args.backbone}_{args.backbone_style}_{args.pu}_{args.reader}_record.jpg")
        self.epoch = []
        self.loss = []
        self.pixAcc = []
        self.mIoU = []

    def add_data_train(self, epoch, loss):
        self.epoch.append(epoch)
        self.loss.append(loss)

    def add_data_val(self, pixAcc, mIoU):
        self.pixAcc.append(pixAcc)
        self.mIoU.append(mIoU)

    def plot(self):

        plt.figure(figsize=(12, 2))
        plt.subplot(121)
        plt.plot(self.epoch, self.loss, label="Train/loss")
        plt.legend(loc="upper right")
        plt.grid()

        plt.subplot(122)
        plt.ylim((0, 1))
        plt.scatter(self.epoch, self.pixAcc, label="Val/pixAcc")
        plt.scatter(self.epoch, self.mIoU, label="Val/mIoU")
        plt.legend(loc="upper right")
        plt.grid()

        plt.savefig(self.filename)
        plt.close()


if __name__ == '__main__':
    import random
    record_vizer = RecordViz("./viz.jpg")
    for i in range(240):
        record_vizer.add_data_train(i+1, random.randrange(0, 2))
        record_vizer.add_data_val(random.randrange(0, 2), random.randrange(0, 2))
    record_vizer.plot()