# coding = utf-8
# python 3.7.3
# created by wuyang on 2020/2/28
import os
from PdSeg.reader.base_reader import SegDataset


class CitysSegDataset(SegDataset):
    NUM_CLASSES = 19

    def __init__(self, data_path="~/work/dataset", mode="train", **kwargs):
        super(CitysSegDataset, self).__init__(mode, **kwargs)
        root = os.path.join(os.path.expanduser(data_path), "cityscapes")
        if mode == "train":
            self.list_path = os.path.join(root, "train.list")
        elif mode == "val":
            self.list_path = os.path.join(root, "val.list")
        elif mode == "test":
            self.list_path = os.path.join(root, "test.list")
        else:
            raise Exception(f"{mode} is a wrong mode.")
        assert os.path.exists(self.list_path), f"{self.list_path} is not existed."
        with open(self.list_path, "r") as f:
            info_lines = f.readlines()
            for info in info_lines:
                info = info.strip("\n")
                info = info.split(" ")
                self.image_paths.append(os.path.join(root, info[0]))
                self.label_paths.append(os.path.join(root, info[1]))

    def __len__(self):
        return len(self.label_paths)


if __name__ == '__main__':
    data_path = r"E:\Code\Python\PaddleSeg\PSPNet\data"
    kwargs = {"base_size": 2048, "crop_size": 768, "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}
    city = CitysSegDataset(data_path, mode="val", **kwargs)
    print(len(city.label_paths))
    # for img, lbl in city.reader()():
    #     print(img.shape)
    #     print(lbl.shape)
    #     break

    # import paddle
    # import paddle.fluid as fluid
    # train_reader = paddle.batch(city.reader(), batch_size=4)
    # for data in train_reader():
    #     print(len(data))
    #     print(len(data[0]))
    #     print(data[0][0].shape, )
    #     break
    import matplotlib.pyplot as plt
    from tqdm import trange
    for epoch in range(2):
        batch_reader = city.batch(batch_size=4, shuffle=True)
        # for img, lbl in batch_reader:
        #     print(f"Epoch:{epoch}", img.shape, lbl.shape)
        steps = int(len(city)/4)
        for i in trange(steps):
            img, lbl = next(batch_reader)
            print(f"Epoch:{epoch}", img.shape, lbl.shape)
            plt.imshow(img[0].transpose())
            import numpy as np
            # plt.imshow(np.transpose(img[0], axes=(1, 2, 0)))
            plt.imshow(lbl[0])
            plt.show()
            break
