# coding = utf-8
# python 3.7.3
# created by wuyang on 2020/3/2
import os
# from .base_reader import SegDataset
from PdSeg.reader.base_reader import SegDataset


# class VOCSegDataset(SegDataset):
#     NUM_CLASSES = 21
#
#     def __init__(self, data_path="~/work/dataset", mode="train", **kwargs):
#         super(VOCSegDataset, self).__init__(mode, **kwargs)
#         root = os.path.join(os.path.expanduser(data_path), "VOC2012")
#         list_root = os.path.join(root, "ImageSets/Segmentation")
#         if mode == "train":
#             self.list_path = os.path.join(list_root, "train.list")
#         elif mode == "val":
#             self.list_path = os.path.join(list_root, "val.list")
#         elif mode == "test":
#             self.list_path = os.path.join(list_root, "test.list")
#         else:
#             raise Exception(f"{mode} is a wrong mode.")
#         assert os.path.exists(self.list_path), f"{self.list_path} is not existed."
#         with open(self.list_path, "r") as f:
#             info_lines = f.readlines()
#             for info in info_lines:
#                 info = info.strip("\n")
#                 info = info.split(" ")
#                 self.image_paths.append(os.path.join(root, info[0]))
#                 self.label_paths.append(os.path.join(root, info[1]))
#
#     def __len__(self):
#         return len(self.image_paths)


class VOCAugSegDataset(SegDataset):
    NUM_CLASSES = 21

    def __init__(self, data_path="~/work/dataset", mode="train", **kwargs):
        super(VOCAugSegDataset, self).__init__(mode, **kwargs)
        root = os.path.join(os.path.expanduser(data_path), "VOC2012")
        list_root = os.path.join(root, "ImageSets/Segmentation/pascal_aug")
        image_folder = os.path.join(root, "JPEGImages")
        label_folder = os.path.join(root, "SegmentationClass")
        if mode == "train":
            self.list_path = os.path.join(list_root, "trainaug.txt")
        elif mode == "val":
            self.list_path = os.path.join(list_root, "val.txt")
        elif mode == "test":
            self.list_path = os.path.join(list_root, "test.txt")
        else:
            raise Exception(f"{mode} is a wrong mode.")
        assert os.path.exists(self.list_path), f"{self.list_path} is not existed."
        with open(self.list_path, "r") as f:
            info_lines = f.readlines()
            for info in info_lines:
                info = info.strip("\n")
                self.image_paths.append(os.path.join(image_folder, info+".jpg"))
                self.label_paths.append(os.path.join(label_folder, info+".png"))

    def __len__(self):
        return len(self.image_paths)


class VOCSegDataset(SegDataset):
    NUM_CLASSES = 21

    def __init__(self, data_path="~/work/dataset", mode="train", **kwargs):
        super(VOCSegDataset, self).__init__(mode, **kwargs)
        root = os.path.join(os.path.expanduser(data_path), "VOC2012")
        list_root = os.path.join(root, "ImageSets/Segmentation/voc2012")
        image_folder = os.path.join(root, "JPEGImages")
        label_folder = os.path.join(root, "SegmentationClass")
        if mode == "train":
            self.list_path = os.path.join(list_root, "train.txt")
        elif mode == "val":
            self.list_path = os.path.join(list_root, "val.txt")
        elif mode == "test":
            self.list_path = os.path.join(list_root, "test.txt")
        else:
            raise Exception(f"{mode} is a wrong mode.")
        assert os.path.exists(self.list_path), f"{self.list_path} is not existed."
        with open(self.list_path, "r") as f:
            info_lines = f.readlines()
            for info in info_lines:
                info = info.strip("\n")
                self.image_paths.append(os.path.join(image_folder, info + ".jpg"))
                self.label_paths.append(os.path.join(label_folder, info + ".png"))

    def __len__(self):
        return len(self.image_paths)


if __name__ == '__main__':
    from tqdm import tqdm
    import numpy as np
    import matplotlib.pyplot as plt
    data_path = r"E:\Data\datasets"
    kwargs = {"base_size": 520, "crop_size": 512, "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}
    voc = VOCAugSegDataset(data_path, mode="val", **kwargs)
    print(len(voc))
    batch_reader = voc.batch(batch_size=4, shuffle=True, buf_size=10, drop_last=True)
    tbar = tqdm(batch_reader)
    # tbar.se
    for img, lbl in tbar:
        # print(img.shape, lbl.shape)
        plt.imshow(lbl[0])
        plt.show()
        break
    pass
