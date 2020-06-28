# coding = utf-8
# python 3.7.3
# created by wuyang on 2020/3/29
import os
# from .base_reader import SegDataset
from PdSeg.reader.base_reader import SegDataset


class Ade20kSegDataset(SegDataset):
    NUM_CLASSES = 150

    def __init__(self, data_path="~/work/dataset", mode="train", **kwargs):
        super(Ade20kSegDataset, self).__init__(mode, **kwargs)
        root = os.path.join(os.path.expanduser(data_path), "ADEChallengeData2016")
        self.image_folder = os.path.join(root, "images")
        self.label_folder = os.path.join(root, "annotations")
        if mode == "train":
            self.list_path = os.path.join(root, "training.txt")
            self.getFile(folder="training")
        elif mode == "val":
            self.list_path = os.path.join(root, "validation.txt")
            self.getFile(folder="validation")
        elif mode == "test":
            self.list_path = os.path.join(root, "testing.txt")
            self.getFile(folder="testing")
        else:
            raise Exception(f"{mode} is a wrong mode.")
        assert os.path.exists(self.list_path), f"{self.list_path} is not existed."

    def getFile(self, folder):
        with open(self.list_path, "r") as f:
            info_lines = f.readlines()
            for info in info_lines:
                info = info.strip("\n")
                self.image_paths.append(os.path.join(self.image_folder, folder, info+".jpg"))
                self.label_paths.append(os.path.join(self.label_folder, folder, info+".png"))

    def __len__(self):
        return len(self.image_paths)


if __name__ == '__main__':
    from tqdm import tqdm
    data_path = r"E:\Data\datasets"
    kwargs = {"base_size": 520, "crop_size": 512, "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}
    ade20k = Ade20kSegDataset(data_path, mode="val", **kwargs)
    print(len(ade20k))
    batch_reader = ade20k.batch(batch_size=4, shuffle=True, buf_size=10, drop_last=True)
    tbar = tqdm(batch_reader)
    # tbar.se
    for img, lbl in tbar:
        # print(img.shape, lbl.shape)
        pass