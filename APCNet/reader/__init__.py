# coding = utf-8
# python 3.7.3
# created by wuyang on 2020/3/24
from .citys_reader import CitysSegDataset
from .voc_reader import VOCSegDataset, VOCAugSegDataset
from .ade20k_reader import Ade20kSegDataset

reader = {
    "voc": VOCSegDataset,
    "voc_aug": VOCAugSegDataset,
    "citys": CitysSegDataset,
    "ade20k": Ade20kSegDataset
}


def get_segmentation_reader(name, **kwargs):
    return reader[name.lower()](**kwargs)
