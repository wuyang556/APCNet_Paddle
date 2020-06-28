# coding = utf-8
# python 3.7.3
# created by wuyang on 2020/3/2
import numpy as np
import random
from PIL import Image, ImageOps, ImageFilter


class SegDataset(object):
    def __init__(self, mode="train", **kwargs, ):
        self.mode = mode
        self.base_size = kwargs["base_size"]
        self.crop_size = kwargs["crop_size"]
        self.mean = kwargs["mean"]
        self.std = kwargs["std"]
        self.image_paths = []
        self.label_paths = []

    def _reader(self):
        for image_path, label_path in zip(self.image_paths, self.label_paths):
            image = Image.open(image_path)
            image = image.convert("RGB")
            label = Image.open(label_path)
            if self.mode == "train":
                image, label = self._sync_transform(image, label)
            elif self.mode == "val":
                image, label = self._val_sync_transform(image, label)
            else:
                assert self.mode == "test"
                image, label = self._img_transform(image), self._mask_transform(label)
            yield image, label

    def _val_sync_transform(self, img, mask):
        outsize = self.crop_size
        short_size = outsize
        w, h = img.size
        if w > h:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - outsize) / 2.))
        y1 = int(round((h - outsize) / 2.))
        img = img.crop((x1, y1, x1 + outsize, y1 + outsize))
        mask = mask.crop((x1, y1, x1 + outsize, y1 + outsize))
        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        # normalize
        img = img.transpose([2, 0, 1])
        img = img / 255.0
        if self.std is None:
            for i, (t, m) in enumerate(zip(img, self.mean)):
                img[i] = np.subtract(t, m)
        else:
            for i, (t, m, s) in enumerate(zip(img, self.mean, self.std)):
                img[i] = np.divide(np.subtract(t, m), s)

        return img, mask

    def _sync_transform(self, img, mask):
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        # random scale (short edge)
        long_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        # gaussian blur as in PSP
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        # normalize
        try:
            img = img.transpose([2, 0, 1])
        except:
            print(img.shape)
        img = img/225.0
        if self.std is None:
            for i, (t, m) in enumerate(zip(img, self.mean)):
                img[i] = np.subtract(t, m)
        else:
            for i, (t, m, s) in enumerate(zip(img, self.mean, self.std)):
                img[i] = np.divide(np.subtract(t, m), s)

        return img, mask

    def _img_transform(self, img):
        return np.array(img).astype("float32")

    def _mask_transform(self, mask):
        return np.array(mask).astype("int64")

    def __len__(self):
        return None

    def batch(self, batch_size, shuffle=False, buf_size=100, drop_last=False):
        batch_img = []
        batch_lbl = []
        reader = self.shuffle(buf_size) if shuffle else self._reader()
        for img, lbl in reader:
            assert img.shape[1:] == lbl.shape
            batch_img.append(img)
            batch_lbl.append(lbl)
            if len(batch_img) == batch_size:
                yield np.array(batch_img, dtype="float32"), np.array(batch_lbl, dtype="int64")
                batch_img = []
                batch_lbl = []
        if len(batch_img) != 0 and drop_last == False:
            yield np.array(batch_img, dtype="float32"), np.array(batch_lbl, dtype="int64")

    def shuffle(self, buf_size=100):
        buf_data = []
        for data in self._reader():
            buf_data.append(data)
            if len(buf_data) > buf_size:
                random.shuffle(buf_data)
                for b in buf_data:
                    yield b[0], b[1]
                buf_data = []
        if len(buf_data) > 0:
            random.shuffle(buf_data)
            for b in buf_data:
                yield b[0], b[1]
