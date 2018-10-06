# coding:utf-8
import os, sys, time
import numpy as np
from PIL import Image
import chainer
import glob, itertools

class StampDataset(chainer.dataset.DatasetMixin):
    def __init__(self, root, augmentation=False):
        print(' Initialize dataset')
        self._root = root
        self._augmentation = augmentation

        # Get image file path
        dnames = glob.glob('{}/*'.format(self._root))
        fnames = glob.glob('/*.png'.format(d) for d in dnames)
        print(fnames)
        fnames = list(itertools.chain.from_iterable(fnames))
        print(fnames)
        self._fnames = fnames

        print(' Initilazation done')

    def __len__(self):
        return len(self._fnames)

    def transform(self, image):
        # Random right left transform
        if np.random.rand() > 0.5:
            img = img[:, :, ::-1]
        img += np.random.uniform(size=img.shape, low=0, high=1./128)
        return img

    def get_example(self, i):
        img = Image.open(self._fnames(i))
        # Normalization [-1,1]
        img = np.asarray(img).astype(np.float32).transpose(2,0,1)/128.0-1. # (ch, y, x)
        if not self._augmentation:
            return img

        return self.transform(img)
