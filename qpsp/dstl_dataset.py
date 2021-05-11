import torch
import numpy as np
import pandas as pd
import tifffile as tiff
from shapely.wkt import loads as wkt_loads
from shapely.geometry import MultiPolygon, Polygon
import pytorch_lightning as pl
import cv2
import random 

from .constants import *

file_names = pd.read_csv("train_wkt_v4.csv").ImageId.unique()

file_names_train = file_names[:round(len(file_names)*TRAIN_SPLIT)]
file_names_val_test = file_names[round(len(file_names)*TRAIN_SPLIT):]
file_names_val = file_names_val_test[:round(len(file_names_val_test)*TEST_SPLIT)]
file_names_test = file_names_val_test[round(len(file_names_val_test)*TEST_SPLIT):]

class DSTLDataset(torch.utils.data.Dataset):

    def __init__(self, id_list, iters=1000, transform=None):
        self.grid_sizes = pd.read_csv("grid_sizes.csv", names=["id", "Xmax", "Ymin"], skiprows=1)
        self.train_wkt = pd.read_csv("train_wkt_v4.csv", names=["id", "class", "poly"], skiprows=1)
        self.iters = iters
        self.id_list = id_list
        self.transform = transform
        self.seed = 4

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, index):

        ratio = 0
        while ratio < 0.01:
            id = self.id_list[index]
            x = tiff.imread(f"sixteen_band/{id}_M.tif").astype(np.float32)
            x = (x - x.min()) / (x.max() - x.min())

            xmax_ymin = self.grid_sizes.loc[self.grid_sizes["id"] == id, ["Xmax", "Ymin"]].values[0]

            height, width = x.shape[-1], x.shape[-2]
            new_width = width * width / ((width + 1) * xmax_ymin[0])
            new_height = height * height / ((height + 1) * xmax_ymin[1])
            msk_stack = []
            for i in range(10):

                polygon = self.train_wkt.loc[(self.train_wkt["id"] == id) & (self.train_wkt["class"] == i + 1)].poly
                msk = np.zeros((width, height), np.int32)

                if len(polygon) > 0:

                    poly_list = wkt_loads(polygon.values[0])

                    perim_list = []
                    interior_list = []
                    for poly in poly_list:

                        perim = np.array(list(poly.exterior.coords))
                        perim[:, 0] *= new_width
                        perim[:, 1] *= new_height
                        perim_list.append(np.round(perim).astype(np.int32))

                        for pi in poly.interiors:
                            interior = np.array(list(pi.coords))
                            interior[:, 0] *= new_width
                            interior[:, 1] *= new_height
                            interior_list.append(np.round(interior).astype(np.int32))

                    if len(perim_list) != 0 and len(interior_list) != 0:
                        cv2.fillPoly(msk, perim_list, 1)
                        cv2.fillPoly(msk, interior_list, 0)

                msk_stack.append(msk)

            msk_stack = np.stack(msk_stack)
            ratio = np.sum(msk_stack[msk_stack == 1]) / np.prod(msk_stack.shape)

            y = np.stack(msk_stack, 2).transpose(2, 0, 1)
            if self.transform:
                for transf in self.transform:
                    np.random.seed(self.seed)
                    x, y = transf(x, y)
                self.seed += 1
            else:
                x = torch.from_numpy(x)
                y = torch.from_numpy(y)

            if index != len(self.id_list) - 1:
                index += 1
            else:
                index = 0

        return x, y



def RandomCrop(img, mask, size=(WIDTH, HEIGHT)):
    width, height = size
    x = random.randint(0, img.shape[-1] - width)
    y = random.randint(0, img.shape[-2] - height)
    img = img[..., y:y + height, x:x + width]
    mask = mask[..., y:y + height, x:x + width]

    return img, mask


def RandomVerticalFlip(img, mask):
    bin = np.random.binomial(1, 0.5)
    if bin:
        img, mask = np.flipud(img).copy(), np.flipud(mask).copy()
    return img, mask


def RandomHorizontalFlip(img, mask):
    bin = np.random.binomial(1, 0.5)
    if bin:
        img, mask = np.fliplr(img).copy(), np.fliplr(mask).copy()
    return img, mask


class LitDSTL(pl.LightningDataModule):

    def __init__(self):
        super().__init__()
        self.transform = [
            RandomCrop,
            RandomVerticalFlip,
            RandomHorizontalFlip
        ]

    def train_dataloader(self):
        repeated = np.repeat(file_names_train, REPETITIONS)
        random.shuffle(repeated)
        
        train = DSTLDataset(repeated, transform=self.transform)
        loader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=SHUFFLE, pin_memory=True,
                                             num_workers=0)
        return loader

    def val_dataloader(self):
        val = DSTLDataset(file_names_val, transform=self.transform)
        return torch.utils.data.DataLoader(val, batch_size=BATCH_SIZE, shuffle=SHUFFLE, pin_memory=True, num_workers=0)

    def test_dataloader(self):
        test = DSTLDataset(file_names_test, transform=self.transform)
        return torch.utils.data.DataLoader(test, batch_size=BATCH_SIZE, shuffle=SHUFFLE, pin_memory=True, num_workers=0)

