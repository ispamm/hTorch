import torch
import numpy as np
import pandas as pd
import tifffile as tiff
from shapely.wkt import loads as wkt_loads
from shapely.geometry import MultiPolygon, Polygon
import cv2
import random

# constants
import configparser

config = configparser.SafeConfigParser()
config.read("hTorch/experiments/constants.cfg")
BATCH_SIZE = config.getint("dataset", "batch_size")
REPETITIONS = config.getint("dataset", "repetitions")
SHUFFLE = config.getboolean("dataset", "shuffle")
TRAIN_SPLIT = config.getfloat("dataset", "train_split")
TEST_SPLIT = config.getfloat("dataset", "test_split")
WIDTH = config.getint("dataset", "width")
HEIGHT = config.getint("dataset", "height")
DATA_SIZE_TRAIN = config.getint("dataset", "data_size_train")
DATA_SIZE_VAL = config.getint("dataset", "data_size_val")

file_names = pd.read_csv("train_wkt_v4.csv").ImageId.unique()

"""
autor is n01z3 from his public kernel on the
DSTL Challenge
"""
def get_patches(img, msk, amt=10000, aug=True):
    is2 = int(1.0 * WIDTH)
    xm, ym = img.shape[0] - is2, img.shape[1] - is2

    x, y = [], []

    tr = [0.4, 0.1, 0.1, 0.15, 0.3, 0.95, 0.1, 0.05, 0.001, 0.005]
    for i in range(amt):
        xc = random.randint(0, xm)
        yc = random.randint(0, ym)

        im = img[xc:xc + is2, yc:yc + is2]
        ms = msk[xc:xc + is2, yc:yc + is2]

        for j in range(10):
            sm = np.sum(ms[:, :, j])
            if 1.0 * sm / is2 ** 2 > tr[j]:
                if aug:
                    if random.uniform(0, 1) > 0.5:
                        im = im[::-1]
                        ms = ms[::-1]
                    if random.uniform(0, 1) > 0.5:
                        im = im[:, ::-1]
                        ms = ms[:, ::-1]

                x.append(im)
                y.append(ms)

    x, y = 2 * np.transpose(x, (0, 3, 1, 2)) - 1, np.transpose(y, (0, 3, 1, 2))
    return x, y

def get_loader(phase, bs):
    file_names = eval("file_names_" + phase)
    if phase == "train":
        file_names = np.repeat(file_names_train, REPETITIONS)
        random.shuffle(file_names)

    data = DSTLDataset(file_names, transform=transform)
    loader = torch.utils.data.DataLoader(data, batch_size=bs, shuffle=SHUFFLE, pin_memory=True,
                                         num_workers=0, drop_last=True)

    return loader
