import time
#from IPython import display
import logging
import os
import shutil
import sys
import tempfile
import random
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
import torch
from urllib.request import urlretrieve
import gzip
from PIL import Image

from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import CacheDataset, DataLoader
from monai.networks.nets import VarAutoEncoder
from monai.transforms import (
    EnsureChannelFirstD,
    Compose,
    LoadImageD,
    ScaleIntensityRanged,
    EnsureTypeD,
    Orientationd,
    Rotate90d,

)
from monai.utils import set_determinism

# latent_size = 3
# dist = torch.distributions.normal.Normal(torch.tensor(0.0), torch.tensor(1.0))

# for i in range(10):
#     sample = torch.zeros(1, latent_size)
#     for s in range(sample.shape[1]):
#         sample[0,s] = dist.icdf((torch.rand(1)*0.9) + 0.05)

# print(sample)

# x = [1,2,3,4,5,6]
# y = [2,4,5,6]
# z = [3,9,15]

# for a in x:
#     if a not in y + z:
#         print(a)

# print(len(x))
# x = x[0:round(len(x)*0.75)]
# print(x)

# train_files_nums = ['27', '125', '124', '113', '23', '92', '36', '120', '13', '50', '110', '126', '99', '118', '73', '59', '109', '24', '44', '29', '116', '78', '104', '31', '66', '56', '88', '43', '7', '83', '108', '40', '77', '35', '121', '86', '55', '18', '69', '70', '81', '9', '11', '22', '103', '74', '107', '58', '90', '17', '12', '26', '127', '96', '5', '101', '21', '16', '62', '39', '72', '112', '71', '6', '0', '85', '102', '3', '65', '64', '128', '122']
# val_files_nums = ['75', '33', '49', '19', '61', '111', '53', '30', '28', '129', '20', '45', '51', '25', '60', '10', '84', '93', '76', '8', '97', '46', '15']
# test_files_nums = ['2', '80', '117', '67', '48', '123', '94', '1', '57', '79', '95', '63', '4', '130', '68', '37', '82', '42', '14', '100', '98', '54', '52']
# no_tumor_file_nums = []
# for x in range(131):
#     if str(x) not in (train_files_nums + val_files_nums + test_files_nums):
#         no_tumor_file_nums.append(str(x))

# print(no_tumor_file_nums)

# print(len(train_files_nums + val_files_nums + test_files_nums))
# print(len(no_tumor_file_nums))

if not False:
    print("hi")