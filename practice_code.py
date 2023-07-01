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

x = [1,2,3]
y = [4,5,6]
print("ok")
for i in x+y:
    print(i)