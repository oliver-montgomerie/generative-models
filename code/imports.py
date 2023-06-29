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
    Spacingd,
    ResizeWithPadOrCropd,
)
from monai.utils import set_determinism

## Top for viewing. Below lines for saving
#import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt