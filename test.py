# %%

import pandas as pd
import numpy as np
from pathlib import Path
import random

import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model

from core.utils import *

# model_path = "./models/human_parsing_mbv2-50epochs"
# model = load_model(model_path)

# %%

# METADATA



# CONFIG

""" Dataset

url: https://sites.google.com/view/11khands

@article{afifi201911kHands,
  title = {11K Hands: gender recognition and biometric identification using a large dataset of hand images},
  author = {Afifi, Mahmoud},
  journal = {Multimedia Tools and Applications},
  doi = {10.1007/s11042-019-7424-8},
  url = {https://doi.org/10.1007/s11042-019-7424-8},
  year={2019}
}

"""

DATASET_PATH = Path("./dataset")
DATA_FILE_PATH = DATASET_PATH / "dataset.txt"
DATA_FOLDER_PATH = DATASET_PATH / "ring_dataset"

def load_path():
    """ return a dataset with shapoe (n, 2): path[n, 0] = hand with ring,
    path [n, 1] = ring
    """
    path = []

    with open(DATA_FILE_PATH, "r") as f:
        for line in f:
            path.append(line.strip().split("\t")[::-1])
    return np.asarray(path)

def load_full_path():
    """ return a dataset with shapoe (n, 2): path[n, 0] = hand with ring,
    path [n, 1] = ring
    """
    path = []

    with open(DATA_FILE_PATH, "r") as f:
        for line in f:
            tmp = line.strip().split("\t")[::-1]
            tmp = [
                str(DATA_FOLDER_PATH / tmp[0][:-12]), 
                str(DATA_FOLDER_PATH / tmp[1][:-5])
            ]
            path.append(tmp)
    return np.asarray(path)

paths = load_full_path()


# %%

# eval
r = random.randint(0, paths.shape[0] - 1)
# Return np repre of the img. Range [0, 255]. dtype int
hand = load_image(paths[r, 0])
# Shape IMG_SHAPE. Range[0, 1]
hand = preprocess_image(hand)
show_img(hand)
# Return np repre of the img. Range [0, 255]. dtype int
ring = load_image(paths[r, 1])
# Shape IMG_SHAPE. Range[0, 1]
ring = preprocess_image(ring)
show_img(ring)



# Eval hand pose (media pipe)

pose_img, pose = get_pose_map(paths[r, 0])
show_img(pose)

# Eval hand segmentation (models)
# prediction = model.predict(tf.expand_dims(hand, axis=0))
# plot_parsing_map(prediction)

# %%

# TODO: Train a simple u net that generate the same image.

# TODO: Train a network: Input: hand-representation: pose and hand segmentation, 
# ring. 
# out: img with ring?



# %%
