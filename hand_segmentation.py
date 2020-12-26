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
https://vision.uvic.ca/pubs/2019/bojja2019handseg/page.md
Reference:
http://vision.soic.indiana.edu/projects/egohands/
"""

DATASET_PATH = Path("./dataset/handseg-150k-20180914/handseg-150k")

IMAGE_FOLDER = DATASET_PATH / "images"
LABEL_FOLDER = DATASET_PATH / "masks"

def load_full_path():
    paths = []
    for img, label in zip(os.listdir(IMAGE_FOLDER), os.listdir(LABEL_FOLDER)):
        assert img == label, "Image and label file not matching!"
        paths.append([str(IMAGE_FOLDER / img), str(LABEL_FOLDER/ label)])
    return np.asarray(paths)

paths = load_full_path()


# %%

# eval
r = random.randint(0, paths.shape[0] - 1)
# Return np repre of the img. Range [0, 255]. dtype int
hand = load_image(paths[r, 0])
# Shape IMG_SHAPE. Range[0, 1]
hand = preprocess_image(tf.expand_dims(hand, axis=-1))
# show_img(hand)
# # Return np repre of the img. Range [0, 255]. dtype int
# ring = load_image(paths[r, 1])
# # Shape IMG_SHAPE. Range[0, 1]
# ring = preprocess_image(ring)
# show_img(ring)
# paths[r, 1]
# hand

# img = np.asarray(Image.open(paths[r, 0]))
# img = tf.cast(img, dtype=float)
# show_img(img)

# %%

show_img(hand)