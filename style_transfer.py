# %% 

import pandas as pd
import numpy as np
from pathlib import Path
import random

import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model

from core.utils import *
from core.models import *

DATASET_PATH = Path("./dataset")
DATA_FILE_PATH = DATASET_PATH / "dataset.txt"
DATA_FOLDER_PATH = DATASET_PATH / "ring_dataset"

IMG_SHAPE = (256, 192, 3)
BATCH_SIZE = 8

STEP_PER_EPOCHS = 40

def load_full_path():
    """ return a dataset with shapoe (n): path[n] = hand with ring,
    """
    path = []

    with open(DATA_FILE_PATH, "r") as f:
        for line in f:
            tmp = line.strip().split("\t")[::-1]
            tmp = str(DATA_FOLDER_PATH / tmp[0][:-12])
            path.append(tmp)
    return np.asarray(path)

paths = load_full_path()

# %%

def train_gen():
    for p in paths:
        img = tf.convert_to_tensor(np.asarray(Image.open(p)))
        img = preprocess_image(img, shape=IMG_SHAPE[:2])
        yield img, img

train_ds = tf.data.Dataset.from_generator(
    train_gen,
    output_signature=(
        tf.TensorSpec(IMG_SHAPE),
        tf.TensorSpec(IMG_SHAPE)
    )
).prefetch(tf.data.experimental.AUTOTUNE)

train_batch_ds = train_ds.batch(BATCH_SIZE)
it = iter(train_ds)
# %%

sample_input, sample_output = next(it)
print(f"Sample input shape: {sample_input.shape}")
print(f"input's range value: [{tf.reduce_min(sample_input)}, {tf.reduce_max(sample_input)}]")
show_img(sample_input)
print(f"Sample output shape: {sample_output.shape}")
print(f"output's range value: [{tf.reduce_min(sample_output)}, {tf.reduce_max(sample_output)}]")
show_img(sample_output)


# %%

# model = get_simple_unet_model()
model = get_res_unet_model()
# %%
optimizer = tf.optimizers.Adam(learning_rate=0.001)

# %%


model.summary()

# %%

# Loss operation

vgg19 = tf.keras.applications.VGG19(
    include_top=False, 
    weights='imagenet',
    input_shape=IMG_SHAPE
)
vgg19.trainable = False
# vgg16.summary()
layer_names = [
    'block1_conv2', # 256 x 192 x 64
    'block2_conv2', # 128 x 96 x 128
    'block3_conv2', # 64 x 48 x 256
    'block4_conv2', # 32 x 24 x 512
    'block5_conv2'  # 16 x 12 x 512
]
layers = [vgg19.get_layer(name).output for name in layer_names]

# Create the feature extraction model
wrap_vgg19_model = tf.keras.Model(inputs=vgg19.input, outputs=layers)
wrap_vgg19_model.trainable = False
wrap_vgg19_model.summary()
def loss_function(real, pred):

    # compute perceptual loss
    content_loss = compute_mse_loss(real, pred)

    # compute perceptual loss
    out_pred = wrap_vgg19_model(pred, training=False)
    out_real = wrap_vgg19_model(real, training=False)

    p1 = compute_mse_loss(out_real[0], out_pred[0]) / 5.3 * 2.5
    p2 = compute_mse_loss(out_real[1], out_pred[1]) / 2.7  / 1.2
    p3 = compute_mse_loss(out_real[2], out_pred[2]) / 1.35 / 2.3
    p4 = compute_mse_loss(out_real[3], out_pred[3]) / 0.67 / 8.2
    p5 = compute_mse_loss(out_real[4], out_pred[4]) / 0.16 

    perceptual_loss = (p1 + p2 + p3 + p4 + p5)  / 5.0 / 128.0

    return 1.0 * content_loss + 3.0 * perceptual_loss

# %%

checkpoint_path = "models/checkpoints/plos.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)
if os.path.exists("models/checkpoints/plos.ckpt.index"):
    model.load_weights(checkpoint_path)
    print("Weights loaded!")


# %%

EPOCHS = 1
with tf.device('/device:CPU:0'):
    for epoch in range(EPOCHS):
        print(f"Starting epoch {epoch + 1}")
        for step, (input_batch, output_batch) in enumerate(train_batch_ds.take(STEP_PER_EPOCHS)):
            with tf.GradientTape() as tape, tf.device('/device:GPU:0'):
                pred_batch = model(input_batch, training=True)
                # print(f"Input batch shape : {input_batch.shape}")
                # print(f"Output batch shape : {output_batch.shape}")
                r = random.randint(0, BATCH_SIZE - 1)
                print("Sample Input:")
                show_img(input_batch[r])
                print("Sample Prediction:")
                show_img(pred_batch[r])

                # Compute loss
                loss = loss_function(output_batch, pred_batch)
                print(f"loss for this batch at step: {step + 1}: {loss}")

            
            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
        # Save check point for each epoch    
        model.save_weights(checkpoint_path)
        print("Checkpoint saved!")

# %%
