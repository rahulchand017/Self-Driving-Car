import os
import cv2
import random
import numpy as np
from pathlib import Path


# dataset path - change this based on where you are running
# on colab set this to your mounted drive path
# on windows use your D: drive path
DATASET_DIR = os.environ.get(
    "DRIVING_DATASET_DIR",
    str(Path(__file__).resolve().parent.parent.parent / "data" / "driving_dataset")
)

DATA_TXT = os.path.join(DATASET_DIR, "data.txt")


xs = []
ys = []

# read all lines from data.txt
# format of each line: "filename.jpg steering_angle_in_degrees"
with open(DATA_TXT, "r") as f:
    for line in f:
        parts = line.strip().split()
        if not parts:
            continue
        filename = parts[0]
        angle_deg = float(parts[1])

        img_path = os.path.join(DATASET_DIR, filename)
        xs.append(img_path)
        # convert degrees to radians because nvidia paper uses radians
        ys.append(angle_deg * np.pi / 180.0)


num_images = len(xs)
print(f"found {num_images} images in dataset")


# preload all images into memory (approach 2)
# each image is cropped to bottom 150 rows to remove sky
# then resized to 200x66 which matches nvidia pilotnet input
# and normalized to 0-1 range
print("preloading all images into memory, this takes a few minutes...")
images_in_memory = np.zeros((num_images, 66, 200, 3), dtype=np.float32)
angles_in_memory = np.zeros((num_images, 1), dtype=np.float32)

for i in range(num_images):
    img = cv2.imread(xs[i])
    if img is None:
        print(f"warning: could not load {xs[i]}")
        continue
    # crop bottom 150 rows (removes sky, keeps road)
    img_cropped = img[-150:]
    # resize to 200x66 as per nvidia paper
    img_resized = cv2.resize(img_cropped, (200, 66))
    # normalize to 0-1
    images_in_memory[i] = img_resized / 255.0
    angles_in_memory[i] = ys[i]

    if i % 5000 == 0 and i > 0:
        print(f"loaded {i}/{num_images} images")

print("done preloading")


# shuffle the indices, very important for training
# without shuffle, we would see thousands of straight-driving frames in a row
# and the model would forget how to turn
indices = list(range(num_images))
random.seed(42)
random.shuffle(indices)

# 80-20 train val split
split = int(num_images * 0.8)
train_indices = indices[:split]
val_indices = indices[split:]

num_train_images = len(train_indices)
num_val_images = len(val_indices)

print(f"train images: {num_train_images}, val images: {num_val_images}")


# batch pointers move through the shuffled indices
train_batch_pointer = 0
val_batch_pointer = 0


def LoadTrainBatch(batch_size):
    global train_batch_pointer
    x_out = []
    y_out = []
    for i in range(batch_size):
        idx = train_indices[(train_batch_pointer + i) % num_train_images]
        x_out.append(images_in_memory[idx])
        y_out.append(angles_in_memory[idx])
    train_batch_pointer += batch_size
    return x_out, y_out


def LoadValBatch(batch_size):
    global val_batch_pointer
    x_out = []
    y_out = []
    for i in range(batch_size):
        idx = val_indices[(val_batch_pointer + i) % num_val_images]
        x_out.append(images_in_memory[idx])
        y_out.append(angles_in_memory[idx])
    val_batch_pointer += batch_size
    return x_out, y_out
