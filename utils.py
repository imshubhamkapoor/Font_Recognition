from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
from PIL import ImageOps, Image
import os
import pathlib
from skimage import io
from tensorflow.keras import datasets, layers, models


AUTOTUNE = tf.data.experimental.AUTOTUNE

# Loading and cropping the images in dataset to fixed dimensions of 100 * 100 from the given path
def crop_dataset(path):
    for fontlist in os.listdir(path):
        for font in os.listdir(os.path.join(path, fontlist)):
            path_new = os.path.join(path, fontlist, font)
            img = Image.open(path_new)
            img = ImageOps.fit(img, (100, 100), method=0, bleed=0.0, centering=(0.5, 0.5))
            img.save(path_new)

# Loading and counting the total number of images and classes in dataset
def list_dataset(path):
    data = pathlib.Path(path)
    image_count = len(list(data.glob('*/*.jpg')))
    CLASS_NAMES = np.array([item.name for item in data.glob('*') if item.name != "LICENSE.txt"])
    return data, image_count, CLASS_NAMES

# Preparing the dataset for training
def load_dataset(dir, BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, CLASS_NAMES):
    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

    data_gen = image_generator.flow_from_directory(directory=str(dir),
                                                   batch_size=BATCH_SIZE,
                                                   shuffle=True,
                                                   target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                   classes=list(CLASS_NAMES))

    return data_gen

# Pre-processing the test image for prediction
def preprocess_test_image(path):
    img = io.imread(path, as_gray=False)
    crop_length = 100
    start_x = img.shape[0] / 2 - crop_length / 2
    start_y = img.shape[1] / 2 - crop_length / 2
    img = img[int(start_x): int(start_x + crop_length), int(start_y):int(start_y + crop_length)]
    img = np.divide(np.asarray(img), 255)
    return np.reshape(img, (1, 100, 100, 3))

# Defining input parameters required for the model
IMG_HEIGHT = 100
IMG_WIDTH = 100
OUTPUT_CLASSES = 100

# Defining the CNN model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.4))
model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.BatchNormalization())
model.add(layers.Flatten())
model.add(layers.Dense(500, activation='relu'))
model.add(layers.Dropout(0.4))
model.add(layers.Dense(OUTPUT_CLASSES, activation='softmax'))
