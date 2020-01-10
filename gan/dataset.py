# Parts of this file are modified from Tensorflow tutorial, licensed under the
# Apache License, Version 2.0, https://www.apache.org/licenses/LICENSE-2.0
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
"""Dataset download and creation, as well as other data-related functions."""
import functools
import logging
import os
import pathlib
import shutil
import tarfile
from typing import Callable

import cv2
import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


def _save_images(images, data_dir, class_name):
    class_dir = os.path.join(data_dir, class_name)
    os.makedirs(class_dir, exist_ok=True)
    for i, img in enumerate(images):
        cv2.imwrite(f"{class_dir}/{i}.png", img)


# Create a bunch of shapes and save them.
def create_shapes(width, height, num_images, channels=3, data_dir="data"):
    """Creates basic shapes with the given width and height."""
    # Add an alpha channel if we are in color. It's not going to matter for
    # Tensorflow, but it's nice anyway.
    if channels == 3:
        channels += 1
    size = (width, height, channels)
    center = (width // 2, height // 2)

    # For now, only create ellipses
    logger.info(f"Creating {num_images} images with size {size}")
    ellipses = [
        cv2.ellipse(
            np.full(size, 0, dtype=int),
            center,
            (
                np.random.randint(width // 4, width // 2),
                np.random.randint(width // 4, width // 2),
            ),
            np.random.randint(0, 180),  # angle
            # start angle and end angle should always be 0 and 360 so we get full
            # circles
            0,
            360,
            255
            if channels == 1
            else (
                np.random.randint(0, 256),
                np.random.randint(0, 256),
                np.random.randint(0, 256),
                255,
            ),
            thickness=-1,  # -1 thickness fills the shape
        )
        for _ in range(num_images)
    ]

    logger.info("Saving images to disk")
    _save_images(ellipses, data_dir, "ellipse")


def prepare_cartoon(data_dir, tar_filename):
    if not os.path.exists(tar_filename):
        raise Exception("Tarfile does not exist")

    extract_dir = os.path.join(data_dir, "__tmp")

    # All images will be stored as "face" class label, no matter what their
    # attributes are.
    class_dir = os.path.join(data_dir, "face")
    os.makedirs(class_dir, exist_ok=True)

    logger.info("Extracting cartoon tarfile")
    with tarfile.open(tar_filename) as f:
        f.extractall(extract_dir)

    path = pathlib.Path(extract_dir)
    logging.info("Saving images")
    for image_path in path.rglob("*.png"):
        image_path.rename(os.path.join(class_dir, image_path.name))

    logging.info("Saving metadata")
    for csv_path in path.rglob("*.csv"):
        csv_path.rename(os.path.join(class_dir, csv_path.name))

    shutil.rmtree(extract_dir, ignore_errors=True)


def _get_label(class_names, file_path):
    # Convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)

    # The second to last is the class-directory. This returns a one-hot
    # encoded array with class labels
    return parts[-2] == class_names


def _decode_img(width, height, img):
    # Convert the compressed string to a uint8 tensor
    img = tf.image.decode_png(img)

    # If we have 4 channels, ignore the alpha channel for now.
    if tf.shape(img)[2] == 4:
        img = img[:, :, :3]

    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)

    # resize the image to the desired size.
    img = tf.image.resize(img, [width, height])

    # Rescale to be between -1 and 1
    return (img - 0.5) / 0.5


def _process_path(get_label: Callable, decode_img: Callable, file_path: str):
    label = get_label(file_path)
    # Load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label


def create_dataset(data_dir: str, width: int, height: int):
    # Prepare a dataset with the images.
    # According to the guide on loading image data, we should probably use tf.data
    # since this way of loading data is more efficient than e.g.
    # keras.preprocessing
    # https://www.tensorflow.org/tutorials/load_data/images

    # Find class names based on directory structure
    data_dir = pathlib.Path(data_dir)
    class_names = np.array(sorted([item.name for item in data_dir.glob("*")]))

    logger.info(f"Found class names: {class_names}")

    # Create helper functions
    get_label = functools.partial(_get_label, class_names)
    decode_img = functools.partial(_decode_img, width, height)
    process_path = functools.partial(_process_path, get_label, decode_img)

    # Make a tf.data.Dataset from the files in the directory.
    list_ds = tf.data.Dataset.list_files(
        [str(data_dir / "*/*.png"), str(data_dir / "*/*.jpg")]
    )
    labeled_ds = list_ds.map(
        process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    return labeled_ds, class_names
