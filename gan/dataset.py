# Parts of this file are modified from Tensorflow tutorial, licensed under the
# Apache License, Version 2.0, https://www.apache.org/licenses/LICENSE-2.0
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
"""Dataset download and creation, as well as other data-related functions."""
import collections
import functools
import json
import logging
import os
import pathlib
import shutil
import tarfile
import zipfile
from typing import Callable

import cv2
import numpy as np
import tensorflow as tf

from gan import util

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
    """Prepare the cartoon avatar dataset for training."""
    if not os.path.exists(tar_filename):
        raise Exception("Tarfile does not exist")

    extract_dir = os.path.join(data_dir, "__tmp")

    # All images will be stored as "face" class label, no matter what their
    # attributes are.
    class_dir = os.path.join(data_dir, "face")
    os.makedirs(class_dir, exist_ok=True)

    logger.info("Extracting cartoon tarfile")
    with tarfile.open(tar_filename) as f:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(f, extract_dir)

    path = pathlib.Path(extract_dir)
    logger.info("Saving images")
    for image_path in path.rglob("*.png"):
        image_path.rename(os.path.join(class_dir, image_path.name))

    logger.info("Saving metadata")
    for csv_path in path.rglob("*.csv"):
        csv_path.rename(os.path.join(class_dir, csv_path.name))

    shutil.rmtree(extract_dir, ignore_errors=True)


def prepare_coco(image_dir, text_dir, zip_dir):
    """Prepare the Coco dataset.

    Only uses the validation dataset from 2017 (since it has a reasonable size).

    """
    # https://cocodataset.org/#download
    # Everything velow is hardcoded for the validation set for 2017 due to its
    # limited file size.
    images_zip_url = "http://images.cocodataset.org/zips/val2017.zip"
    annotations_zip_url = (
        "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    )
    instances_json_filename = "instances_val2017.json"
    captions_json_filename = "captions_val2017.json"
    file_source_dir = "val2017"

    images_zip_filename = os.path.join(zip_dir, os.path.basename(images_zip_url))
    annotations_zip_filename = os.path.join(
        zip_dir, os.path.basename(annotations_zip_url)
    )

    if not os.path.exists(annotations_zip_filename):
        logger.info(f"Annotations not found, downloading to {annotations_zip_filename}")
        util.download_file(annotations_zip_url, annotations_zip_filename)
    else:
        logger.info(f"Annotations zip found at {annotations_zip_filename}")

    if not os.path.exists(images_zip_filename):
        logger.info(f"Images not found, downloading to {images_zip_filename}")
        util.download_file(images_zip_url, images_zip_filename)
    else:
        logger.info(f"Images zip found at {images_zip_filename}")

    extract_dir = os.path.join(zip_dir, "__tmp")

    logger.info("Extracting images zipfile")
    with zipfile.ZipFile(images_zip_filename) as f:
        f.extractall(extract_dir)

    logger.info("Extracting annotations zipfile")
    with zipfile.ZipFile(annotations_zip_filename) as f:
        f.extractall(extract_dir)

    # Move images to their appropriate place according to their annotations
    # This will cause duplicates if there are multiple images with the same
    # class label.
    logger.info("Preparing annotations")
    with open(os.path.join(extract_dir, "annotations", instances_json_filename)) as f:
        instances = json.load(f)

    with open(os.path.join(extract_dir, "annotations", captions_json_filename)) as f:
        captions = json.load(f)

    category_lookup = {
        c["id"]: (c["name"], c["supercategory"]) for c in instances["categories"]
    }

    image_lookup = {i["id"]: i["file_name"] for i in instances["images"]}

    logger.info("Copying images")
    for annotation in instances["annotations"]:
        image_id = annotation["image_id"]
        category_id = annotation["category_id"]
        category_name, _ = category_lookup[category_id]
        file_source = os.path.join(extract_dir, file_source_dir, image_lookup[image_id])
        ext = os.path.splitext(file_source)[1]
        dir_destination = os.path.join(image_dir, category_name, f"{image_id}{ext}")
        os.makedirs(os.path.dirname(dir_destination), exist_ok=True)
        shutil.copy(file_source, dir_destination)

    grouped_captions = collections.defaultdict(list)
    for annotation in captions["annotations"]:
        grouped_captions[annotation["image_id"]].append(annotation["caption"])

    os.makedirs(text_dir, exist_ok=True)

    logger.info("Copying captions")
    for image_id, captions in grouped_captions.items():
        with open(os.path.join(text_dir, f"{image_id}.txt"), "w") as f:
            f.write("\n".join(captions))

    shutil.rmtree(extract_dir, ignore_errors=True)


def _get_label(class_names, file_path):
    # Convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)

    # The second to last is the class-directory. This returns a one-hot
    # encoded array with class labels
    return parts[-2] == class_names


def _decode_img(width, height, channels, img):
    # Convert the compressed string to a uint8 tensor
    img = tf.image.decode_png(img)

    # If we have 4 channels, ignore the alpha channel for now.
    if tf.shape(img)[2] == 4:
        img = img[:, :, :3]

    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)

    # Resize the image to the desired size.
    img = tf.image.resize(img, [width, height])

    # If we have 1 channel but require 3, convert to rgb!
    if channels == 3 and tf.shape(img)[2] == 1:
        img = tf.image.grayscale_to_rgb(img)

    # Rescale to be between -1 and 1
    return (img - 0.5) / 0.5


def _process_path(get_label: Callable, decode_img: Callable, file_path: str):
    label = get_label(file_path)
    # Load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label


def create_dataset(data_dir: str, width: int, height: int, channels: int):
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
    decode_img = functools.partial(_decode_img, width, height, channels)
    process_path = functools.partial(_process_path, get_label, decode_img)

    # Make a tf.data.Dataset from the files in the directory.
    list_ds = tf.data.Dataset.list_files(
        [str(data_dir / "*/*.png"), str(data_dir / "*/*.jpg")]
    )
    labeled_ds = list_ds.map(
        process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    return labeled_ds, class_names
