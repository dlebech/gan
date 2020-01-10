# Parts of this file are modified from Tensorflow tutorial, licensed under the
# Apache License, Version 2.0, https://www.apache.org/licenses/LICENSE-2.0
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
"""Utility functions"""
import glob

import imageio


def generate_gif(gif_file, image_dir):
    """Search the given output directory for image files and stitch them together into a GIF."""
    with imageio.get_writer(gif_file, mode="I") as writer:
        filenames = glob.glob(f"{image_dir}/image*.png")
        filenames = sorted(filenames)
        last = -1
        for i, filename in enumerate(filenames):
            frame = 2 * (i ** 0.5)
            if round(frame) > round(last):
                last = frame
            else:
                continue
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)
