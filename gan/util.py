# Parts of this file are modified from Tensorflow tutorial, licensed under the
# Apache License, Version 2.0, https://www.apache.org/licenses/LICENSE-2.0
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
"""Utility functions"""
import glob

import imageio
import requests
from tqdm import tqdm


def generate_gif(gif_file, image_dir):
    """Search the given output directory for image files and stitch them
    together into a GIF."""
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


def generate_mp4(mp4_file, image_dir):
    """Search the given output directory for image files and stitch them
    together into a GIF."""
    with imageio.get_writer(mp4_file, mode="I", fps=15) as writer:
        filenames = glob.glob(f"{image_dir}/image*.png")
        for filename in sorted(filenames):
            writer.append_data(imageio.imread(filename))


def download_file(url, filename):
    # https://stackoverflow.com/a/37573701/2021517
    # Streaming, so we can iterate over the response.
    r = requests.get(url, stream=True)

    # Total size in bytes.
    total_size = int(r.headers.get("content-length", 0))
    block_size = 1024  # 1 Kibibyte
    t = tqdm(total=total_size, unit="iB", unit_scale=True)
    with open(filename, "wb") as f:
        for data in r.iter_content(block_size):
            t.update(len(data))
            f.write(data)
    t.close()
    if total_size != 0 and t.n != total_size:
        print("ERROR, something went wrong")
