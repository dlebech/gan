"""Generate new images based on trained GAN."""
import logging
import os

import imageio
import tensorflow as tf
from tensorflow import keras

logger = logging.getLogger(__name__)

physical_devices = tf.config.experimental.list_physical_devices("GPU")
try:
    # Assert we have GPU
    assert len(physical_devices) > 0

    # This prevents cuDNN errors when training on GPU for some reason...
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except AssertionError as e:
    logger.warning("No GPU found")


def generate_image(generator_filename, image_dir, num_images=20, noise_dim=100):
    generator = keras.models.load_model(generator_filename)
    seed_batch = tf.random.normal([num_images, noise_dim])
    image_batch = generator(seed_batch, training=False)
    for i, image in enumerate(image_batch):
        image_filename = os.path.join(image_dir, f"{i}.png")
        imageio.imwrite(image_filename, image * 0.5 + 0.5)
