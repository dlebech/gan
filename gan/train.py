"""GAN training."""
import datetime
import logging
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from gan import dataset, util

logger = logging.getLogger(__name__)

physical_devices = tf.config.experimental.list_physical_devices("GPU")
try:
    # Assert we have GPU
    assert len(physical_devices) > 0

    # This prevents cuDNN errors when training on GPU for some reason...
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except AssertionError as e:
    logger.warning("No GPU found")


def _upsample(x, units, kernel_initializer, leaky_alpha):
    x = layers.Conv2DTranspose(
        units,
        (5, 5),
        strides=(2, 2),
        padding="same",
        use_bias=False,
        kernel_initializer=kernel_initializer,
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(leaky_alpha)(x)
    return x


def create_generator_model(
    noise_dim: int, width: int, height: int, channels: int = 3, leaky_alpha: float = 0.2
):
    # Through a series of layers, a 100-dimensional seed is converted to a
    # small image and then upsampled to the full width and height.
    gen_rows = height // 16
    gen_cols = width // 16
    rows = height
    cols = width
    initializer = tf.random_normal_initializer(mean=0.0, stddev=0.02)

    inp = keras.Input(shape=(noise_dim,))

    # Create a flat feature vector
    x = layers.Dense(gen_rows * gen_cols * 1024, use_bias=False)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(leaky_alpha)(x)

    # Reshape to look like a small image with 1024 features.
    x = layers.Reshape((gen_rows, gen_cols, 1024))(x)

    # Do a series of upsampling layers
    x = _upsample(x, 512, initializer, leaky_alpha)
    x = _upsample(x, 256, initializer, leaky_alpha)
    x = _upsample(x, 128, initializer, leaky_alpha)

    # The final upsampling layer needs a tanh activation.
    x = layers.Conv2DTranspose(
        channels,
        (5, 5),
        strides=(2, 2),
        padding="same",
        use_bias=False,
        activation="tanh",
        kernel_initializer=initializer,
    )(x)
    assert x.shape.as_list() == [None, rows, cols, channels]

    return keras.Model(inputs=inp, outputs=x)


def create_discriminator_model(
    width: int, height: int, channels: int = 3, leaky_alpha: float = 0.2
):
    rows = height
    cols = width
    initializer = tf.random_normal_initializer(mean=0.0, stddev=0.02)

    inp = keras.Input(shape=(rows, cols, channels))

    x = layers.Conv2D(
        64, (5, 5), strides=(2, 2), padding="same", kernel_initializer=initializer
    )(inp)
    x = layers.LeakyReLU(leaky_alpha)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(
        128, (5, 5), strides=(2, 2), padding="same", kernel_initializer=initializer
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(leaky_alpha)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(
        256, (5, 5), strides=(2, 2), padding="same", kernel_initializer=initializer
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(leaky_alpha)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(1)(x)

    return keras.Model(inputs=inp, outputs=x)


def generate_and_save_images(model, epoch, test_input, output_dir):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    plt.figure(figsize=(8, 8))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        img = predictions[i]
        if img.shape[2] == 1:
            img = np.squeeze(img)
            plt.imshow(img * 0.5 + 0.5, cmap="gray")
        else:
            plt.imshow(img * 0.5 + 0.5)
        plt.axis("off")

    plt.savefig(os.path.join(output_dir, f"image_at_epoch_{epoch:04d}.png"))
    plt.close()


def train(data_dir, width, height, channels, epochs, batch_size, output_dir="output"):
    run_name = datetime.datetime.now().isoformat(timespec="seconds").replace(":", "")
    run_dir = os.path.join(output_dir, run_name)
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    image_dir = os.path.join(run_dir, "images")
    os.makedirs(checkpoint_dir)
    os.makedirs(image_dir)

    noise_dim = 100
    leaky_alpha = 0.2
    learning_rate = 0.0002
    momentum = 0.5

    # Create 16 random seeds with that stay the same during training.
    seed = tf.random.normal([16, noise_dim])

    ds, _ = dataset.create_dataset(data_dir, width, height, channels)
    train_ds = ds.shuffle(1000).batch(batch_size)

    generator = create_generator_model(
        noise_dim, width, height, channels=channels, leaky_alpha=leaky_alpha
    )
    generator.summary()

    discriminator = create_discriminator_model(
        width, height, channels=channels, leaky_alpha=leaky_alpha
    )
    discriminator.summary()

    cross_entropy = keras.losses.BinaryCrossentropy(from_logits=True)

    def discriminator_loss(real_output, fake_output):
        real_loss = cross_entropy(tf.fill(tf.shape(real_output), 0.9), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(fake_output):
        return cross_entropy(tf.ones_like(fake_output), fake_output)

    gen_opt = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=momentum)
    disc_opt = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=momentum)

    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(
        generator_optimizer=gen_opt,
        discriminator_optimizer=disc_opt,
        generator=generator,
        discriminator=discriminator,
    )

    # Notice the use of `tf.function`
    # This annotation causes the function to be "compiled".
    @tf.function
    def train_step(images):
        noise = tf.random.normal([batch_size, noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = generator(noise, training=True)

            real_output = discriminator(images, training=True)
            fake_output = discriminator(generated_images, training=True)

            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(
            gen_loss, generator.trainable_variables
        )
        gradients_of_discriminator = disc_tape.gradient(
            disc_loss, discriminator.trainable_variables
        )

        gen_opt.apply_gradients(
            zip(gradients_of_generator, generator.trainable_variables)
        )
        disc_opt.apply_gradients(
            zip(gradients_of_discriminator, discriminator.trainable_variables)
        )

        return gen_loss, disc_loss

    logger.info(
        f"Starting training at most {epochs} epochs with batch size {batch_size} and learning rate {learning_rate}"
    )

    for epoch in range(epochs):
        start = time.time()

        gen_losses = []
        disc_losses = []
        for batch in train_ds:
            # A batch consists of a tuple (image, label)
            image_batch, _ = batch
            gen_loss, disc_loss = train_step(image_batch)
            gen_losses.append(gen_loss.numpy())
            disc_losses.append(disc_loss.numpy())

        # Produce images for the GIF as we go
        generate_and_save_images(generator, epoch + 1, seed, image_dir)

        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
            generator.save(os.path.join(checkpoint_dir, "generator_model.h5"))
            discriminator.save(os.path.join(checkpoint_dir, "discriminator_model.h5"))

        logger.info(f"Time for epoch {epoch + 1} is {time.time() - start:,.2f} sec")
        logger.info(f"After {gen_opt.iterations.numpy()} steps:")
        logger.info(
            f"Generator loss (latest/epoch avg.): {gen_losses[-1]} / {np.mean(gen_losses)}"
        )
        logger.info(
            f"Discriminator loss (latest/epoch avg.): {disc_losses[-1]} / {np.mean(disc_losses)}"
        )

    util.generate_gif(os.path.join(run_dir, "dcgan.gif"), image_dir)
