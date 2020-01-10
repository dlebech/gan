"""Command-line interface."""
import argparse
import logging
import os
import shutil


def dataset_run(args):
    from gan import dataset

    data_dir = args.data_dir
    width = args.width
    height = width
    channels = args.channels
    num_images = args.num_images

    os.makedirs(data_dir, exist_ok=True)

    if args.dataset_name == "shapes":
        # Create shapes in a sub-directory to the data dir
        data_dir = os.path.join(data_dir, "shapes")
        shutil.rmtree(data_dir, ignore_errors=True)
        dataset.create_shapes(
            width, height, num_images, channels=channels, data_dir=data_dir
        )
    elif args.dataset_name == "cartoon":
        tar_filename = os.path.join(data_dir, "cartoon.tgz")
        data_dir = os.path.join(data_dir, "cartoon")
        dataset.prepare_cartoon(data_dir, tar_filename)


def train_run(args):
    from gan import train

    data_dir = args.data_dir
    width = args.width
    height = width
    channels = args.channels
    epochs = args.epochs
    batch_size = args.batch_size

    train.train(data_dir, width, height, channels, epochs, batch_size)


def generate_run(args):
    from gan import generate

    print(args)
    generator_filename = args.generator_filename
    output_dir = args.output_dir
    num_images = args.num_images

    generate.generate_image(generator_filename, output_dir, num_images=num_images)


def cli():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    # Create first-level subcommand parsers
    dataset = subparsers.add_parser(
        "dataset",
        help="Dataset related commands",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    train = subparsers.add_parser(
        "train",
        help="Train related commands",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    generate = subparsers.add_parser(
        "generate",
        help="Generation related commands",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Datasources for data preparation.
    dataset.add_argument(
        "dataset_name", help="Name of the dataset", choices=["cartoon", "shapes"]
    )
    dataset.add_argument(
        "-d",
        "--data-dir",
        default="data",
        help="Location of data files. Datasets will be created in sub-directories.",
    )
    dataset.add_argument(
        "-w",
        "--width",
        type=int,
        default=64,
        help="Width of image, only relevant when creating images. Height will be same as width",
    )
    dataset.add_argument(
        "-c",
        "--channels",
        type=int,
        default=3,
        help="Number of color channels, only relevant when creating images",
    )
    dataset.add_argument(
        "-n",
        "--num-images",
        type=int,
        default=2000,
        help="Number of images to create, only relevant when creating images",
    )
    dataset.set_defaults(func=dataset_run)

    train.add_argument(
        "-d", "--data-dir", default="data/shapes", help="Location of data files.",
    )
    train.add_argument(
        "-w",
        "--width",
        type=int,
        default=64,
        help="Width of image. Images will be resized to this width, and height will be the same as width",
    )
    train.add_argument(
        "-c", "--channels", type=int, default=3, help="Number of color channels.",
    )
    train.add_argument(
        "--epochs", type=int, default=50, help="Maximum number of epochs to train for.",
    )
    train.add_argument(
        "--batch-size", type=int, default=128, help="Batch size",
    )
    train.set_defaults(func=train_run)

    generate.add_argument(
        "generator_filename", help="Path to generator model",
    )
    generate.add_argument(
        "output_dir", help="Path to output directory",
    )
    generate.add_argument(
        "--num-images", type=int, default=20, help="Number of images to generate",
    )
    generate.set_defaults(func=generate_run)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cli()
