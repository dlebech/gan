import os

import imageio
import numpy as np

from gan import dataset


def _create_img(data_dir, class_name, color, include_alpha=False):
    img_dir = os.path.join(data_dir, class_name)
    img_file = os.path.join(img_dir, "0.png")
    os.makedirs(img_dir, exist_ok=True)
    img = [color, color, color]
    if include_alpha:
        img.append(255)
    img = np.array(img, dtype="uint8").reshape((1, 1, 4 if include_alpha else 3))
    imageio.imwrite(img_file, img)


def test_create_shapes(data_dir):
    """It should create color shapes in the given data directory."""
    dataset.create_shapes(10, 10, 1, data_dir=data_dir)
    img_path = os.path.join(data_dir, "ellipse/0.png")
    assert os.path.exists(img_path)
    img = imageio.imread(img_path)
    assert img.shape == (10, 10, 4)


def test_create_shapes_grayscale(data_dir):
    """It should create grayscale shapes in the given data directory."""
    dataset.create_shapes(10, 10, 1, channels=1, data_dir=data_dir)
    img_path = os.path.join(data_dir, "ellipse/0.png")
    assert os.path.exists(img_path)
    img = imageio.imread(img_path)
    assert img.shape == (10, 10)


def test_create_dataset(data_dir):
    _create_img(data_dir, "black", 0)
    _create_img(data_dir, "white", 255)
    _create_img(data_dir, "gray", 128, include_alpha=True)

    # Ensure it changes the value of pixels to -1 to 1
    black_expected = np.array([-1, -1, -1]).reshape((1, 1, 3))
    white_expected = np.array([1, 1, 1]).reshape((1, 1, 3))
    gray_expected = np.array([0, 0, 0]).reshape((1, 1, 3))

    # Create the dataset
    ds, class_names = dataset.create_dataset(data_dir, 1, 1)
    assert list(class_names) == ["black", "gray", "white"]

    # Find the first batch
    batch = next(iter(ds.batch(10)))
    image_batch, label_batch = batch
    image_batch = image_batch.numpy()
    label_batch = label_batch.numpy()

    # Make sure the batch consists of two elements
    assert len(image_batch) == 3
    assert len(label_batch) == 3

    for i, img in enumerate(image_batch):
        label = label_batch[i]
        # Black
        if list(label) == [True, False, False]:
            np.testing.assert_equal(img, black_expected)
        # Gray
        elif list(label) == [False, True, False]:
            np.testing.assert_almost_equal(img, gray_expected, decimal=2)
        # White
        elif list(label) == [False, False, True]:
            np.testing.assert_equal(img, white_expected)
        else:
            raise Exception("It should have a valid label")
