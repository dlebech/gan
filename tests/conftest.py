import shutil
import tempfile

import pytest


@pytest.fixture
def data_dir():
    data_dir = tempfile.mkdtemp(prefix="gan_avatar_data_")
    yield data_dir
    shutil.rmtree(data_dir, ignore_errors=True)
