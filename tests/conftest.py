import shutil
import tempfile

import pytest


@pytest.fixture
def data_dir():
    temp_dir = tempfile.mkdtemp(prefix="gan_data_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)
