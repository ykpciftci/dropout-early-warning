from pathlib import Path

import pytest


@pytest.fixture
def real_sample_csv():
    return Path(__file__).parent / "data" / "sample.csv"
