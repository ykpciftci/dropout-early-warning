from pathlib import Path

import pytest

from src import config
from src.data import load_data


@pytest.fixture
def real_sample_csv():
    return Path(__file__).parent / "data" / "sample.csv"


@pytest.fixture
def sample_df(real_sample_csv):
    df = load_data(real_sample_csv, target=config.TARGET)
    return df
