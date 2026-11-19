import pytest
import pandas as pd

from src import config


@pytest.fixture
def real_sample_csv(tmp_path):
    df_full = pd.read_csv(config.DATA_PATH, sep=";")
    df_test = df_full.head(200)
    path = tmp_path / "sample.csv"
    df_test.to_csv(path, sep=";", index=False)
    return path
