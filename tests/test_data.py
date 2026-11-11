import pytest
import pandas as pd

from src.data import load_data, split_data
from src import config


@pytest.fixture
def sample_csv(tmp_path):
    df = pd.DataFrame(
        {
            "Course\t": [171, 9238, 9773, 9070, 171],
            " Admission grade": [124.6, 116.4, 121.4, 117.6, 110.0],
            "Gender  ": [0, 0, 1, 0, 1],
            "Target": ["Dropout", "Graduate", "Enrolled", "Graduate", "Dropout"],
        }
    )
    path = tmp_path / "sample.csv"
    df.to_csv(path, sep=";", index=False)
    return path


def test_load_data_strips_column_names(sample_csv):
    df = load_data(sample_csv, target="Target")
    assert list(df.columns) == [
        "Course",
        "Admission grade",
        "Gender",
        "Target",
    ]


def test_load_data_maps_dropout_to_1(sample_csv):
    df = load_data(sample_csv, target="Target")
    assert df["Target"].tolist() == [1, 0, 0, 1]


def test_load_data_drops_enrolled(sample_csv):
    df = load_data(sample_csv, target="Target")
    assert len(df) == 4


@pytest.fixture
def splits():
    df = load_data(config.DATA_PATH, config.TARGET)
    X_train, X_test, y_train, y_test = split_data(
        df=df,
        features=config.ENROLLMENT_MODEL_FEATURES,
        target=config.TARGET,
        random_state=config.RANDOM_STATE,
        test_size=config.TEST_SIZE,
    )
    return df, X_train, X_test, y_train, y_test


def test_split_data_keeps_all_rows(splits):
    df, X_train, X_test, y_train, y_test = splits
    assert len(df) == len(X_train) + len(X_test)
    assert len(df) == len(y_train) + len(y_test)


def test_split_data_has_no_overlap(splits):
    df, X_train, X_test, y_train, y_test = splits
    for i in df.index:
        assert (i in X_train.index) != (i in X_test.index)
        assert (i in y_train.index) != (i in y_test.index)


def test_split_data_uses_model_features(splits):
    df, X_train, X_test, y_train, y_test = splits
    assert X_train.columns.to_list() == config.ENROLLMENT_MODEL_FEATURES
    assert X_test.columns.to_list() == config.ENROLLMENT_MODEL_FEATURES


def test_split_data_test_size(splits):
    df, X_train, X_test, y_train, y_test = splits
    assert len(X_test) == pytest.approx(config.TEST_SIZE * len(df), abs=1.0)


def test_split_data_stratification(splits):
    df, X_train, X_test, y_train, y_test = splits
    assert y_train.value_counts(normalize=True)[1] == pytest.approx(
        df["Target"].value_counts(normalize=True)[1], rel=0.01
    )

    assert y_test.value_counts(normalize=True)[1] == pytest.approx(
        df["Target"].value_counts(normalize=True)[1], rel=0.01
    )


def test_split_data_x_and_y_are_aligned(splits):
    df, X_train, X_test, y_train, y_test = splits
    assert X_train.index.equals(y_train.index)
    assert X_test.index.equals(y_test.index)
