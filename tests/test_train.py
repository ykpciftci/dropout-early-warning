import pytest
import pandas as pd

from src import config
from src.data import load_data, split_data
from src.train import build_pipeline


@pytest.fixture
def sample_csv(tmp_path):
    df_full = pd.read_csv(config.DATA_PATH, sep=";")
    df_test = df_full.head(200)
    path = tmp_path / "sample.csv"
    df_test.to_csv(path, sep=";", index=False)
    return path


@pytest.fixture
def training_data(sample_csv):

    df = load_data(sample_csv, target=config.TARGET)
    X_train, X_test, y_train, y_test = split_data(
        df=df,
        features=config.ENROLLMENT_MODEL_FEATURES,
        target=config.TARGET,
        random_state=config.RANDOM_STATE,
        test_size=config.TEST_SIZE,
    )

    return X_train, X_test, y_train, y_test


@pytest.fixture
def model_fitted(training_data):
    X_train, _, y_train, _ = training_data
    numeric_features = [f for f in config.NUMERICAL_FEATURES if f in X_train.columns]
    categoric_features = [
        f for f in config.CATEGORICAL_FEATURES if f in X_train.columns
    ]
    model = build_pipeline(numerical=numeric_features, categorical=categoric_features)
    return model.fit(X_train, y_train)


def test_train_pipeline_predicts_only_0_1(model_fitted, training_data):
    _, X_test, _, _ = training_data
    y_pred = model_fitted.predict(X_test)
    for p in y_pred.tolist():
        assert p in [0, 1]


def test_train_pipeline_for_one_student(model_fitted, training_data):
    _, X_test, _, _ = training_data
    single = X_test.iloc[[0]]
    pred = model_fitted.predict(single)
    assert len(pred) == 1
    assert pred[0] in [0, 1]


def test_train_unseen_category_not_crash(model_fitted, training_data):
    _, X_test, _, _ = training_data
    student = X_test.iloc[[0]].copy()
    student["Course"] = 99999
    pred = model_fitted.predict(student)
    for p in list(pred):
        assert p in [0, 1]
