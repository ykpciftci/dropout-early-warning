import pytest
from fastapi.testclient import TestClient

from src import config
from src.api import app, get_enrollment_model, get_semester_1_model
from src.train import build_pipeline


def train_small_model(df, features, target):
    X = df[features]
    y = df[target]
    numeric_features = [f for f in config.NUMERICAL_FEATURES if f in X.columns]
    categoric_features = [f for f in config.CATEGORICAL_FEATURES if f in X.columns]
    small_model = build_pipeline(
        numerical=numeric_features, categorical=categoric_features
    )
    return small_model.fit(X, y)


@pytest.fixture
def enrollment_small_model(sample_df):
    enrollment_model = train_small_model(
        df=sample_df, features=config.ENROLLMENT_MODEL_FEATURES, target=config.TARGET
    )
    return enrollment_model


@pytest.fixture
def semester_1_small_model(sample_df):
    semester_1_model = train_small_model(
        df=sample_df, features=config.SEMESTER_1_MODEL_FEATURES, target=config.TARGET
    )
    return semester_1_model


@pytest.fixture
def client(enrollment_small_model, semester_1_small_model):
    app.dependency_overrides[get_enrollment_model] = lambda: enrollment_small_model
    app.dependency_overrides[get_semester_1_model] = lambda: semester_1_small_model

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()


@pytest.fixture
def enrollment_body(sample_df):
    X = sample_df[config.ENROLLMENT_MODEL_FEATURES]
    return X.iloc[0].to_dict()


@pytest.fixture
def semester_1_body(sample_df):
    X = sample_df[config.SEMESTER_1_MODEL_FEATURES]
    return X.iloc[0].to_dict()


def test_predict_enrollment_valid_student(client, enrollment_body):
    response = client.post("predict/enrollment", json=enrollment_body)
    assert response.status_code == 200
    assert response.json().keys() == {"dropout", "dropout_risk_score"}
    assert (
        response.json().get("dropout_risk_score") > 0
        and response.json().get("dropout_risk_score") < 1
    )


def test_predict_semester_1_valid_student(client, semester_1_body):
    response = client.post("predict/semester-1", json=semester_1_body)
    assert response.status_code == 200
    assert response.json().keys() == {"dropout", "dropout_risk_score"}
    score = response.json()["dropout_risk_score"]
    assert 0 < score < 1


def test_predict_enrollment_missing_field(client, enrollment_body):
    bad_json = enrollment_body.copy()
    bad_json.pop("Course")
    response = client.post("/predict/enrollment", json=bad_json)
    assert response.status_code == 422


def test_predict_semester_1_missing_field(client, semester_1_body):
    bad_json = semester_1_body.copy()
    bad_json.pop("Curricular units 1st sem (without evaluations)")
    response = client.post("/predict/semester-1", json=bad_json)
    assert response.status_code == 422


def test_predict_enrollment_wrong_type(client, enrollment_body):
    bad_json = enrollment_body.copy()
    bad_json["Admission grade"] = "forty two"
    response = client.post("/predict/enrollment", json=bad_json)
    assert response.status_code == 422


def test_predict_enrollment_out_of_range(client, enrollment_body):
    bad_json = enrollment_body.copy()
    bad_json["Age at enrollment"] = 12
    response = client.post("/predict/enrollment", json=bad_json)
    assert response.status_code == 422


def test_predict_semester_1_out_of_range(client, semester_1_body):
    bad_json = semester_1_body.copy()
    bad_json["Curricular units 1st sem (grade)"] = 25
    response = client.post("/predict/semester-1", json=bad_json)
    assert response.status_code == 422


def test_predict_enrollment_unseen_category(client, enrollment_body):
    new_json = enrollment_body.copy()
    new_json["Course"] = 999999
    response = client.post("/predict/enrollment", json=new_json)
    assert response.status_code == 200
