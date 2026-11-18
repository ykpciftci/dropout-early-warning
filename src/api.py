from fastapi import FastAPI, Depends
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from functools import lru_cache

from src import config

app = FastAPI()


class EnrollmentFeatures(BaseModel):
    marital_status: int = Field(alias="Marital status")
    application_mode: int = Field(alias="Application mode")
    application_order: int = Field(alias="Application order")
    course: int = Field(alias="Course")
    daytime_evening_attendance: int = Field(alias="Daytime/evening attendance")
    previous_qualification: int = Field(alias="Previous qualification")
    previous_qualification_grade: float = Field(
        alias="Previous qualification (grade)", ge=0.0, le=200.0
    )
    nationality: int = Field(alias="Nacionality")
    mothers_qualification: int = Field(alias="Mother's qualification")
    fathers_qualification: int = Field(alias="Father's qualification")
    mothers_occupation: int = Field(alias="Mother's occupation")
    fathers_occupation: int = Field(alias="Father's occupation")
    admission_grade: float = Field(alias="Admission grade", ge=0.0, le=200.0)
    displaced: int = Field(alias="Displaced")
    educational_special_needs: int = Field(alias="Educational special needs")
    gender: int = Field(alias="Gender")
    age_at_enrollment: int = Field(alias="Age at enrollment", ge=15, le=80)
    international: int = Field(alias="International")
    inflation_rate: float = Field(alias="Inflation rate", ge=-10, le=50)
    unemployment_rate: float = Field(alias="Unemployment rate", ge=0.0, le=20.0)
    gdp: float = Field(alias="GDP", ge=-15.0, le=17.0)


class Semester1Features(EnrollmentFeatures):
    first_sem_credited: int = Field(alias="Curricular units 1st sem (credited)", ge=0)
    first_sem_enrolled: int = Field(alias="Curricular units 1st sem (enrolled)", ge=0)
    first_sem_evaluations: int = Field(
        alias="Curricular units 1st sem (evaluations)", ge=0
    )
    first_sem_without_evaluations: int = Field(
        alias="Curricular units 1st sem (without evaluations)", ge=0
    )
    first_sem_approved: int = Field(alias="Curricular units 1st sem (approved)", ge=0)
    first_sem_grade: float = Field(
        alias="Curricular units 1st sem (grade)", ge=0, le=20
    )


@lru_cache
def get_enrollment_model():
    enrollment_model = joblib.load(config.ENROLLMENT_MODEL_PATH)
    return enrollment_model


@lru_cache
def get_semester_1_model():
    semester_1_model = joblib.load(config.SEMESTER_1_MODEL_PATH)
    return semester_1_model


def predict_dropout(model, features):
    X = pd.DataFrame([features.model_dump(by_alias=True)])
    probabilities = model.predict_proba(X)[0]
    return {
        "dropout": bool(probabilities[1] > 0.5),
        "dropout_risk_score": round(float(probabilities[1]), 3),
    }


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict/enrollment")
def predict_enrollment(
    enrollment_data: EnrollmentFeatures, model=Depends(get_enrollment_model)
):
    return predict_dropout(features=enrollment_data, model=model)


@app.post("/predict/semester-1")
def predict_semester_1(
    semester_1_data: Semester1Features, model=Depends(get_semester_1_model)
):
    return predict_dropout(features=semester_1_data, model=model)
