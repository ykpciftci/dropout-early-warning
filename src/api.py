from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd

from src import config

app = FastAPI()

enrollment_model = joblib.load(config.ENROLLMENT_MODEL_PATH)
semester_1_model = joblib.load(config.SEMESTER_1_MODEL_PATH)


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


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict/enrollment")
def predict_enrollment(enrollment_data: EnrollmentFeatures):
    X = pd.DataFrame([enrollment_data.model_dump(by_alias=True)])
    predict_probabilities = enrollment_model.predict_proba(X)[0]

    return {
        "dropout": bool(predict_probabilities[1] > 0.5),
        "dropout_risk_score": round(float(predict_probabilities[1], 3)),
    }
