
import joblib
import pandas as pd

from fastapi import FastAPI
from app.schemas import StudentInput, PredictionResponse

app = FastAPI(
    title="Student Performance Prediction API",
    version="1.0.0"
)

model = joblib.load("models/student_model.pkl")

FEATURES = [
    "study_hours",
    "attendance",
    "previous_marks",
    "assignment_score"
]


@app.get("/")
def home():
    return {"message": "Student Performance Prediction API is running"}


@app.post("/predict", response_model=PredictionResponse)
def predict(student: StudentInput):
    student_data = pd.DataFrame(
        [student.model_dump()],
        columns=FEATURES
    )

    prediction = model.predict(student_data)[0]

    return {
        "predicted_final_score": round(
            max(0, min(100, float(prediction))), 2
        )
    }
