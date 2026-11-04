
import joblib
import pandas as pd

FEATURES = [
    "study_hours",
    "attendance",
    "previous_marks",
    "assignment_score"
]

model = joblib.load("models/student_model.pkl")

student = pd.DataFrame([{
    "study_hours": 6,
    "attendance": 85,
    "previous_marks": 75,
    "assignment_score": 80
}], columns=FEATURES)

prediction = model.predict(student)

print("Predicted final score:", round(float(prediction[0]), 2))
