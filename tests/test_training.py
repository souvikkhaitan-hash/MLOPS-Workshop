
import os
import joblib

def test_model_exists():
    assert os.path.exists("models/student_model.pkl")

def test_model_can_predict():
    model = joblib.load("models/student_model.pkl")

    prediction = model.predict([[
        6, 85, 75, 80
    ]])

    assert len(prediction) == 1
    assert 0 <= prediction[0] <= 100
