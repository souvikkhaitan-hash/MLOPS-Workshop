
from pydantic import BaseModel, Field

class StudentInput(BaseModel):
    study_hours: float = Field(ge=0, le=24)
    attendance: float = Field(ge=0, le=100)
    previous_marks: float = Field(ge=0, le=100)
    assignment_score: float = Field(ge=0, le=100)

class PredictionResponse(BaseModel):
    predicted_final_score: float
