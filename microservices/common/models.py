from dataclasses import dataclass

@dataclass
class Patient:
    id: str
    name: str
    age: int
    gender: str

@dataclass
class Prediction:
    id: str
    patient_id: str
    diagnosis: str
    confidence: float

@dataclass
class Feedback:
    prediction_id: str
    doctor_id: str
    feedback: str
    notes: str
