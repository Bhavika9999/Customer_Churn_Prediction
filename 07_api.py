
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# Create FastAPI application
app = FastAPI(title="Customer Churn Prediction API")

# Load trained model
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
model = joblib.load(BASE_DIR / "best_churn_model.pkl")


# Input data format
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float
    TenureGroup: str
    HasFamily: int
    NumberOfServices: int


@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API is running!"}


@app.post("/predict")
def predict_churn(customer: CustomerData):

    data = pd.DataFrame([customer.model_dump()])

    probability = float(model.predict_proba(data)[0][1])
    prediction = int(probability >= 0.6)

    return {
        "churn_probability": round(probability, 4),
        "churn_prediction": prediction,
        "result": "Yes" if prediction == 1 else "No"
    }
