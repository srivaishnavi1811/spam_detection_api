from app.database import engine
from app.models import Base

Base.metadata.create_all(bind=engine)
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load the trained model
model = joblib.load("model/spam_model.pkl")

# Create FastAPI app
app = FastAPI()

# Define request structure
class Message(BaseModel):
    message: str

# Home route
@app.get("/")
def home():
    return {"message": "Spam Detection API is running"}

# Prediction route
@app.post("/predict")
def predict(data: Message):
    prediction = model.predict([data.message])[0]

    if prediction == 1:
        result = "spam"
    else:
        result = "ham"

    return {"prediction": result}