from fastapi import FastAPI
import joblib

from app.database import SessionLocal, engine
from app.models import Base, Message

# Create table
Base.metadata.create_all(bind=engine)

app = FastAPI()

# Load trained model
model = joblib.load("model/spam_model.pkl")


@app.post("/predict")
def predict(message: str):

    # Make prediction
    result = model.predict([message])[0]

    prediction = "spam" if result == 1 else "ham"

    # Open database session
    db = SessionLocal()

    # Save message and prediction
    new_message = Message(
        text=message,
        prediction=prediction
    )

    db.add(new_message)
    db.commit()
    db.close()

    return {
        "message": message,
        "prediction": prediction
    }