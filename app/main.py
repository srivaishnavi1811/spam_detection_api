import os
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import joblib

# --------------------------
# Initialize FastAPI app
# --------------------------
app = FastAPI()

# --------------------------
# Serve static files (CSS, images)
# --------------------------
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "../static")), name="static")

# --------------------------
# Setup templates path
# --------------------------
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "../templates"))

# --------------------------
# Load the trained spam model
# --------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../model/spam_model.pkl")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"{MODEL_PATH} not found. Make sure your model is here.")
model = joblib.load(MODEL_PATH)

# --------------------------
# GET route: show web page
# --------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html", {"request": request, "prediction": "", "message": ""}
    )

# --------------------------
# POST route: predict spam
# --------------------------
@app.post("/", response_class=HTMLResponse)
async def predict_web(request: Request, message: str = Form(...)):
    # Predict using the model
    raw_pred = model.predict([message])[0]
    prediction = "Spam" if raw_pred == 1 else "Ham"

    # Optional: confidence if model supports predict_proba
    try:
        confidence = max(model.predict_proba([message])[0])
        confidence = round(confidence * 100, 2)
    except:
        confidence = None

    # Render template with prediction
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "prediction": prediction,
            "message": message,
            "confidence": confidence
        }
    )