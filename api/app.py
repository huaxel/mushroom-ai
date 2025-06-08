from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from fastapi.staticfiles import StaticFiles

app = FastAPI()
app.mount("/", StaticFiles(directory="static", html=True), name="static")

# Load model
model = joblib.load("../models/Extra_Trees_v1.pkl")

## Input schema
class MushroomInput(BaseModel):
    stem_length: float
    stem_width: float
    cap_surface: str
    cap_color: str
    # and other features

@app.post("/predict")
def predict(input_data: MushroomInput):
    df = pd.DataFrame([input_data.dict()])
    prediction = model.predict(df)
    return {"prediction": int(prediction[0])}