from fastapi import FastAPI
from pydantic import create_model
import joblib
import pandas as pd
import json
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
pipeline = joblib.load("models/XGBoost.pkl")

# Load columns used in the model
with open("models/columns_used.json", "r") as f:
    columns_used = json.load(f)

with open("models/columns_types.json", "r") as f:
    columns_types = json.load(f)

field_definitions = {
    col: (float if t == "float" else str, ...)
    for col, t in columns_types.items()
}

MushroomInput = create_model("MushroomInput", **field_definitions)

@app.post("/predict")
def predict(input_data: MushroomInput):  # type: ignore
    df = pd.DataFrame([input_data.model_dump()])

    missing_cols = [col for col in columns_used if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    df = df[columns_used]
    
    prediction = pipeline.predict(df)
    return {"prediction": int(prediction[0])}

@app.get("/")
def read_root():
    return {"status": "ok"}