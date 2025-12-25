from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import json
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
import os
import logging
from .schemas import MushroomInput, PredictionOutput

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Mushroom AI API")

# Mount static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir, html=True), name="static")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and metadata
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(BASE_DIR, "models")
    
    pipeline = joblib.load(os.path.join(models_dir, "XGBoost.pkl"))
    with open(os.path.join(models_dir, "columns_used.json"), "r") as f:
        columns_used = json.load(f)
    logger.info("Model and metadata loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model or metadata: {e}")
    raise RuntimeError("Model loading failed")

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: MushroomInput):
    try:
        # Convert input to DataFrame using alias (to match model feature names with hyphens)
        data_dict = input_data.model_dump(by_alias=True)
        df = pd.DataFrame([data_dict])

        # Validate columns
        missing_cols = [col for col in columns_used if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing columns: {missing_cols}")
            raise HTTPException(status_code=400, detail=f"Missing required columns: {missing_cols}")
        
        # Ensure correct column order
        df = df[columns_used]
        
        prediction = pipeline.predict(df)
        result = int(prediction[0])
        logger.info(f"Prediction made: {result}")
        return {"prediction": result}
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def serve_frontend():
    return FileResponse(os.path.join("static", "index.html"))

@app.get("/{full_path:path}", response_class=HTMLResponse)
async def frontend_catch_all(full_path: str):
    # Serve index.html for any other path to let frontend router handle it (if there was one)
    # For now it just redirects navigation to the single page app
    path = os.path.join("static", "index.html")
    if os.path.exists(path):
         return FileResponse(path)
    return HTMLResponse(status_code=404, content="<h1>404 - Not Found</h1>")