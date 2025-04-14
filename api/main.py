from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import logging
import os
import numpy as np
from data import preprocess_input  # Import the updated preprocessing function

# Set up basic logging configuration
logging.basicConfig(level=logging.INFO)

# Load model and encoder from pickle files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'housing_data', 'housing_demand_model.pkl')  # Path to model
encoder_path = os.path.join(BASE_DIR, 'housing_data', 'label_encoder.pkl')  # Path to encoder

# Load the trained model and label encoder
model = joblib.load(model_path)
label_encoder = joblib.load(encoder_path)

app = FastAPI()

# Allow CORS for all origins (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define input model using Pydantic (5 parameters only)
class PredictionRequest(BaseModel):
    bhk: int
    location: str
    rera: bool
    gym: str
    pool: str

# POST route for predictions
@app.post("/predict")
def predict(req: PredictionRequest):
    try:
        # Preprocess input using the function from data.py
        processed_input = preprocess_input(req.location, req.bhk, req.rera, req.gym, req.pool)
        if processed_input[0][0] == -1:
            return {"error": "Unknown location. Please choose from valid options."}

        # Make prediction using the model
        prediction = model.predict(processed_input)
        logging.info(f"Prediction successful for location: {req.location}, Score: {float(prediction[0])}")

        return {"prediction": float(prediction[0])}

    except Exception as e:
        return {"error": str(e)}
