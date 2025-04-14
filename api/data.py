import numpy as np
import joblib
import os

# Load the label encoder (since it's saved as a pickle file)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
encoder_path = os.path.join(BASE_DIR, 'housing_data', 'label_encoder.pkl')
label_encoder = joblib.load(encoder_path)

def preprocess_input(location: str, bhk: int, rera: bool, gym: str, pool: str):
    # Encode location using the label encoder
    loc_encoded = label_encoder.transform([location])[0] if location in label_encoder.classes_ else -1
    
    # Convert RERA flag to binary (1/0)
    rera_val = 1 if rera else 0
    
    # Convert Gym and Pool to binary (1/0)
    gym_val = 1 if gym.lower() == "yes" else 0
    pool_val = 1 if pool.lower() == "yes" else 0
    
    # Return the processed input as a numpy array
    return np.array([[loc_encoded, bhk, rera_val, gym_val, pool_val]])
