import pandas as pd
import numpy as np
import joblib
import os

# Load the label encoder (since it's saved as a pickle file)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
encoder_path = os.path.join(BASE_DIR, 'housing_data', 'label_encoder.pkl')
label_encoder = joblib.load(encoder_path)

# Correct path to the CSV file (based on your system)
properties_path = os.path.join(BASE_DIR, 'housing_data', 'Final_Demand_Prediction_With_Amenities.csv')
df_properties = pd.read_csv(properties_path)

def preprocess_input(location: str, bhk: int, rera: bool, gym: str, pool: str):
    # Encode location using the label encoder
    loc_encoded = label_encoder.transform([location])[0] if location in label_encoder.classes_ else -1
    
    # Convert RERA flag to binary (1/0)
    rera_val = 1 if rera else 0
    
    # Convert Gym and Pool to binary (1/0)
    gym_val = 1 if gym.lower() == "yes" else 0
    pool_val = 1 if pool.lower() == "yes" else 0
    
    # Start with the full dataset and apply filters progressively
    filtered_properties = df_properties.copy()

    # First, filter based on BHK, Gym, and Pool preference
    filtered_properties = filtered_properties[
        (filtered_properties['BHK'] == bhk) &
        (filtered_properties['Gym Available'] == gym_val) &
        (filtered_properties['Swimming Pool Available'] == pool_val)
    ]
    
    # If the location is valid, apply the location filter
    if loc_encoded != -1:
        filtered_properties = filtered_properties[filtered_properties['Location'] == loc_encoded]

    # Relax filters progressively if no matching results
    if filtered_properties.empty:
        # Relax Gym and Pool preference first
        filtered_properties = df_properties[
            (df_properties['BHK'] == bhk)
        ]
        
        if filtered_properties.empty:
            # Relax BHK preference
            filtered_properties = df_properties[
                (df_properties['Location'] == loc_encoded)
            ]
            
            if filtered_properties.empty:
                # Show top options across all locations if still no match
                filtered_properties = df_properties.copy()

    # Return filtered properties with relevant details (like name, location, price, etc.)
    return filtered_properties[['Society Name', 'Location', 'Price', 'Gym Available', 'Swimming Pool Available', 'Star Rating', 'Estimated Rent']]
