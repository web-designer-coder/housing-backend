
# 🏡 Property Prices Prediction

This project is a full-stack application with a web-based frontend and a Python backend API that predicts property prices or demand based on user input and housing data.

---

## 🚀 Features

- 🎯 Predict property demand or price using machine learning
- 🖥️ Clean frontend using **HTML**, **CSS**, and **JavaScript**
- 🔗 API integration using **Python Fastapi**
- 📊 Uses a trained model (`.pkl`) for prediction
- 🧠 Encodes input using label encoders for categorical data

---

## 📂 Project Structure

```
housing-backend-main/
│
├── index.html               # Main web interface
├── style.css                # Styling for the frontend
├── main.js                  # Handles form submission & API requests
│
├── api/                     # Backend API folder
│   ├── main.py              # Flask app that serves predictions
│   ├── data.py              # Helper functions to process input
│   ├── model_loader.py      # Loads model and encoders
│   ├── requirements.txt     # Backend dependencies
│   └── housing_data/
│       ├── housing_demand_model.pkl     # Trained ML model
│       ├── label_encoder.pkl            # Label encoder for inputs
│       └── Final_Demand_Prediction_With_Amenities.csv
│
├── README.md                # Project documentation
└── .gitignore               # Files to ignore in version control
```

---

## 📊 Dataset Used

- **File Name**: `Final_Demand_Prediction_With_Amenities.csv`  
- **Path**: `api/housing_data/`  

### 📄 Description:

This dataset includes real estate features such as:
- Location
- Number of Bedrooms
- Area (sq ft)
- Furnishing Status
- Amenities
- Target variable: Price or demand


---

## 📡 API Endpoint

- **URL**: `http://localhost:5000/predict`
- **Method**: `POST`
- **Content-Type**: `application/json`
- **Sample Payload**:
```json
{
  "location": "Mumbai",
  "bedrooms": 2,
  "area": 1200,
  "furnishing": "Furnished"
}
```
