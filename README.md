
# 🌾 Crop and Yield Prediction Using Machine Learning

This project is a Flask-based machine learning web application that predicts:

1. **The most suitable crop** based on soil and weather parameters  
2. **Expected crop yield** using environmental and nutrient data  

It uses deep learning (LSTM), classical ML preprocessing, and a clean web UI to
help farmers, agronomists and researchers make data-driven agricultural
decisions.

---

## 🚀 Features

### 🌱 **Crop Recommendation**
Predicts the best crop using:
- Nitrogen (N)
- Phosphorus (P)
- Potassium (K)
- pH level
- Rainfall
- Humidity
- Temperature
- Location (optional)

### 📈 **Crop Yield Prediction**
Predicts expected yield (e.g., quintals/acre) using LSTM based on:
- Soil nutrients  
- Fertilizer use  
- Temperature  
- Other environmental factors  

### 🖥️ **Flask Web Interface**
- Clean input forms  
- Easy-to-read prediction outputs  
- Optional live location extraction (via `live_location.html`)  

### 📦 **Models Included**
- `models/lstm_crop_modelfinal.h5`  
- `models/lstm_crop_yieldfinal.h5`  
- Preprocessing files:  
  - `scaler_lstmfinal.pkl`  
  - `feature_names_lstmfinal.pkl`  
  - `label_encoderfinal.pkl`

### 🔧 **Backend Utilities**
- `clean_ds.py` → dataset preprocessing  
- `.ipynb` notebooks → model training, evaluation, saving  
- `config.py` → Flask config  
- `adding.py`, `live.py`, `app.py` → Flask routes/controllers  

---

## 📂 Project Structure

