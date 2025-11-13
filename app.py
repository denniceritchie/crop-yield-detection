
# from flask import Flask, render_template, request, jsonify
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from tensorflow.keras.losses import MeanSquaredError
# import joblib
# from sklearn.preprocessing import StandardScaler

# app = Flask(__name__)

# # Custom objects for model loading
# custom_objects = {
#     'MeanSquaredError': MeanSquaredError,
#     'mse': MeanSquaredError()
# }

# # Load models and scalers
# try:
#     model = tf.keras.models.load_model(
#         'lstm_crop_yieldfinal.h5',
#         custom_objects=custom_objects
#     )
#     scaler = joblib.load('scaler_lstmfinal.pkl')
#     feature_names = joblib.load('feature_names_lstmfinal.pkl')
#     print("All models loaded successfully!")
# except Exception as e:
#     print(f"Error loading models: {e}")
#     model = None
#     scaler = None
#     feature_names = None

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if model is None or scaler is None:
#         return render_template('error.html', error="Model not loaded properly. Please check server logs.")
    
#     if request.method == 'POST':
#         try:
#             # Get form data
#             data = request.form.to_dict()
            
#             # Convert to float
#             input_data = [
#                 float(data['fertilizer']),
#                 float(data['temperature']),
#                 float(data['nitrogen']),
#                 float(data['phosphorus']),
#                 float(data['potassium'])
#             ]
            
#             # Create DataFrame
#             new_data = pd.DataFrame([input_data], columns=feature_names)
            
#             # Scale data
#             new_data_scaled = scaler.transform(new_data)
#             new_data_scaled = new_data_scaled.reshape((new_data_scaled.shape[0], 1, new_data_scaled.shape[1]))
            
#             # Predict
#             predicted_yield = model.predict(new_data_scaled)
#             prediction = predicted_yield[0][0]
            
#             return render_template('result.html', 
#                                  prediction=f"{prediction:.2f}",
#                                  input_data=data)
            
#         except ValueError as ve:
#             return render_template('error.html', error=f"Invalid input: {str(ve)}")
#         except Exception as e:
#             return render_template('error.html', error=f"Prediction error: {str(e)}")

# @app.route('/api/predict', methods=['POST'])
# def api_predict():
#     """API endpoint for JSON requests"""
#     if model is None or scaler is None:
#         return jsonify({'error': 'Model not loaded'}), 500
    
#     try:
#         data = request.get_json()
#         input_data = [
#             float(data['fertilizer']),
#             float(data['temperature']),
#             float(data['nitrogen']),
#             float(data['phosphorus']),
#             float(data['potassium'])
#         ]
        
#         new_data = pd.DataFrame([input_data], columns=feature_names)
#         new_data_scaled = scaler.transform(new_data)
#         new_data_scaled = new_data_scaled.reshape((new_data_scaled.shape[0], 1, new_data_scaled.shape[1]))
        
#         predicted_yield = model.predict(new_data_scaled)
        
#         return jsonify({
#             'prediction': float(predicted_yield[0][0]),
#             'unit': 'Q/acre'
#         })
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# if __name__ == '__main__':
#     # Disable AVX/FMA warnings
#     import os
#     os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
#     app.run(debug=True, host='0.0.0.0')









from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError
import numpy as np
import pandas as pd
import joblib
import pickle
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os   

app = Flask(__name__)
app.config.from_pyfile('config.py')

# Database setup
db = SQLAlchemy(app)

# User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    predictions = db.relationship('Prediction', backref='user', lazy=True)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    prediction_type = db.Column(db.String(50), nullable=False)
    input_data = db.Column(db.JSON, nullable=False)
    result = db.Column(db.String(200), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

# Create database tables
with app.app_context():
    db.create_all()

# Load models and scalers
def load_models():
    try:
        # Crop type prediction model
        crop_model = tf.keras.models.load_model(
            'models/lstm_crop_modelfinal.h5',
            custom_objects={'sparse_categorical_crossentropy': tf.keras.losses.SparseCategoricalCrossentropy()}
        )
        
        # Crop yield prediction model
        yield_model = tf.keras.models.load_model(
            'models/lstm_crop_yieldfinal.h5',
            custom_objects={'MeanSquaredError': MeanSquaredError, 'mse': MeanSquaredError()}
        )
        
        # Load encoders and scalers
        with open('models/label_encoderfinal.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        
        with open('models/scalerfinal.pkl', 'rb') as f:
            crop_scaler = pickle.load(f)
        
        yield_scaler = joblib.load('models/scaler_lstmfinal.pkl')
        feature_names = joblib.load('models/feature_names_lstmfinal.pkl')
        
        return {
            'crop_model': crop_model,
            'yield_model': yield_model,
            'label_encoder': label_encoder,
            'crop_scaler': crop_scaler,
            'yield_scaler': yield_scaler,
            'yield_features': feature_names
        }
    except Exception as e:
        print(f"Error loading models: {e}")
        return None

models = load_models()

# Routes
@app.route('/')
def home():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        # Check if user exists
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'error')
            return redirect(url_for('register'))
        
        if User.query.filter_by(email=email).first():
            flash('Email already exists', 'error')
            return redirect(url_for('register'))
        
        # Create new user
        new_user = User(
            username=username,
            email=email,
            password=generate_password_hash(password)
        )
        
        db.session.add(new_user)
        db.session.commit()
        
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['username'] = user.username
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'error')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out', 'info')
    return redirect(url_for('home'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_predictions = Prediction.query.filter_by(user_id=session['user_id']).order_by(Prediction.timestamp.desc()).limit(5).all()
    return render_template('dashboard.html', predictions=user_predictions)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    if models is None:
        flash('Models failed to load. Please contact administrator.', 'error')
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        prediction_type = request.form['prediction_type']
        
        try:
            if prediction_type == 'crop_type':
                input_data = {
                    "N": float(request.form['nitrogen']),
                    "P": float(request.form['phosphorus']),
                    "K": float(request.form['potassium']),
                    "temperature": float(request.form['temperature']),
                    "humidity": float(request.form['humidity']),
                    "ph": float(request.form['ph']),
                    "rainfall": float(request.form['rainfall']),
                    "Latitude": float(request.form['latitude']),
                    "Longitude": float(request.form['longitude']),
                    "Location": request.form['location']
                }
                print(input_data,"input data")
                # Preprocess input
                new_data = pd.DataFrame([input_data])
                new_data = pd.get_dummies(new_data, columns=["Location"])
                
                # Add missing columns
                for col in models['crop_scaler'].feature_names_in_:
                    if col not in new_data.columns:
                        new_data[col] = 0
                
                new_data = new_data[models['crop_scaler'].feature_names_in_]
                new_data_scaled = models['crop_scaler'].transform(new_data)
                new_data_scaled = new_data_scaled.reshape((1, 1, new_data_scaled.shape[1]))
                
                # Predict
                pred = models['crop_model'].predict(new_data_scaled)
                predicted_crop = models['label_encoder'].inverse_transform([np.argmax(pred)])[0]
                print(pred,'pred')
                print('-------------------')
                print(predicted_crop,'-------predicted crop-----')
                # Save prediction
                new_pred = Prediction(
                    user_id=session['user_id'],
                    prediction_type='crop_type',
                    input_data=input_data,
                    result=predicted_crop
                )
                db.session.add(new_pred)
                db.session.commit()
                
                return render_template('result.html', 
                                    prediction_type='Crop Type',
                                    result=predicted_crop,
                                    input_data=input_data)
            
            elif prediction_type == 'yield':
                input_data = {
                    "Fertilizer": float(request.form['fertilizer']),
                    "Temperatue": float(request.form['temperature']),
                    "Nitrogen (N)": float(request.form['nitrogen']),
                    "Phosphorus (P)": float(request.form['phosphorus']),
                    "Potassium (K)": float(request.form['potassium'])
                }
                
                # Create DataFrame
                new_data = pd.DataFrame([input_data], columns=models['yield_features'])
                
                # Scale data
                new_data_scaled = models['yield_scaler'].transform(new_data)
                new_data_scaled = new_data_scaled.reshape((1, 1, new_data_scaled.shape[1]))
                
                # Predict
                predicted_yield = models['yield_model'].predict(new_data_scaled)
                result = f"{predicted_yield[0][0]:.2f} Q/acre"
                
                # Save prediction
                new_pred = Prediction(
                    user_id=session['user_id'],
                    prediction_type='crop_yield',
                    input_data=input_data,
                    result=result
                )
                db.session.add(new_pred)
                db.session.commit()
                
                return render_template('result.html', 
                                    prediction_type='Crop Yield',
                                    result=result,
                                    input_data=input_data)
        
        except Exception as e:
            flash(f'Prediction error: {str(e)}', 'error')
            return redirect(url_for('predict'))
    
    return render_template('predict.html')

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    app.run(debug=True)