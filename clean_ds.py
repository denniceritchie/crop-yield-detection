import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import joblib
from sklearn.preprocessing import LabelEncoder

def load_models():
    try:
        # Crop type prediction model
        crop_model = tf.keras.models.load_model(
            'models/lstm_crop_modelfinal.h5',
            custom_objects={'sparse_categorical_crossentropy': tf.keras.losses.SparseCategoricalCrossentropy()}
        )
        
        # Load encoders and scalers
        with open('models/label_encoderfinal.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        
        with open('models/scalerfinal.pkl', 'rb') as f:
            crop_scaler = pickle.load(f)
        
        return {
            'crop_model': crop_model,
            'label_encoder': label_encoder,
            'crop_scaler': crop_scaler
        }
    except Exception as e:
        print(f"Error loading models: {e}")
        return None

def clean_dataset(input_csv_path, output_csv_path):
    # Load models
    models = load_models()
    if models is None:
        print("Failed to load models. Exiting.")
        return
    
    # Load dataset
    df = pd.read_csv(input_csv_path)
    
    # Make sure the dataset has the expected columns
    required_columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 
                       'Latitude', 'Longitude', 'Location', 'label']
    if not all(col in df.columns for col in required_columns):
        print("Dataset doesn't have all required columns")
        return
    
    # Preprocess the data similar to your prediction code
    X = df.drop(columns=['label'])
    y_true = df['label']
    
    # One-hot encode Location
    X_processed = pd.get_dummies(X, columns=["Location"])
    
    # Add missing columns
    for col in models['crop_scaler'].feature_names_in_:
        if col not in X_processed.columns:
            X_processed[col] = 0
    
    # Keep only the columns that the scaler expects
    X_processed = X_processed[models['crop_scaler'].feature_names_in_]
    
    # Scale the features
    X_scaled = models['crop_scaler'].transform(X_processed)
    
    # Reshape for LSTM input (assuming your model expects 3D input)
    X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
    
    # Make predictions
    preds = models['crop_model'].predict(X_scaled)
    predicted_labels = models['label_encoder'].inverse_transform(np.argmax(preds, axis=1))
    
    # Compare predictions with true labels
    correct_predictions = (predicted_labels == y_true)
    
    # Filter the original dataset to keep only correctly predicted rows
    cleaned_df = df[correct_predictions]
    
    # Save the cleaned dataset
    cleaned_df.to_csv(output_csv_path, index=False)
    print(f"Saved cleaned dataset to {output_csv_path}")
    print(f"Original size: {len(df)}, Cleaned size: {len(cleaned_df)}")
    print(f"Accuracy on original dataset: {correct_predictions.mean():.2%}")

# Example usage
if __name__ == "__main__":
    input_csv_path = "updated_dataset.csv"  # Replace with your input CSV path
    output_csv_path = "cleaned_dataset.csv"  # Replace with desired output path
    clean_dataset(input_csv_path, output_csv_path)