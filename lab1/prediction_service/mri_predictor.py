# MRI Classifier logic will go here 

import joblib
import numpy as np
import pandas as pd
import nibabel as nib
import os
import sys # Added import
from typing import Union, Dict # Added for older Python compatibility
from google.cloud import storage # Added for GCS
import io # Added for GCS

# --- GCS Configuration (must match where training_service saves the model) ---
GCS_BUCKET_NAME = "ada2-training"  # As updated by user in classification.py
MODEL_GCS_PATH = "models/knn_model.pkl" # As updated by user in classification.py
# --- End GCS Configuration ---

MODEL = None # Global variable to hold the loaded model

def download_blob_to_memory(bucket_name, source_blob_name):
    """Downloads a blob from GCS into memory."""
    storage_client = storage.Client() # Assumes ADC or service account for authentication
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    
    try:
        print(f"Attempting to download model gs://{bucket_name}/{source_blob_name}...")
        file_as_bytes = blob.download_as_bytes()
        print(f"Successfully downloaded model gs://{bucket_name}/{source_blob_name} to memory.")
        return file_as_bytes
    except Exception as e:
        print(f"Error downloading model gs://{bucket_name}/{source_blob_name}: {e}")
        raise

def load_model_from_gcs():
    """Loads the KNN model from GCS into the global MODEL variable."""
    global MODEL
    if MODEL is None:
        print(f"Loading model from GCS path: gs://{GCS_BUCKET_NAME}/{MODEL_GCS_PATH}")
        try:
            model_bytes = download_blob_to_memory(GCS_BUCKET_NAME, MODEL_GCS_PATH)
            model_file_like_object = io.BytesIO(model_bytes)
            MODEL = joblib.load(model_file_like_object)
            print("Model loaded successfully from GCS.")
        except Exception as e:
            MODEL = None # Ensure model is None if loading failed
            print(f"Failed to load model from GCS: {e}")
            # Depending on requirements, you might want to raise an error here
            # or handle it in the calling code (e.g., app.py might refuse to start)
            raise RuntimeError(f"Failed to load model from GCS: {e}") # Make it critical
    return MODEL

# Define the slicing for feature reduction, consistent with training
FOURTH_SLICE = np.s_[::4, ::4, ::4, 0] # Using np.s_ for slice object

def preprocess_single_nifti_for_prediction(image_path: str) -> np.ndarray:
    """
    Loads a single NIfTI image, preprocesses it for prediction similar to training data.
    This version is simplified for prediction where we expect a .nii.gz file.
    """
    try:
        img_object = nib.load(image_path)
        img_data = img_object.get_fdata()

        if img_data.ndim == 3:
            img_data = img_data[..., np.newaxis] # Add channel dim if 3D
        
        if img_data.ndim != 4:
            raise ValueError(f"Image at {image_path} has unexpected dimensions: {img_data.ndim} after attempting to make it 4D. Expected 3D or 4D raw.")

        # Apply the predefined slice for feature reduction.
        processed_image_data = img_data[FOURTH_SLICE]
        
        # Flatten the image data to a 1D array (features)
        flattened_image = processed_image_data.reshape(-1)
        return flattened_image
    except FileNotFoundError:
        print(f"Error: The file was not found at {image_path}")
        raise
    except Exception as e:
        print(f"An error occurred while processing {image_path} for prediction: {e}")
        raise

def predict_mri_path(image_path: str):
    """
    Predicts on a single NIfTI image file path after preprocessing.
    Assumes the model is loaded into the global MODEL variable.
    """
    global MODEL
    if MODEL is None:
        print("Model not loaded. Attempting to load now.")
        load_model_from_gcs() # Attempt to load if not already loaded
        if MODEL is None: # Check again if loading failed
             return {"error": "Model is not available, cannot predict."}

    try:
        # Preprocess the image
        processed_image_features = preprocess_single_nifti_for_prediction(image_path)
        
        # Model expects a 2D array (batch_size, n_features)
        # For a single prediction, reshape to (1, n_features)
        features_for_prediction = processed_image_features.reshape(1, -1)
        
        # Make prediction
        prediction = MODEL.predict(features_for_prediction)
        # prediction_proba = MODEL.predict_proba(features_for_prediction) # If you need probabilities
        
        # Assuming prediction is an array like [0] or [1]
        return {"prediction": int(prediction[0])}
    except Exception as e:
        print(f"Error during prediction for {image_path}: {e}")
        # It's good practice to return a specific error structure or raise an exception
        # that the API layer (app.py) can catch and translate into an HTTP error.
        return {"error": f"Prediction failed: {str(e)}"} 

# Example of how to load the model when the module is imported (optional, app.py can also trigger this)
# load_model_from_gcs() # You could uncomment this if you want the model to load upon first import.
# However, it's generally better to control initialization from the main app script (app.py)
# to handle potential errors during startup gracefully.