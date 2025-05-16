import numpy as np
import nibabel as nib
import joblib
from google.cloud import storage

GCS_BUCKET_NAME = "ada2024-450119"
MODEL_GCS_PATH = "models/knn_model.pkl"
MODEL = None

def download_blob_to_memory(bucket_name, source_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    return blob.download_as_bytes()

def load_model_from_gcs():
    global MODEL
    if MODEL is None:
        model_bytes = download_blob_to_memory(GCS_BUCKET_NAME, MODEL_GCS_PATH)
        MODEL = joblib.load(joblib.io.BytesIO(model_bytes))
    return MODEL

FOURTH_SLICE = np.s_[::4, ::4, ::4, 0]

def preprocess_single_nifti_for_prediction(image_path: str) -> np.ndarray:
    img_object = nib.load(image_path)
    img_data = img_object.get_fdata()
    if img_data.ndim == 3:
        img_data = img_data[..., np.newaxis]
    processed_image_data = img_data[FOURTH_SLICE]
    flattened_image = processed_image_data.reshape(-1)
    return flattened_image

def predict_mri_path(image_path: str):
    global MODEL
    if MODEL is None:
        load_model_from_gcs()
    features = preprocess_single_nifti_for_prediction(image_path)
    pred = MODEL.predict([features])[0]
    conf = max(MODEL.predict_proba([features])[0])
    return pred, conf
