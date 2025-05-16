from flask import Flask, request, jsonify
from google.cloud import storage
import pandas as pd
import os
import io

GCS_BUCKET = "ada2024-450119"
GCS_PATIENTS_PATH = "training_sets/Full_Image_data.pkl"

app = Flask(__name__)

# In-memory patient store: {Subject ID: patient_dict}
PATIENTS = {}

def load_patients_from_gcs():
    global PATIENTS
    storage_client = storage.Client()
    bucket = storage_client.bucket(GCS_BUCKET)
    blob = bucket.blob(GCS_PATIENTS_PATH)
    data_bytes = blob.download_as_bytes()
    df = pd.read_pickle(io.BytesIO(data_bytes))
    # Fill NaNs with None for JSON compatibility
    df = df.where(pd.notnull(df), None)
    for _, row in df.iterrows():
        patient = row.to_dict()
        subject_id = patient["Subject ID"]
        PATIENTS[subject_id] = patient

@app.before_first_request
def startup():
    load_patients_from_gcs()

@app.route('/patients', methods=['GET'])
def list_patients():
    return jsonify(list(PATIENTS.values())), 200

@app.route('/patients/<subject_id>', methods=['GET'])
def get_patient(subject_id):
    patient = PATIENTS.get(subject_id)
    if not patient:
        return jsonify({"error": "Not found"}), 404
    return jsonify(patient), 200

@app.route('/patients/<subject_id>', methods=['PUT'])
def update_patient(subject_id):
    data = request.get_json()
    if subject_id not in PATIENTS:
        return jsonify({"error": "Not found"}), 404
    PATIENTS[subject_id].update(data)
    # TODO: Optionally, write back to GCS here
    return jsonify({"status": "updated", "patient": PATIENTS[subject_id]}), 200

@app.route('/patients/<subject_id>', methods=['DELETE'])
def delete_patient(subject_id):
    if subject_id in PATIENTS:
        del PATIENTS[subject_id]
        # TODO: Optionally, write back to GCS here
        return jsonify({"status": "deleted"}), 200
    return jsonify({"error": "Not found"}), 404

@app.route('/patients', methods=['POST'])
def add_patient():
    data = request.get_json()
    subject_id = data.get("Subject ID")
    if not subject_id:
        return jsonify({"error": "Subject ID required"}), 400
    if subject_id in PATIENTS:
        return jsonify({"error": "Patient already exists"}), 409
    PATIENTS[subject_id] = data
    # TODO: Optionally, write back to GCS here
    return jsonify({"status": "created", "patient": data}), 201

@app.route('/patients/<subject_id>/history', methods=['GET'])
def get_history(subject_id):
    patient = PATIENTS.get(subject_id)
    if not patient:
        return jsonify({"error": "Not found"}), 404
    # Example: return all visits for this Subject ID (if multiple in data)
    # For now, just return the patient record
    return jsonify({"history": [patient]}), 200

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200

if __name__ == '__main__':
    # For local testing only; on Cloud Run, use gunicorn
    load_patients_from_gcs()
    app.run(host='0.0.0.0', port=8081)
