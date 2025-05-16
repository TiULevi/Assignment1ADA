import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from mri_predictor import load_model_from_gcs, predict_mri_path, GCS_BUCKET_NAME, MODEL_GCS_PATH

app = Flask(__name__)

UPLOAD_FOLDER = '/tmp/mri_uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
ALLOWED_EXTENSIONS = {'nii', 'img'}

if not os.path.exists(UPLOAD_FOLDER):
    try:
        os.makedirs(UPLOAD_FOLDER)
        print(f"Created upload folder: {UPLOAD_FOLDER}")
    except OSError as e:
        print(f"Error creating upload folder {UPLOAD_FOLDER}: {e}")

# Load the model when the application starts.
# If the model fails to load, the application will fail to start (due to RuntimeError in load_model_from_gcs),
# which is often the desired behavior for a critical component like the model.
print("Attempting to load MRI model from GCS at application startup...")
try:
    load_model_from_gcs() # This populates the MODEL variable in mri_predictor module
    print(f"Successfully initiated model loading from gs://{GCS_BUCKET_NAME}/{MODEL_GCS_PATH}")
except RuntimeError as e:
    print(f"CRITICAL: Failed to load model on startup: {e}")
    # In a production K8s environment, the pod would likely crash and restart, which is okay.
    # For local dev, this print statement will be visible.
    # You might exit or prevent app from running further if model loading is absolutely critical for startup.
    # For now, if load_model_from_gcs raises RuntimeError, Flask won't start serving requests if it happens here.
    raise # Re-raise the exception to stop the app if model loading fails

def allowed_file(filename):
    if '.' not in filename:
        return False
    if filename.lower().endswith('.nii.gz'):
        return True
    ext = filename.rsplit('.', 1)[1].lower()
    return ext in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def health_check():
    # Simple health check endpoint
    # You could add a check here to see if mri_predictor.MODEL is not None for a more specific health check
    return jsonify({"status": "healthy", "message": "Prediction service is running."}), 200

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'image_name' not in data:
        return jsonify({"error": "Missing image_name in JSON payload"}), 400

    image_name = data['image_name']

    # IMPORTANT: For a full GCS workflow, 'image_name' here should ideally be a GCS URI
    # or the endpoint should handle file uploads. 
    # For now, it's assumed to be a file path accessible by the service.
    # If running in Docker, this path needs to be valid *within the container*,
    # which might involve volume mounting for local files or downloading from GCS if it's a GCS URI.
    
    # Example: If image_name is a GCS URI like "gs://your-bucket/your-image.nii.gz"
    # you would first need to download it to a temporary local path before passing to predict_mri_path.
    # Or, modify predict_mri_path to accept a GCS URI and handle the download itself.
    # For this iteration, we assume image_name is a local path that predict_mri_path can access.

    if not os.path.exists(image_name):
         # This check is for a local file path. If image_name were a GCS URI, this check would be different.
         print(f"Warning: image_name '{image_name}' provided for prediction does not exist at that local path.")
         # Depending on how predict_mri_path handles non-existent paths (it raises FileNotFoundError),
         # you might want to catch that specifically or let it propagate to a generic error.
         # return jsonify({"error": f"Image file not found at path: {image_name}"}), 404

    try:
        result = predict_mri_path(image_name) # predict_mri_path expects a file path
        if 'error' in result:
            # Handle errors reported by the prediction function (e.g., model not loaded, preprocessing error)
            return jsonify(result), 500 # Internal Server Error or appropriate status
        return jsonify(result), 200
    except FileNotFoundError:
        # This catches FileNotFoundError if image_name is a local path and not found by nib.load inside preprocess
        return jsonify({"error": f"Image file not found or inaccessible: {image_name}"}), 404
    except Exception as e:
        # Catch-all for other unexpected errors during prediction
        print(f"Unexpected error during /predict for image '{image_name}': {e}")
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    # When running locally (python app.py), Flask's development server is used.
    # The Dockerfile currently uses CMD ["python", "app.py"], so this will also be the entry point in Docker.
    # For production, you'd typically use a WSGI server like Gunicorn (already in requirements.txt).
    # Example: gunicorn --bind 0.0.0.0:5000 app:app
    # The model loading happens above, before app.run(), so if it fails, the app won't start.
    print("Starting Flask development server...")
    app.run(host='0.0.0.0', port=5000, debug=True) # debug=True is useful for development 