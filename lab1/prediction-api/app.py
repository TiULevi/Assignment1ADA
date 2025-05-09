import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from mri_predictor import MRIClassifier

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

classifier = MRIClassifier()

def allowed_file(filename):
    if '.' not in filename:
        return False
    if filename.lower().endswith('.nii.gz'):
        return True
    ext = filename.rsplit('.', 1)[1].lower()
    return ext in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "MRI Prediction API is running"}), 200

@app.route('/predict', methods=['POST'])
def predict_image():
    if classifier.model is None:
        return jsonify({"error": "Model not loaded. API cannot process requests."}), 503
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        temp_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(temp_file_path)
            print(f"File saved temporarily to {temp_file_path}")
            prediction_result = classifier.predict_single_image(temp_file_path)
            if prediction_result is None:
                return jsonify({"error": "Prediction failed due to an internal classifier error."}), 500
            return jsonify(prediction_result), 200
        except Exception as e:
            print(f"Error during file processing or prediction: {e}")
            return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500
        finally:
            if os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                    print(f"Temporary file {temp_file_path} removed.")
                except Exception as e:
                    print(f"Error removing temporary file {temp_file_path}: {e}")
    else:
        return jsonify({"error": f"File type not allowed. Allowed: .nii, .nii.gz, .img. Filename: {file.filename}"}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5001)), debug=True) 