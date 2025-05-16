from flask import Flask, request, jsonify
import os

app = Flask(__name__)
UPLOAD_FOLDER = '/tmp/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload_mri():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if not file.filename.endswith('.nii.gz'):
        return jsonify({"error": "Invalid file type"}), 400
    save_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(save_path)
    # Simulate event: MRIUploadedEvent (in real, publish to Pub/Sub)
    return jsonify({"status": "uploaded", "path": save_path}), 201

@app.route('/validate', methods=['POST'])
def validate_scan():
    data = request.get_json()
    path = data.get('path')
    if not path or not os.path.exists(path):
        return jsonify({"error": "File not found"}), 404
    # Simulate validation
    return jsonify({"status": "validated", "path": path}), 200

@app.route('/uploads', methods=['GET'])
def list_uploads():
    files = os.listdir(UPLOAD_FOLDER)
    return jsonify({"uploads": files}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8082)
