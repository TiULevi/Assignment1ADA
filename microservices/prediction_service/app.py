from flask import Flask, request, jsonify
from mri_predictor import predict_mri_path, load_model_from_gcs

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image_path = data.get('image_name')
    if not image_path:
        return jsonify({"error": "No image_name provided"}), 400
    try:
        prediction, confidence = predict_mri_path(image_path)
        return jsonify({
            "prediction": prediction,
            "confidence": confidence
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200

if __name__ == '__main__':
    load_model_from_gcs()
    app.run(host='0.0.0.0', port=5000)
