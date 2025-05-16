from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

PREDICTIONS = {
    "pred1": {
        "patient_id": "p1",
        "diagnosis": "Alzheimer's likely",
        "confidence": 0.92,
        "metrics": {"accuracy": 0.91, "precision": 0.89, "recall": 0.93, "f1": 0.91}
    }
}

@app.route('/report/<prediction_id>', methods=['GET'])
def get_report(prediction_id):
    pred = PREDICTIONS.get(prediction_id)
    if not pred:
        return jsonify({"error": "Not found"}), 404
    return jsonify(pred), 200

@app.route('/dashboard/<prediction_id>', methods=['GET'])
def dashboard(prediction_id):
    pred = PREDICTIONS.get(prediction_id)
    if not pred:
        return "Prediction not found", 404
    html = """
    <html>
    <head><title>Prediction Dashboard</title></head>
    <body>
      <h2>Prediction Results for Patient: {{patient_id}}</h2>
      <p><b>Diagnosis:</b> {{diagnosis}}</p>
      <p><b>Confidence:</b> {{confidence}}</p>
      <h3>Model Performance Metrics</h3>
      <ul>
        <li>Accuracy: {{metrics['accuracy']}}</li>
        <li>Precision: {{metrics['precision']}}</li>
        <li>Recall: {{metrics['recall']}}</li>
        <li>F1 Score: {{metrics['f1']}}</li>
      </ul>
    </body>
    </html>
    """
    return render_template_string(html, **pred)

@app.route('/metrics', methods=['GET'])
def get_metrics():
    metrics = {
        "accuracy": 0.91,
        "precision": 0.89,
        "recall": 0.93,
        "f1": 0.91
    }
    return jsonify(metrics), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8084)
