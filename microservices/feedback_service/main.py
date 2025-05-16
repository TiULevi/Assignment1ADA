import base64
import json
from flask import Flask, request, jsonify

app = Flask(__name__)
FEEDBACKS = []

@app.route('/feedback', methods=['POST'])
def submit_feedback():
    data = request.get_json()
    FEEDBACKS.append(data)
    # In real: publish to Pub/Sub FeedbackSubmittedEvent
    return jsonify({"status": "feedback received", "feedback": data}), 201

@app.route('/feedbacks', methods=['GET'])
def list_feedbacks():
    return jsonify(FEEDBACKS), 200

def feedback_handler(event, context):
    message = base64.b64decode(event['data']).decode('utf-8')
    print("Received feedback event:", message)
    return {"status": "feedback processed"}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8083)
