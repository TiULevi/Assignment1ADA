import base64
import json
from flask import Flask, request, jsonify
from google.cloud import pubsub_v1
import os

app = Flask(__name__)
FEEDBACKS = []

# GCP Pub/Sub configuration
PROJECT_ID = "ada2024-450119"
TOPIC_ID = "FeedbackSubmittedEvent"

@app.route('/feedback', methods=['POST'])
def submit_feedback():
    data = request.get_json()
    FEEDBACKS.append(data)
    # Publish FeedbackSubmittedEvent to Pub/Sub
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(PROJECT_ID, TOPIC_ID)
    message_json = json.dumps(data).encode("utf-8")
    future = publisher.publish(topic_path, message_json)
    future.result()  # Wait for publish to complete
    return jsonify({"status": "feedback received", "feedback": data}), 201

@app.route('/feedbacks', methods=['GET'])
def list_feedbacks():
    return jsonify(FEEDBACKS), 200

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200

def feedback_handler(event, context):
    message = base64.b64decode(event['data']).decode('utf-8')
    print("Received feedback event:", message)
    return {"status": "feedback processed"}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8083)
