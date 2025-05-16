# Deployment Instructions

## Build and Deploy REST Services (Cloud Run)
cd microservices/patient_service
gcloud builds submit --tag gcr.io/ada2024-450119/patient-service .
gcloud run deploy patient-service --image gcr.io/ada2024-450119/patient-service --region us-central1 --platform managed

# Repeat for each service

## Deploy Feedback Service as Cloud Function (FaaS)
gcloud functions deploy feedback_handler \
  --runtime python310 \
  --trigger-topic FeedbackSubmittedEvent \
  --entry-point feedback_handler \
  --region us-central1 \
  --source microservices/feedback_service

## Create Pub/Sub Topics
bash microservices/gcp/pubsub_topics.sh
