# Assignment1ADA

# **Service-Oriented Application for MRI Brain Image Analysis**

## **1. Project Overview**
This project implements a **microservices-based** Service-Oriented Application (SOA) using **Domain-Driven Design (DDD)** principles. The application processes **MRI brain images** for patient diagnosis using **Google Cloud Storage, BigQuery, Docker, Kubernetes, and CI/CD pipelines**.

### **Key Components:**
- **Training API:** Trains multiple machine learning models. During Data Engineering we were told to train multiple models and not only 1
- **Prediction API:** Performs batch inference on MRI images.
- **Prediction UI:** Frontend interface for users.
- **CI/CD Pipelines:** Automated deployment with Google Cloud Build in a Kubernetes Cluster.

---
## **2. Architecture Overview**
The system follows a **microservices architecture** with Kubernetes-based containerized deployment.

### **2.1 High-Level Components**
- **Data Storage:** Pre-processed MRI images in **Google Cloud Storage (GCS)** and metadata in **BigQuery**.
- **Training API:** Trains ML models and saves them in Cloud Storage.
- **Prediction API:** Loads trained models and performs inference.
- **Prediction UI:** User interface for interacting with the Prediction API.
- **Monitoring & Logging:** Cloud Logging & Prometheus/Grafana.

### **2.2 Technology Stack**

| Component        | Technology Used          |
|-----------------|-------------------------|
| Data Storage    | Google Cloud Storage, BigQuery |
| Model Training  | TensorFlow/PyTorch, Vertex AI |
| APIs            | RESTful APIs        |
| UI              | Not chosen yet (flask?)       |
| Containerization| Docker                   |
| Orchestration   | Kubernetes (GKE)        |
| CI/CD           | Cloud Build, Cloud Functions |
| Monitoring      | Prometheus, Grafana      |

---
## **3. Microservices & Domain-Driven Design (DDD)**
Applying **DDD principles**, the system is broken into subdomains:

### **3.1 Identified Subdomains**
#### **Data Management Subdomain**
- Manages pre-processed MRI data storage & retrieval.
- Uses **BigQuery for metadata** & **Cloud Storage for images**.

#### **Training Subdomain**
- Trains ML models on MRI images.
- Saves models in **Cloud Storage**.
- Exposes APIs for training & evaluation.

#### **Prediction Subdomain**
- Loads trained models and serves inference requests.
- Supports **batch-processing** of MRI images.

#### **UI/Visualization Subdomain**
- Web-based frontend for submitting images & viewing predictions.

---
## **4. Microservices Architecture**
Each microservice communicates via **RESTful APIs**.

### **4.1 Training API**
#### **Responsibilities:**
- Accepts training requests.
- Loads pre-processed images from Cloud Storage.
- Trains an ML model & saves it.
- Stores model metadata in BigQuery.

#### **Endpoints:**

| Method | Endpoint          | Description |
|--------|------------------|-------------|
| POST   | `/train`          | Start training a new model |
| GET    | `/models`         | List available models |
| GET    | `/models/{id}`    | Get model details |

### **4.2 Prediction API**
#### **Responsibilities:**
- Loads pre-trained models from Cloud Storage.
- Accepts MRI images for batch inference.
- Stores logs in BigQuery.

#### **Endpoints:**

| Method | Endpoint          | Description |
|--------|------------------|-------------|
| POST   | `/predict`        | Perform MRI image prediction |
| GET    | `/predictions/{id}` | Retrieve prediction results |

### **4.3 Prediction UI**
#### **Responsibilities:**
- Upload MRI images & request predictions.
- Displays batch processing results.
- User authentication (**OAuth2/Google Auth**).

#### **Features:**
- Upload MRI scans.
- View batch-processing results.
- Model performance visualization.

---
## **5. Kubernetes & Deployment Strategy**
This project uses **Google Kubernetes Engine (GKE)** with **auto-scaling**.

### **5.1 Kubernetes Architecture**

| Service           | Deployment  | Replicas | Storage  |
|------------------|------------|----------|---------|
| Training API     | Python API | 3        | Cloud Storage, BigQuery |
| Prediction API   | Python API | 3        | Cloud Storage, BigQuery |
| Prediction UI    | React App  | 2        | Cloud Storage |
| Logging & Metrics | Prometheus | 1        | Cloud Logging |

### **5.2 Kubernetes Resources**
- **Deployments:** Manage Training API, Prediction API, UI.
- **Services:** Expose APIs internally within Kubernetes.
- **Ingress:** Allow external access to the UI & APIs.
- **Persistent Volumes:** Store logs & model artifacts.

---
## **6. CI/CD & Cloud Build Pipelines**
Automated pipelines using **Google Cloud Build**.

### **6.1 Cloud Build Steps**
1. **Push to GitHub** → Triggers **Cloud Build**.
2. **Build & Push Docker Images** to **Google Artifact Registry**.
3. **Deploy to Kubernetes Cluster** using **kubectl**.

### **6.2 Cloud Build Triggers**

| Trigger | Action |
|---------|--------|
| Push to `main` | Builds & deploys all services |
| PR to `main` | Runs tests & security scans |
| Manual Trigger | Deploys specific services |

---
## **7. Logging, Monitoring & Security**
- **Logging:** Google Cloud Logging, ELK Stack.
- **Monitoring:** Prometheus, Grafana.
- **Security:** Role-Based Access Control (RBAC), Cloud IAM.

#Stappenplan
# Alzheimer's Diagnosis Microservices: Full Project Scaffold
# =======================================
# This scaffold covers all key components required for Assignment 2.

# ✅ Assume this structure will be zipped and deployed, following GCP standards.
# ✅ Your AI model is trained in a notebook, and you simulate classification in a microservice (REST or FaaS).
# ✅ MRI images are preprocessed; training is simulated.

# ============================================================
# I. MICROSERVICE STRUCTURE OVERVIEW
# ============================================================
# - patient-service          (REST - Cloud Run)
# - data-ingestion-service   (REST - Cloud Run)
# - ai-classification-fn     (FaaS - Cloud Function, triggered by Pub/Sub)
# - feedback-service-fn      (FaaS - Cloud Function, triggered by HTTP)
# - report-service           (REST - Cloud Run)
# - model-management-service (REST - Cloud Run)
# - common/                  (Shared data structures)

# ============================================================
# II. EXAMPLE FILES (ONLY KEY FILES SHOWN HERE)
# ============================================================

# =============================
# patient-service/main.py
# =============================
from flask import Flask, request, jsonify
app = Flask(__name__)

patients = {}  # Simulated storage

@app.route('/patients', methods=['POST'])
def add_patient():
    data = request.json
    patient_id = data['id']
    patients[patient_id] = data
    return jsonify({'status': 'created'}), 201

@app.route('/patients/<patient_id>', methods=['GET'])
def get_patient(patient_id):
    return jsonify(patients.get(patient_id, {}))

@app.route('/patients/<patient_id>', methods=['PUT'])
def update_patient(patient_id):
    data = request.json
    patients[patient_id] = data
    return jsonify({'status': 'updated'})

if __name__ == '__main__':
    app.run(debug=True)


# =============================
# Dockerfile (for REST services)
# =============================
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "main.py"]


# =============================
# requirements.txt (for REST services)
# =============================
Flask==2.2.5
google-cloud-pubsub


# =============================
# ai-classification-fn/main.py
# =============================
import base64
import json

def classify(event, context):
    message = base64.b64decode(event['data']).decode('utf-8')
    print("Received MRIUploadedEvent:", message)

    # Simulate classification
    prediction = {"diagnosis": "Alzheimer's likely", "confidence": 0.92}
    print("Prediction generated:", prediction)

    # Normally, publish PredictionGeneratedEvent here using Pub/Sub client


# =============================
# ai-classification-fn/function.json
# =============================
{
  "name": "ai-classify",
  "entryPoint": "classify",
  "runtime": "python310",
  "trigger": {
    "eventType": "google.pubsub.topic.publish",
    "resource": "projects/YOUR_PROJECT_ID/topics/MRIUploadedEvent"
  }
}

# NOTE: Replace YOUR_PROJECT_ID with your GCP project ID


# =============================
# gcp/cloudbuild.yaml
# =============================
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/YOUR_PROJECT_ID/patient-service', '.']
    dir: 'patient-service'

  - name: 'gcr.io/cloud-builders/gcloud'
    args: ['run', 'deploy', 'patient-service',
           '--image', 'gcr.io/YOUR_PROJECT_ID/patient-service',
           '--region', 'europe-west1', '--platform', 'managed']

# NOTE: Replace YOUR_PROJECT_ID with your actual project ID


# =============================
# gcp/pubsub-topics.sh
# =============================
#!/bin/bash
gcloud pubsub topics create MRIUploadedEvent
gcloud pubsub topics create PredictionGeneratedEvent
gcloud pubsub topics create FeedbackSubmittedEvent


# =============================
# gcp/secrets.env
# =============================
# Placeholder for secrets like bucket names or model URI
MODEL_BUCKET=gs://YOUR_BUCKET_NAME/model.pkl
# NOTE: Replace YOUR_BUCKET_NAME


# =============================
# common/models.py
# =============================
from pydantic import BaseModel

class Patient(BaseModel):
    id: str
    age: int
    gender: int
    mmse: float

class DiagnosisPrediction(BaseModel):
    diagnosis: str
    confidence: float


# ============================================================
# III. FINAL CHECKLIST FOR YOU
# ============================================================
# [ ] Replace ALL placeholders like YOUR_PROJECT_ID or YOUR_BUCKET_NAME
# [ ] Ensure you enable Cloud Functions, Pub/Sub, and Cloud Run in GCP
# [ ] Store trained model in GCS manually
# [ ] Setup permissions for Cloud Run + Pub/Sub
# [ ] Add full README.md with run and test instructions

