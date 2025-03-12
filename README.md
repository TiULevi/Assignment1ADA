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

---
## **8. Files & Directories**

```bash
📂 ADA2025_Project/
 ├── 📂 training_service/
 │   ├── train.py
 │   ├── Dockerfile
 │   ├── requirements.txt
 ├── 📂 prediction_service/
 │   ├── predict.py
 │   ├── Dockerfile
 │   ├── requirements.txt
 ├── 📂 ui/
 │   ├── app.js
 │   ├── package.json
 ├── 📂 kubernetes/
 │   ├── training-deployment.yaml
 │   ├── prediction-deployment.yaml
 │   ├── service.yaml
 ├── 📂 cloudbuild/
 │   ├── cloudbuild.yaml
 │   ├── cloudbuild-triggers.yaml
 ├── README.md
```
