# 💧 Water Quality Prediction - End-to-End ML Project

An end-to-end production-ready Machine Learning project that predicts the **potability of water** based on its chemical properties.
This project covers the full lifecycle from **data ingestion → training → deployment → monitoring → CI/CD automation**.

---

## 🚀 Project Overview

This system takes water quality parameters such as pH, hardness, dissolved solids, etc., and predicts whether the water is **potable (safe to drink)** or **not potable**.

The project is designed with a **production-grade architecture**, including:

* Modular pipeline design
* Cloud deployment (AWS EC2)
* CI/CD automation
* Real-time monitoring (Prometheus + Grafana)
* Experiment tracking (MLflow + DagsHub)
* Data versioning (DVC)

---

## 🧠 Key Features

* ✅ End-to-End ML Pipeline
* ✅ Artifact-driven architecture
* ✅ Data versioning using DVC
* ✅ Hyperparameter tuning using MLflow
* ✅ Model versioning & comparison
* ✅ FastAPI-based prediction service
* ✅ Dockerized deployment
* ✅ CI/CD with GitHub Actions
* ✅ Monitoring using Prometheus & Grafana

---

## 🏗️ Project Architecture

### 🔹 Training Pipeline

```
Data Source (S3 + DVC)
        ↓
Data Ingestion
        ↓
Data Validation
        ↓
Data Transformation
        ↓
Model Trainer (MLflow tuning)
        ↓
Model Evaluation (compare with production)
        ↓
Model Pusher (S3 deployment)
```

### 🔹 Prediction Pipeline

```
User Input → Web Form → FastAPI Backend → DataFrame Creation
→ Load Model from S3 → Model Inference → Output (Potable / Not Potable)
```

### 🔹 CI/CD Pipeline

```
GitHub Push → GitHub Actions → SSH to EC2 → Docker Compose Deploy
→ Live Application (FastAPI + Prometheus + Grafana)
```

---

## ⚙️ Tech Stack

| Category         | Tools Used                  |
| ---------------- | --------------------------- |
| Programming      | Python                      |
| ML Libraries     | Scikit-learn, Pandas, NumPy |
| Backend          | FastAPI                     |
| Experimentation  | MLflow, DagsHub             |
| Data Versioning  | DVC                         |
| Cloud            | AWS S3, EC2                 |
| Containerization | Docker                      |
| CI/CD            | GitHub Actions              |
| Monitoring       | Prometheus, Grafana         |

---

## 📊 Dataset

* Dataset: **Water Potability Dataset**
* Features:

  * pH
  * Hardness
  * Solids
  * Chloramines
  * Sulfate
  * Conductivity
  * Organic Carbon
  * Trihalomethanes
  * Turbidity
* Target:

  * Potability (0 / 1)

---

## 🔬 Model Details

* Algorithm: **Random Forest Classifier**
* Preprocessing:

  * KNN Imputer (for missing values)
  * StandardScaler
* Evaluation Metrics:

  * Accuracy
  * Precision
  * Recall
  * F1 Score

---

## 🧪 Experiment Tracking

* Performed experiments on multiple models:

  * Logistic Regression
  * Decision Tree
  * Random Forest (best)
* Top models tuned using **MLflow**
* Experiments tracked on **DagsHub**

---

## 📦 Artifacts

Each stage outputs structured artifacts:

* DataIngestionArtifact
* DataValidationArtifact
* DataTransformationArtifact
* ModelTrainerArtifact
* ModelEvaluationArtifact
* ModelPusherArtifact

---

## ☁️ Deployment

* Hosted on **AWS EC2**
* Application runs using Docker containers:

  * FastAPI App (Port 8000)
  * Prometheus (Port 9090)
  * Grafana (Port 3000)

---

## 🔄 CI/CD Pipeline

Automated deployment using GitHub Actions:

* Trigger: Push to `main`
* Steps:

  * SSH into EC2
  * Pull latest code
  * Stop existing containers
  * Clean old images
  * Rebuild and start services

---

## 📈 Monitoring

### Prometheus

* Collects application metrics

### Grafana Dashboards

* Total Requests
* Request Rate
* Latency
* CPU Usage

---

## 🖥️ How to Run Locally

### 1. Clone Repository

```bash
git clone https://github.com/geetmantri07/water-potability-prediction.git
cd water-potability-prediction
```

### 2. Create Virtual Environment

```bash
conda create -n water python=3.10
conda activate water
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run Application

```bash
python app.py
```

---

## 🐳 Run with Docker

```bash
docker-compose up --build
```

---

## 📌 Future Improvements

* Add model drift detection
* Add alerting system (Grafana alerts)
* Use Kubernetes for scaling
* Add user authentication

---

## 👨‍💻 Author

**Geet Mantri**

* GitHub: https://github.com/geetmantri07
* LinkedIn: (Add your profile link)

---

## ⭐ Acknowledgements

* Scikit-learn
* FastAPI
* MLflow
* DVC
* AWS
* Prometheus & Grafana

---

## 📢 Note

This project demonstrates a **production-level ML system design** with real-world components including CI/CD, monitoring, and cloud deployment.

---

