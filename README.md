# 🚗 MLOps Vehicle Insurance Prediction

An end-to-end MLOps pipeline for predicting vehicle insurance cross-sell opportunities. This project demonstrates industry-standard MLOps practices including MongoDB integration, AWS S3 model registry, CI/CD pipelines with GitHub Actions, and Docker-based deployment on AWS EC2.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Architecture](#project-architecture)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Getting Started](#getting-started)
- [MongoDB Setup](#mongodb-setup)
- [AWS Configuration](#aws-configuration)
- [Pipeline Components](#pipeline-components)
- [CI/CD Deployment](#cicd-deployment)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

This project aims to predict whether a health insurance customer would be interested in purchasing vehicle insurance. The solution implements a complete MLOps workflow from data ingestion to model deployment and monitoring.

### Business Problem

An insurance company needs to build a model to predict whether policyholders from the past year will also be interested in vehicle insurance. This helps the company plan its communication strategy to reach out to customers and optimize its business model and revenue.

---

## 🏗️ Project Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  MongoDB Atlas  │────▶│  Data Ingestion │────▶│ Data Validation │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Model Pusher   │◀────│ Model Evaluation│◀────│  Model Trainer  │
│   (AWS S3)      │     └─────────────────┘     └─────────────────┘
└─────────────────┘                                      ▲
         │                                               │
         ▼                                      ┌─────────────────┐
┌─────────────────┐                             │Data Transformation│
│  Flask App      │                             └─────────────────┘
│  (AWS EC2)      │
└─────────────────┘
```

---

## 📁 Project Structure

```
MLOPS-Project-Vehicle-insurance/
│
├── .github/
│   └── workflows/
│       └── aws.yaml                 # CI/CD pipeline configuration
│
├── config/
│   ├── model.yaml                   # Model configuration
│   └── schema.yaml                  # Dataset schema for validation
│
├── src/
│   ├── __init__.py
│   │
│   ├── components/                  # Pipeline components
│   │   ├── __init__.py
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   ├── model_evaluation.py
│   │   └── model_pusher.py
│   │
│   ├── pipline/                     # Training and prediction pipelines
│   │   ├── __init__.py
│   │   ├── training_pipeline.py
│   │   └── prediction_pipeline.py
│   │
│   ├── configuration/               # Connection configurations
│   │   ├── __init__.py
│   │   ├── mongo_db_connection.py
│   │   └── aws_connection.py
│   │
│   ├── cloud_storage/               # AWS S3 operations
│   │   ├── __init__.py
│   │   └── aws_storage.py
│   │
│   ├── data_access/                 # Data access layer
│   │   ├── __init__.py
│   │   └── proj1_data.py
│   │
│   ├── entity/                      # Data classes and configurations
│   │   ├── __init__.py
│   │   ├── config_entity.py
│   │   ├── artifact_entity.py
│   │   ├── estimator.py
│   │   └── s3_estimator.py
│   │
│   ├── constants/                   # Project constants
│   │   └── __init__.py
│   │
│   ├── utils/                       # Utility functions
│   │   ├── __init__.py
│   │   └── main_utils.py
│   │
│   ├── logger/                      # Logging configuration
│   │   └── __init__.py
│   │
│   └── exception/                   # Custom exception handling
│       └── __init__.py
│
├── notebook/                        # Jupyter notebooks
│   ├── mongoDB_demo.ipynb           # MongoDB data upload
│   └── EDA_Feature_Engineering.ipynb
│
├── static/                          # Static files (CSS, JS)
├── templates/                       # HTML templates for Flask app
├── artifacts/                       # Generated artifacts (in .gitignore)
│
├── app.py                           # Flask application
├── demo.py                          # Pipeline testing script
├── template.py                      # Project template generator
├── requirements.txt                 # Python dependencies
├── setup.py                         # Package setup file
├── pyproject.toml                   # Project configuration
├── Dockerfile                       # Docker configuration
├── .dockerignore                    # Docker ignore file
└── README.md
```

---

## 🛠️ Technologies Used

| Category | Technologies |
|----------|-------------|
| **Programming Language** | Python 3.10 |
| **ML Framework** | Scikit-learn |
| **Data Processing** | Pandas, NumPy |
| **Database** | MongoDB Atlas |
| **Cloud Platform** | AWS (S3, EC2, ECR) |
| **Model Registry** | AWS S3 Bucket |
| **Web Framework** | Flask |
| **Containerization** | Docker |
| **CI/CD** | GitHub Actions |
| **Visualization** | Matplotlib, Seaborn |

---

## ⚙️ Getting Started

### Prerequisites

- Python 3.10
- Conda (recommended)
- Git
- MongoDB Atlas account
- AWS Account

### Step 1: Create Project Template

```bash
python template.py
```

### Step 2: Setup Virtual Environment

```bash
# Create conda environment
conda create -n vehicle python=3.10 -y

# Activate environment
conda activate vehicle

# Install dependencies
pip install -r requirements.txt

# Verify local packages are installed
pip list
```

### Step 3: Configure Environment Variables

**For Bash:**
```bash
export MONGODB_URL="mongodb+srv://<username>:<password>@cluster.mongodb.net/?retryWrites=true&w=majority"
export AWS_ACCESS_KEY_ID="your_access_key"
export AWS_SECRET_ACCESS_KEY="your_secret_key"
```

**For PowerShell:**
```powershell
$env:MONGODB_URL = "mongodb+srv://<username>:<password>@cluster.mongodb.net/?retryWrites=true&w=majority"
$env:AWS_ACCESS_KEY_ID = "your_access_key"
$env:AWS_SECRET_ACCESS_KEY = "your_secret_key"
```

---

## 🍃 MongoDB Setup

1. **Create MongoDB Atlas Account**
   - Sign up at [MongoDB Atlas](https://www.mongodb.com/atlas)
   - Create a new project

2. **Create Cluster**
   - Click "Create Cluster"
   - Select M0 (Free Tier)
   - Click "Create Deployment"

3. **Configure Database User**
   - Set username and password
   - Create DB user

4. **Configure Network Access**
   - Go to "Network Access"
   - Add IP Address: `0.0.0.0/0` (allows access from anywhere)

5. **Get Connection String**
   - Go to project → "Get Connection String" → "Drivers"
   - Select Driver: Python, Version: 3.6 or later
   - Copy connection string and replace `<password>`

6. **Upload Data to MongoDB**
   - Open `notebook/mongoDB_demo.ipynb`
   - Select kernel: Python (vehicle environment)
   - Run cells to push data to MongoDB

7. **Verify Data**
   - Go to MongoDB Atlas → Database → Browse Collections
   - View your data in key-value format

---

## ☁️ AWS Configuration

### IAM User Setup

1. Login to AWS Console
2. Set region to `us-east-1`
3. Go to IAM → Create new user
4. Attach policy: `AdministratorAccess`
5. Create Access Key (CLI) and download CSV

### S3 Bucket Setup

1. Go to S3 Service → Create Bucket
2. **Region:** us-east-1
3. **Bucket Name:** `my-model-mlopsproj`
4. Uncheck "Block all public access"
5. Create Bucket

### Constants Configuration

Update `src/constants/__init__.py`:
```python
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.02
MODEL_BUCKET_NAME = "my-model-mlopsproj"
MODEL_PUSHER_S3_KEY = "model-registry"
```

---

## 📊 Pipeline Components

### 1. Data Ingestion
- Connects to MongoDB using connection URL
- Fetches data in key-value format
- Transforms data to DataFrame
- Splits into train and test sets
- Saves raw data as artifacts

### 2. Data Validation
- Schema validation using `config/schema.yaml`
- Data quality checks
- Data drift detection

### 3. Data Transformation
- Feature engineering
- Handling missing values
- Encoding categorical variables
- Feature scaling

### 4. Model Trainer
- Model training with selected algorithm
- Hyperparameter configuration
- Model artifact generation

### 5. Model Evaluation
- Performance metrics calculation
- Comparison with existing model in S3
- Threshold-based model acceptance (0.02 improvement)

### 6. Model Pusher
- Push accepted model to AWS S3
- Model versioning in S3 bucket

---

## 🚀 CI/CD Deployment

### Docker Setup

The project includes `Dockerfile` and `.dockerignore` for containerization.

### EC2 Instance Setup

1. **Launch EC2 Instance**
   - Name: `vehicledata-machine`
   - Image: Ubuntu Server 24.04 (Free tier)
   - Instance Type: T2 Medium
   - Create key pair: `proj1key`
   - Allow HTTP/HTTPS traffic
   - Storage: 30GB

2. **Install Docker on EC2**
   ```bash
   sudo apt-get update -y
   sudo apt-get upgrade
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   sudo usermod -aG docker ubuntu
   newgrp docker
   ```

### ECR Repository Setup

1. Go to AWS ECR → Create Repository
2. **Region:** us-east-1
3. **Repository Name:** `vehicleproj`
4. Copy and save the URI

### GitHub Self-Hosted Runner

1. Go to GitHub Repository → Settings → Actions → Runners
2. Click "New self-hosted runner"
3. Select OS: Linux
4. Run "Download" commands on EC2
5. Run "Configure" command:
   ```bash
   ./config.sh  # Runner name: self-hosted
   ./run.sh     # Start the runner
   ```

### GitHub Secrets Configuration

Go to GitHub → Settings → Secrets and Variables → Actions → New Repository Secret

| Secret Name | Value |
|-------------|-------|
| `AWS_ACCESS_KEY_ID` | Your AWS Access Key |
| `AWS_SECRET_ACCESS_KEY` | Your AWS Secret Key |
| `AWS_DEFAULT_REGION` | us-east-1 |
| `ECR_REPO` | ECR Repository URI |

### Configure EC2 Security Group

1. Go to EC2 Instance → Security → Security Groups
2. Edit Inbound Rules → Add Rule
3. **Type:** Custom TCP
4. **Port Range:** 5080
5. **Source:** 0.0.0.0/0
6. Save Rules

### Trigger Deployment

CI/CD pipeline triggers automatically on push to main branch.

---

## 🖥️ Usage

### Run Training Pipeline

```bash
python demo.py
```

### Start Web Application

```bash
python app.py
```

### Access Application

- **Local:** `http://localhost:5080`
- **EC2:** `http://<ec2-public-ip>:5080`

### Endpoints

| Endpoint | Description |
|----------|-------------|
| `/` | Home page with prediction form |
| `/training` | Trigger model training |
| `/predict` | Make predictions |

---

## 📈 Features Used

| Feature | Description |
|---------|-------------|
| Gender | Gender of the customer |
| Age | Age of the customer |
| Driving_License | Whether customer has driving license |
| Region_Code | Unique code for customer's region |
| Previously_Insured | Already has vehicle insurance |
| Vehicle_Age | Age of the vehicle |
| Vehicle_Damage | Vehicle was damaged in past |
| Annual_Premium | Amount paid as premium |
| Policy_Sales_Channel | Channel for policy outreach |
| Vintage | Number of days customer associated |

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📧 Contact

**Author:** Anjit  
**Project Link:** [https://github.com/taanjit/MLOPS-Project-Vehicle-insurance](https://github.com/taanjit/MLOPS-Project-Vehicle-insurance)

---

<p align="center">
  Made with ❤️ for MLOps Learning
</p>
