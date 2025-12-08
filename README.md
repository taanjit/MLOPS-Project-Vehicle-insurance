# 🚗 MLOps Vehicle Insurance Prediction

An end-to-end MLOps pipeline for predicting vehicle insurance cross-sell opportunities. This project demonstrates industry-standard MLOps practices including data versioning, experiment tracking, model training, CI/CD pipelines, and model deployment.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Architecture](#project-architecture)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Pipeline Stages](#pipeline-stages)
- [Model Training](#model-training)
- [Deployment](#deployment)
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
│   Data Source   │────▶│  Data Ingestion │────▶│ Data Validation │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Deployment    │◀────│ Model Evaluation│◀────│  Model Training │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                                               ▲
         ▼                                               │
┌─────────────────┐                             ┌─────────────────┐
│   Monitoring    │                             │Data Transformation│
└─────────────────┘                             └─────────────────┘
```

---

## 📁 Project Structure

```
MLOPS-Project-Vehicle-insurance/
│
├── .github/
│   └── workflows/           # CI/CD pipeline configurations
│
├── config/
│   └── config.yaml          # Configuration parameters
│
├── src/
│   ├── components/          # Pipeline components
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   └── model_evaluation.py
│   │
│   ├── pipeline/            # Training and prediction pipelines
│   │   ├── training_pipeline.py
│   │   └── prediction_pipeline.py
│   │
│   ├── entity/              # Data classes and configurations
│   ├── constants/           # Project constants
│   ├── utils/               # Utility functions
│   ├── logger/              # Logging configuration
│   └── exception/           # Custom exception handling
│
├── notebooks/               # Jupyter notebooks for EDA
├── artifacts/               # Generated artifacts (models, data)
├── logs/                    # Application logs
├── templates/               # HTML templates for web app
├── static/                  # Static files (CSS, JS)
│
├── app.py                   # Flask/FastAPI application
├── main.py                  # Pipeline execution entry point
├── requirements.txt         # Python dependencies
├── setup.py                 # Package setup file
├── Dockerfile               # Docker configuration
├── docker-compose.yaml      # Docker compose configuration
└── README.md
```

---

## 🛠️ Technologies Used

| Category | Technologies |
|----------|-------------|
| **Programming Language** | Python 3.9+ |
| **ML Framework** | Scikit-learn, XGBoost, LightGBM |
| **Data Processing** | Pandas, NumPy |
| **Experiment Tracking** | MLflow |
| **Data Versioning** | DVC |
| **Model Registry** | MLflow Model Registry |
| **Web Framework** | Flask / FastAPI |
| **Containerization** | Docker |
| **Cloud Platform** | AWS (S3, EC2, ECR) |
| **CI/CD** | GitHub Actions |
| **Database** | MongoDB |
| **Visualization** | Matplotlib, Seaborn |

---

## ⚙️ Installation

### Prerequisites

- Python 3.9 or higher
- Git
- Docker (optional)
- AWS CLI configured (for cloud deployment)

### Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/MLOPS-Project-Vehicle-insurance.git
   cd MLOPS-Project-Vehicle-insurance
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

### Docker Setup

```bash
docker build -t vehicle-insurance-mlops .
docker run -p 8080:8080 vehicle-insurance-mlops
```

---

## 🚀 Usage

### Run Training Pipeline

```bash
python main.py
```

### Run Individual Components

```python
from src.pipeline.training_pipeline import TrainingPipeline

pipeline = TrainingPipeline()
pipeline.run_pipeline()
```

### Start Web Application

```bash
python app.py
```

Access the application at `http://localhost:8080`

### Make Predictions via API

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Gender": "Male",
    "Age": 35,
    "Driving_License": 1,
    "Region_Code": 28,
    "Previously_Insured": 0,
    "Vehicle_Age": "1-2 Year",
    "Vehicle_Damage": "Yes",
    "Annual_Premium": 30000,
    "Policy_Sales_Channel": 26,
    "Vintage": 150
  }'
```

---

## 📊 Pipeline Stages

### 1. Data Ingestion
- Fetches data from MongoDB/data source
- Splits data into train and test sets
- Saves raw data as artifacts

### 2. Data Validation
- Schema validation
- Data drift detection
- Quality checks

### 3. Data Transformation
- Feature engineering
- Handling missing values
- Encoding categorical variables
- Feature scaling

### 4. Model Training
- Multiple algorithm experimentation
- Hyperparameter tuning
- Model selection

### 5. Model Evaluation
- Performance metrics calculation
- Model comparison
- Threshold optimization

---

## 📈 Model Training

### Features Used

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

### Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| Random Forest | - | - | - | - | - |
| XGBoost | - | - | - | - | - |
| LightGBM | - | - | - | - | - |

*Results will be updated after training*

---

## 🌐 Deployment

### AWS Deployment Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   GitHub     │────▶│GitHub Actions│────▶│   AWS ECR    │
│  Repository  │     │   (CI/CD)    │     │  (Container) │
└──────────────┘     └──────────────┘     └──────────────┘
                                                  │
                                                  ▼
                     ┌──────────────┐     ┌──────────────┐
                     │  End Users   │◀────│   AWS EC2    │
                     └──────────────┘     │ (Application)│
                                          └──────────────┘
```

### Deployment Steps

1. Configure AWS credentials
2. Create ECR repository
3. Set up EC2 instance
4. Configure GitHub secrets
5. Push to main branch to trigger deployment

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
**Project Link:** [https://github.com/your-username/MLOPS-Project-Vehicle-insurance](https://github.com/your-username/MLOPS-Project-Vehicle-insurance)

---

<p align="center">
  Made with ❤️ for MLOps Learning
</p>

