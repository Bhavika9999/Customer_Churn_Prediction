# Customer Churn Prediction Pipeline

## Project Overview
Customer Churn Prediction is a Machine Learning project that predicts whether a customer is likely to leave a telecom service.

The project uses customer information, subscription details, and service usage data to identify potential churn customers. It also provides a REST API using FastAPI to generate predictions for new customers.

## Objectives
- Understand and clean customer data.
- Perform Exploratory Data Analysis (EDA).
- Create meaningful features using Feature Engineering.
- Train and evaluate multiple Machine Learning models.
- Select the best model based on performance and business impact.
- Provide predictions through a REST API.

## Technologies Used
- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- FastAPI and Uvicorn
- Joblib

## Dataset
**Dataset:** Telco Customer Churn Dataset

- Total Records: 7,043
- Target Variable: Churn
- Target Classes: Yes and No

The dataset contains customer information, subscription details, service usage, and billing information.

## Project Workflow
1. Data Understanding
2. Data Cleaning
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. Data Preprocessing
6. Model Training and Evaluation
7. Business Impact Analysis
8. REST API Development using FastAPI

## Feature Engineering
Three new features were created:

- **TenureGroup:** Groups customers based on their tenure.
- **HasFamily:** Identifies whether a customer has a partner or dependents.
- **NumberOfServices:** Counts the services used by a customer.

## Machine Learning Models
The following classification algorithms were trained:

1. Logistic Regression
2. Random Forest Classifier
3. Decision Tree Classifier

Hyperparameter tuning was performed using GridSearchCV with 5-fold cross-validation.

**Best Model:** Logistic Regression

**Cross-Validation ROC-AUC:** 0.8465

## Model Evaluation
The models were evaluated using:

- ROC-AUC Score
- Precision
- Recall
- Precision-Recall AUC

Business impact was analyzed using different prediction thresholds. A threshold of 0.6 was selected as a balanced starting point because it achieved the highest F1-score among the tested thresholds.

## REST API
FastAPI is used to provide predictions through a REST API.

**Available Endpoints:**

- `GET /` – Checks whether the API is running.
- `POST /predict` – Predicts customer churn probability and class.

**API Documentation:**

After starting the API locally, open:

`http://127.0.0.1:8000/docs`

The prediction response includes:

- Churn Probability
- Churn Prediction (0 or 1)
- Result (Yes or No)

## Project Structure

```text
Customer_Churn_Prediction/
│
├── data/
│   ├── Telco-Customer-Churn.csv
│   ├── cleaned_telco_churn.csv
│   └── featured_telco_churn.csv
│
├── 01_data_understanding.py
├── 02_data_cleaning.py
├── 03_eda.py
├── 04_feature_engineering.py
├── 05_preprocessing.py
├── 06_model_training.py
├── 07_api.py
├── 08_business_impact.py
│
├── best_churn_model.pkl
├── requirements.txt
├── .gitignore
└── README.md
```

## How to Run the Project

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_LINK
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the API

```bash
uvicorn 07_api:app --reload
```

### 4. Open API Documentation

Visit:

`http://127.0.0.1:8000/docs`

Use the `POST /predict` endpoint to test customer churn predictions.

## Business Benefits
- Identifies customers who may leave the service.
- Helps businesses plan customer retention strategies.
- Supports data-driven decision-making.
- Helps reduce potential customer loss.

## Conclusion
This project demonstrates an end-to-end Machine Learning workflow, from data understanding and preprocessing to model training, evaluation, business impact analysis, and REST API development.
