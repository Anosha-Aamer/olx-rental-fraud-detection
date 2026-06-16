# olx-rental-fraud-detection
Machine learning-based OLX rental fraud detection system using Logistic Regression, Decision Tree, and Random Forest with Streamlit deployment for real-time prediction.

# OLX Rental Fraud Detection System

## Project Overview

This project is a Machine Learning-based system designed to detect fraudulent rental listings on online platforms such as OLX Pakistan. The system analyzes property details and predicts whether a rental listing is fraudulent or legitimate.

The project follows a complete machine learning workflow including data collection, preprocessing, exploratory data analysis, model implementation, evaluation, and deployment through a Streamlit application.

The project includes:

* Web scraping and dataset collection
* Data preprocessing and visualization
* Multiple machine learning models
* Model evaluation and comparison
* Random Forest model selection
* Streamlit-based application deployment
* Real-time prediction system

---

## Problem Statement

Online rental platforms often contain fake or misleading listings that can cause financial loss and inconvenience to users. Fraudulent property advertisements may include unrealistic prices, fake property details, or misleading information intended to deceive users.

Detecting such listings manually is difficult and time-consuming. The objective of this project is to build an intelligent machine learning system capable of automatically identifying suspicious rental listings.

---

## Data Collection

The dataset was collected using web scraping techniques from OLX Pakistan rental listings.

Relevant information extracted from property advertisements included:

* Property Price
* Area (sqft)
* Number of Bedrooms
* Number of Bathrooms
* Listing Label (Fraud/Legitimate)

The collected raw data was then processed and prepared for machine learning implementation.

---

## Dataset Features

The following features were used in the model:

| Feature    | Description                  |
| ---------- | ---------------------------- |
| Price      | Rental property price        |
| Area_sqft  | Property area in square feet |
| Bed_Count  | Number of bedrooms           |
| Bath_Count | Number of bathrooms          |

Target Variable:

**Label**

* 0 → Legitimate Listing
* 1 → Fraud Listing

---

## Data Preprocessing

The following preprocessing steps were applied:

* Selected important features
* Removed inconsistencies from the dataset
* Split data into training and testing sets (80/20)
* Applied feature scaling using StandardScaler
* Prepared data for machine learning algorithms

---

## Exploratory Data Analysis (EDA)

Data analysis and visualization techniques were used to better understand the dataset.

EDA included:

* Fraud vs Legitimate listing distribution
* Property price distribution
* Correlation heatmap

Visualizations helped identify relationships between variables and understand feature behavior.

---

## Machine Learning Models Implemented

The following classification algorithms were trained and evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest

---

## Model Evaluation Metrics

Models were evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* Cross Validation

### Performance Results

| Model               | Accuracy | Precision | Recall | F1 Score |
| ------------------- | -------- | --------- | ------ | -------- |
| Logistic Regression | 0.78     | 0.66      | 0.71   | 0.68     |
| Decision Tree       | 0.92     | 0.84      | 0.94   | 0.89     |
| Random Forest       | 0.94     | 0.89      | 0.94   | 0.91     |

Random Forest achieved the highest performance among all implemented models and was selected as the final model.

---

## Model Selection

Random Forest was selected because it demonstrated superior performance across all evaluation metrics.

Reasons for selecting Random Forest:

* Highest prediction accuracy
* Better precision and recall values
* Strong F1-score performance
* Handles non-linear relationships effectively
* More robust against overfitting

---

## Cross Validation

Cross-validation was applied to validate the robustness and generalization capability of the model.

This process ensured that the model performs consistently across different subsets of the data and reduces the risk of overfitting.

---

## Application Integration

A Streamlit-based web application was developed to provide a user-friendly interface for real-time prediction.

Users can:

* Enter property details
* Detect fraud in real time
* View prediction confidence

### Input Fields

* Price
* Area (sqft)
* Bedrooms
* Bathrooms

### Output

* Fraud Listing Detected
* Legitimate Listing

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* Streamlit
* Pickle
* Web Scraping (BeautifulSoup / Requests)

---

## Project Structure

```text
OLX-Rental-Fraud-Detection/
│ ├── image/
│ ├── app_ui.png
│ ├── class_distribution.png
│ ├── confusion_matrix.png
│ ├── frauddetect.png
│ ├── heatmap.png
│ ├── legitimate.png
│ └── price_distribution.png
│
├── app.py
├── data_cleaned.ipynb
├── data_scraped.ipynb
├── model_and_application.ipynb
├── model.pkl
├── scaler.pkl
├── OLXdata_full1.csv
├── OLXdata_labelled.csv
├── urls_backup.pkl
└── README.md
```

---

## Running the Project

Install required libraries:
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import streamlit as st 
import pickle 
from sklearn.model_selection import train_test_split, cross_val_score 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

Run Streamlit application:

```bash
python -m streamlit run app.py
```

Open browser:

```text
http://localhost:8501
```

## Author

Anosha Aamer

BS Data Science / Artificial Intelligence

National University of Computer and Emerging Sciences
