## 🚀 Project Overview

The Smart Logistics Decision System aims to optimize delivery operations using data-driven decision making. The system integrates operational data, environmental factors, and predictive modeling to improve route efficiency, delivery time estimation, and cost optimization.

## 🎯 Project Objectives

### 1️⃣ Risk-Based Delay Prediction
- Build a machine learning model that predicts delivery delay probability.
- Interpret model output probability as a structured **risk score**.

### 2️⃣ Risk Tier Classification
Convert predicted delay probability into operational risk levels:

| Probability Range | Risk Level |
|-------------------|------------|
| < 0.40            | Low Risk   |
| 0.40 – 0.70       | Medium Risk|
| > 0.70            | High Risk  |
| > 0.85            | Critical Risk |

### 3️⃣ Rule-Based Risk Overrides
Enhance ML predictions with domain-driven business logic:

- **Weather-Traffic Critical Rule:**  
  If Precipitation > 20mm AND Traffic = Heavy → Risk = Critical

- **Asset Stress Rule:**  
  If Asset_Utilization > 90% → Operational Risk = High

### 4️⃣ Risk-Driven Decision Engine
Attach automated actions based on risk level:

- 🟢 Low Risk → Normal delivery  
- 🟡 Medium Risk → Monitoring + slight route optimization  
- 🔴 High Risk → Re-route vehicle + notify operations  
- ⚫ Critical Risk → Immediate rerouting + AI-generated client notification + fleet redistribution  

### 5️⃣ Simulated Optimization Logic
- Simulate route optimization by adjusting estimated delivery time.
- Simulate fleet redistribution when asset utilization exceeds safe thresholds.

### 6️⃣ AI Communication Layer
Automatically generate customer notifications when risk is High or Critical.


## 📊 Dataset

**Base Dataset:**  
Smart Logistics Supply Chain Dataset 
https://www.kaggle.com/datasets/ziya07/smart-logistics-supply-chain-dataset

The original dataset contains shipment details, delivery timelines, geographic information, and operational attributes.

### Feature Engineering

A precipitation factor was added to the dataset using quantile-based binning to simulate weather impact on logistics performance. This allows the system to incorporate environmental conditions into delivery time and decision modeling.

## 🧠 Current Progress

- Repository initialized  
- Base dataset structured  
- Precipitation feature engineered using quantile binning  
- README documentation created  

## 🛠 Development Phases

---

## 📊 Phase 1 – Data Engineering & Feature Preparation

### 🎯 Objective
Transform the raw logistics dataset into a structured, model-ready dataset suitable for predictive modeling and risk analysis.

---

### 🧹 Data Cleaning

- Removed inconsistencies and invalid records  
- Standardized column formats  
- Ensured numerical consistency for modeling  
- Validated missing values and corrected data types  

---

### 🌧 Feature Engineering

Enhanced the dataset with domain-driven features:

- Generated **Precipitation (mm)** using humidity and temperature  
- Structured environmental factors (Temperature, Humidity, Precipitation)  
- Organized operational variables (Inventory Level, Waiting Time, Utilization metrics)  
- Prepared features for downstream ML modeling  

---

### 🏗 Dataset Structuring

Separated datasets into:

- `data/raw/` → Original dataset  
- `data/processed/` → Cleaned & feature-engineered dataset  

Generated:

- `clean_model_dataset.csv` – Model-ready dataset  

---

### 📓 Reproducible Pipeline

Created:

- `notebooks/phase1_data_engineering.ipynb`

This notebook:

- Performs data cleaning  
- Applies feature engineering logic  
- Generates processed dataset  
- Ensures full reproducibility of Phase 1  

---

### ✅ Output of Phase 1

A structured, validated, and reproducible dataset ready for predictive modeling in Phase 2.

---

## 🤖 Phase 2 – Delay Prediction Model & Risk Scoring

### 🎯 Objective
Develop a machine learning model to predict **delivery delay probability** (`delay_probability`) to serve as the foundation for structured risk classification.

---

### 🧪 Model Benchmarking
Evaluated multiple models:

- Logistic Regression  
- Random Forest  
- Gradient Boosting  
- KNN  
- SVM  
- XGBoost  

Tuning performed using 5-fold cross-validation with ROC-AUC as the primary metric.

---

### 🏆 Final Model
**Selected Model:** Logistic Regression  
Best Parameters:
- C = 0.01  
- penalty = l2  
- solver = lbfgs  
- max_iter = 5000  

Chosen for balanced F1 score, stable ROC-AUC, interpretability, and deployment simplicity.

---

### 📈 Final Performance (Threshold = 0.6)

- Accuracy ≈ 0.77  
- Precision ≈ 0.98  
- Recall ≈ 0.61  
- F1 Score ≈ 0.75  
- ROC-AUC ≈ 0.79  

---

### 📦 Artifacts Generated
- `models/delay_model.pkl`
- `models/scaler.pkl`
- `data/processed/dataset_with_delay_probability.csv`

This completes the predictive layer of the Smart Logistics Decision System.
