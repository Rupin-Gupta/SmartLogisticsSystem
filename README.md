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
- ROC-AUC ≈ 0.80  

---

### 📦 Artifacts Generated
- `models/delay_model.pkl`
- `models/scaler.pkl`
- `data/processed/dataset_with_delay_probability.csv`

This completes the predictive layer of the Smart Logistics Decision System.

---

## ⚖️ Phase 3 – Risk Classification & Decision Layer

### 🎯 Objective
Transform predicted delay probabilities into structured operational risk levels and integrate business rule overrides to create a robust decision engine.

---

### 🔄 Architecture Overview

Phase 3 converts:

Model Output → Risk Tier → Rule Overrides → Final Risk Level

Pipeline:

1. `delay_probability` (from Phase 2 model)
2. Probability-to-risk mapping
3. Rule-based escalation
4. Final operational risk classification

This creates a hybrid ML + domain-intelligence system.

---

### 📊 Step 3.1 – Probability → Risk Mapping

Delay probability is converted into structured risk tiers:

- `< 0.40` → Low  
- `0.40 – 0.70` → Medium  
- `> 0.70` → High  
- `> 0.85` → Critical  

Implemented via:

`get_risk_level(probability)`

Generated column:

`ml_risk_level`

This ensures model output becomes actionable.

---

### 🏢 Step 3.2 – Rule-Based Risk Overrides

To enhance robustness, business logic rules were introduced.

#### Rule 1 – Weather-Traffic Escalation
If:
- Precipitation (mm) > 15  
- Traffic_Status_Heavy == 1  

→ Risk escalated to **Critical**

#### Rule 2 – Asset Stress Escalation
If:
- Asset_Utilization > 90  

→ Risk escalated to **High**

Final risk level is determined as the maximum severity between:

- ML-derived risk  
- Rule-based escalation  

Generated column:

`final_risk_level`

---

### 🧠 Why Combine ML + Rule Logic?

Pure ML models may miss rare but operationally dangerous scenarios.

By integrating domain rules:

- Critical weather conditions are never ignored  
- Fleet stress is proactively managed  
- Operational safety is prioritized  
- Decision robustness increases  

This design reflects real-world logistics intelligence systems.

---

### 📈 Output

Generated:

`data/processed/dataset_with_risk_levels.csv`

Risk distribution example:

- Medium ≈ 32%  
- High ≈ 24%
- Critical ≈ 24%  
- Low ≈ 20%

Phase 3 completes the structured risk engine of the Smart Logistics Decision System.

---

## 🚦 Phase 4 – Risk-Driven Decision Engine

### 🎯 Objective
Translate structured risk levels into operational actions using a mathematically grounded and system-aware decision framework.

Phase 4 connects predictive intelligence (Phases 2–3) to executable logistics decisions.

---

### 🔁 Risk → Action Mapping

Risk levels are converted into structured operational actions:

- **Low** → `A_Normal`
- **Medium** → `B_Monitor`
- **High** → `C_Reroute_Notify`
- **Critical** → `D_Reroute_Notify_Redistribute`
- **High + Asset_Utilization > 90%** → Escalated to `D_Reroute_Notify_Redistribute`

This ensures fleet stress conditions can trigger escalation even when base risk is High.

---

### ⏱ Dynamic Baseline ETA Modeling

Baseline ETA is computed mathematically, not heuristically:

- `operational_base_time` derived from mean waiting time
- `traffic_delay_factor` computed using average delay probability grouped by traffic level
- `baseline_eta` calculated as:

Baseline ETA = operational_base_time × traffic_delay_factor

This ensures ETA reflects real model-driven traffic impact rather than arbitrary assumptions.

---

### 🛣 Simulated Route Optimization

When rerouting is triggered:

- Improvement factor calculated from difference between original traffic factor and clear-traffic factor
- Optimized ETA adjusted proportionally
- Ensured optimized ETA logically falls between heavy and clear traffic bounds
- No artificial traffic-status reassignment

This preserves mathematical consistency while simulating realistic route improvement.

---

### 📊 Utilization Impact Analysis

Fleet stress behavior modeled analytically:

- Asset_Utilization bucketed into:
  - (0–70]
  - (70–90]
  - (90–100]

- Computed `stress_gap` between high and medium utilization segments
- Redistribution logic applied only if impact exceeds defined threshold
- Redistribution skipped if operational improvement is negligible

This prevents unnecessary fleet movements and avoids overreaction.

---

### 💬 AI-Generated Customer Notifications (yet to do)

When action includes `"Notify"`:

- Structured message automatically generated
- Context-aware (weather, traffic, utilization stress)

Example output:
"Due to heavy traffic conditions, your shipment has been proactively rerouted to minimize delay."

---

### 🧠 Architectural Significance

Phase 4 transforms the system into a full decision intelligence pipeline:

Model → Probability → Risk → Action → ETA Adjustment → Notification

This completes the operational decision layer of the Smart Logistics Decision System.
