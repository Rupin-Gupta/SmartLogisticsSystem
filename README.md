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

### Phase 1 – Data Engineering
- Cleaned raw dataset
- Engineered precipitation feature
- Extracted structured features
- Generated processed dataset
- Created reproducible Jupyter notebook pipeline
