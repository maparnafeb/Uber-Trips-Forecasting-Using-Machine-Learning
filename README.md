# 🚕 Uber Trips Forecasting Using Machine Learning

## 📌 Project Overview

This project focuses on forecasting the number of Uber pickups in New York City using historical data. The forecasting is achieved using various machine learning models such as XGBoost, Random Forest, and Gradient Boosted Trees, followed by an ensemble technique to combine the strengths of all three models.

## 🎯 Objective

- Analyze historical Uber trip data to identify hourly trends.
- Forecast future ride demand using time series-based machine learning models.
- Compare and evaluate model performance using MAPE (Mean Absolute Percentage Error).
- Implement an ensemble model to improve accuracy and robustness.

## 📁 Dataset

- **Source**: NYC Taxi & Limousine Commission (TLC)
- **File Used**: `Uber-Jan-Feb-FOIL.csv`
- **Features**:
  - `Dispatching_base_num`: TLC base company code of the dispatch
  - `Pickup_date`: Date and time of pickup
  - `Affiliated_base_num`: Affiliated TLC base code
  - `locationID`: Pickup location identifier

### 📊 Data Preprocessing Steps

- Parsed datetime from `Pickup_date`
- Resampled data to **hourly frequency** for time series modeling
- Created lag-based features (previous 24 hours as input)
- Train/Test split based on a temporal cutoff (not random)

## 🔍 Exploratory Data Analysis

- Visualized hourly trip demand
- Applied **seasonal decomposition** to detect trend and seasonality
- Identified periodic spikes in demand consistent with morning and evening commutes

## 🛠️ Tools & Technologies

- **Languages**: Python
- **Libraries**: 
  - `pandas`, `numpy`
  - `matplotlib`, `seaborn`
  - `xgboost`, `scikit-learn`
  - `statsmodels`
- **Notebook**: Jupyter / VS Code

## 📦 ML Models Used

| Model                         | Description                                                                 |
|------------------------------|-----------------------------------------------------------------------------|
| **XGBoost Regressor**        | Gradient boosting framework optimized for performance                      |
| **Random Forest Regressor**  | Ensemble of decision trees using bootstrap aggregation                     |
| **Gradient Boosted Trees**   | Sequential boosting on decision trees                                      |
| **Ensemble Model**           | Weighted average of the above models based on inverse MAPE                 |

## 🧠 Feature Engineering

- Lag window of 24 hours used to model previous patterns
- Normalized weights for ensemble prediction:
