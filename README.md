# Universal Bank - Streamlit Dashboard

This repository contains a Streamlit app to explore the Universal Bank dataset, train tree-based models (Decision Tree, Random Forest, Gradient Boosted Tree), and predict whether a customer will accept a personal loan.

## Files in this package
- `app.py` : Main Streamlit app (entrypoint).
- `utils.py`: Helper functions for preprocessing, training, plotting.
- `UniversalBank.csv` : (optional) Example dataset included if present.
- `requirements.txt`: Packages to install on Streamlit Cloud (no versions).
- `.gitignore` : Basic ignores.

## How to deploy on Streamlit Cloud
1. Create a new GitHub repository and push these files (no folders required).
2. On https://streamlit.io/cloud, create a new app and connect your repo and branch.
3. Set the main file to `app.py` and deploy.

## Features
- Dashboard with 5 actionable charts for marketing insights:
  1. Income bins & acceptance rate.
  2. Income vs CCAvg scatter (targeting).
  3. Education x Family stacked acceptance (segmentation).
  4. PCA 2D customer clusters.
  5. Cumulative Gain (Lift) chart.

- Model Trainer tab: Train Decision Tree, Random Forest, Gradient Boosted Tree with 5-fold CV. Shows performance table, ROC curves, confusion matrices, and feature importances.

- Predict & Upload tab: Upload a new CSV and predict `Predicted_PersonalLoan` using a Random Forest trained on the provided dataset; download predictions as CSV.
