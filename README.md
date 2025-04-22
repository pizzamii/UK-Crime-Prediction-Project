# UK-Crime-Prediction-Project

This repository contains the full implementation of the **UK Crime Prediction and Visualization System**, a final-year BUPT-QMUL joint programme project by **Gao Yuxuan** (2021123456). The goal of this project is to forecast regional crime volumes using machine learning models and to provide an interactive visualization tool to support public safety strategy and urban resource planning.

## Project Objectives

- Predict future total crimes in 0.1°×0.1° geographic grids across the UK
- Leverage historical crime data to generate spatial features
- Train interpretable machine learning models (XGBoost and Random Forest)
- Quantify predictive uncertainty using bootstrapped confidence intervals
- Build a fully interactive web app for visualization and user interaction

---


---

## 🛠️ Technical Stack

- **Languages**: Python 3.10
- **Libraries**: pandas, numpy, scikit-learn, xgboost, skopt, matplotlib, seaborn, folium, streamlit
- **Visualization**: Interactive dashboard with Streamlit + Folium maps + Plotly charts
- **Modeling**: XGBoost (main), with comparisons to Random Forest, Ridge, Lasso, etc.
- **Data**: Open UK Police Crime Data (2019–2023)

---

## Core Features

- **Geospatial Feature Engineering**: Encodes grid location, urban proximity, and past crime densities.
- **Uncertainty-Aware Predictions**: Confidence intervals derived from bootstrapped predictions.
- **Regional and Grid-Level Analysis**: Results aggregated by region and micro grid.
- **Interactive Dashboard**: 
  - Explore historical crime patterns
  - View future crime hotspots via heatmap
  - Understand feature importance via bar chart
  - Adjust crime types and forecast granularity on the fly

> You can explore screenshots and figures used in the final report via the `SupportDocs` ZIP file.

---

## How to Run

1. Clone this repo:

```bash
git clone https://github.com/pizzamii/UK-Crime-Prediction-Project.git
cd UK-Crime-Prediction-Project
streamlit run visualization_app.py

