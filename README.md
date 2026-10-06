# Breast Cancer Diagnosis Predictor

BiotechTrek AI in Healthcare & Drug Discovery Bootcamp: Capstone (Option A).

**Student:** Mayuri Tembhurne

A Streamlit app that predicts whether a breast-mass sample is **benign** or **malignant** from 30 cell-nucleus measurements, using a Logistic Regression model trained on scikit-learn's Wisconsin Breast Cancer dataset (test accuracy ~98.2%).

> Educational project only. Not a medical device.

## Files
| File | Purpose |
|---|---|
| `Breast_Cancer_Capstone.ipynb` | Full workflow: explore, clean, visualize, train, evaluate, save |
| `app.py` | Streamlit web app |
| `model.joblib`, `scaler.joblib` | Trained model and fitted scaler |
| `feature_stats.csv`, `examples.csv`, `metrics.json` | Slider ranges, example cases, displayed metrics |
| `requirements.txt` | Dependencies for Streamlit Community Cloud |

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Live app
(paste your Streamlit link here)
