# AI-Based Business Location Selector Using ML, Clustering, and Multi-Criteria Decision Analysis
Scope: Café, Bhopal. **All data is synthetic** (seed 42) – it demonstrates the workflow, not real-world accuracy.

## Run
    pip install -r requirements.txt
    python pipeline.py        # builds dataset.csv, models, charts, rankings in outputs/
    streamlit run app.py      # interactive dashboard

## Files
- pipeline.py – data generation, cleaning, feature engineering, EDA, Linear/Tree/RF models (CV + grid search), K-Means, scoring, sensitivity
- app.py – Streamlit dashboard (budget, area filter, adjustable weights, top-5, explanations, clusters, CSV export)
- dataset.csv, rf_model.joblib, outputs/* – data, model, results, charts

## Data dictionary
See guide §5. Extra: monthly_sales (synthetic target), competitor_density = competitors per 10k people, footfall_per_rent, cluster, predicted_sales (5-fold out-of-fold RF prediction), s_D..s_R (normalised 0–100 components), suitability_score.

## Score
S = 0.30D + 0.20A + 0.15F + 0.15P + 0.10C + 0.10R; D = predicted sales, C = lower competitor density, R = 1 − rent/budget (all 0–100).
