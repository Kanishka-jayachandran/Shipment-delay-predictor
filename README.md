# 🚢 Supply Chain Shipment Delay Predictor

Predicts the probability that a shipment will arrive late, using route, carrier,
customs, weather, and port-congestion features. Built as an end-to-end ML project:
data generation → EDA → feature engineering → model comparison → explainability →
interactive demo app.

## Problem

Late shipments cost logistics companies money in penalties, customer trust, and
downstream planning disruption. This project frames delay prediction as a binary
classification problem: **given a shipment's route, carrier, and conditions, what's
the probability it arrives late?**

## Data

A synthetic but statistically realistic dataset of 15,000 shipments (`data/generate_data.py`)
with known ground-truth relationships between features and delay risk — carrier
reliability, port congestion by season, customs complexity, and weather all
genuinely drive the delay probability, mirroring patterns reported in real
logistics data. Overall delay rate: **32.1%**.

Features include: carrier, shipping mode (Sea/Air/Rail/Road), origin/destination
port, season, distance, planned transit time, port congestion indices, customs
complexity and stop count, weather risk, carrier reliability score, shipment
value, hazmat flag, and prior-shipment count on the route.

## Approach

1. **EDA** — delay rate breakdowns by mode and carrier (`plots/eda_delay_rates.png`)
2. **Feature engineering** — congestion gap/total, transit-time-per-1000km,
   log-scaled shipment value, route experience (log of prior shipment count)
3. **Modeling** — three models trained and compared on held-out test data:
   - Logistic Regression (interpretable baseline)
   - Gradient Boosting (`GradientBoostingClassifier`, the same ensemble family as
     XGBoost/LightGBM)
   - Neural Network (`MLPClassifier`, 2 hidden layers)
4. **Explainability** — permutation importance on the best model to show which
   features most affect predictions (`plots/feature_importance.png`)
5. **Demo** — interactive Streamlit app for live delay-risk scoring (`app.py`)

## Results

| Model | ROC-AUC | F1 | Precision | Recall |
|---|---|---|---|---|
| Logistic Regression | 0.697 | 0.534 | 0.465 | 0.628 |
| Gradient Boosting | 0.694 | 0.370 | 0.584 | 0.271 |
| Neural Network (MLP) | 0.687 | 0.269 | 0.601 | 0.174 |

Logistic Regression edges out the more complex models on ROC-AUC and F1 here —
a useful, resume-worthy talking point: with class imbalance (32% positive) and
moderately noisy real-world-style features, a well-regularized linear model with
`class_weight="balanced"` can match or beat more complex models on recall-sensitive
metrics, while staying far more interpretable for operational teams.

**Top delay drivers** (via permutation importance): number of customs stops,
customs complexity, origin port congestion, weather risk, and total congestion —
consistent with real-world supply chain delay research.

## Project Structure

```
supply_chain_delay_predictor/
├── data/
│   ├── generate_data.py       # synthetic dataset generator
│   └── shipments.csv          # generated dataset
├── models/
│   ├── best_model.joblib      # trained sklearn pipeline (preprocessing + model)
│   ├── feature_schema.joblib  # feature list used at inference time
│   ├── metrics.json           # evaluation metrics for all 3 models
│   └── feature_importance.csv
├── plots/
│   ├── eda_delay_rates.png
│   ├── roc_comparison.png
│   └── feature_importance.png
├── train_models.py            # full training + evaluation pipeline
├── app.py                     # Streamlit interactive demo
├── requirements.txt
└── README.md
```

## Running It

```bash
pip install -r requirements.txt
python data/generate_data.py      # regenerate the dataset (optional, already included)
python train_models.py            # retrain all 3 models, regenerate plots
streamlit run app.py              # launch the interactive demo
```

## Resume Line

> Built an end-to-end shipment delay prediction system comparing Logistic
> Regression, Gradient Boosting, and Neural Network models on route, carrier,
> and customs features; identified key delay drivers via permutation importance
> and deployed an interactive Streamlit risk-scoring demo.

## Possible Extensions

- Swap in real logistics data (e.g., DataCo Smart Supply Chain dataset on Kaggle)
- Add XGBoost/LightGBM directly and SHAP for per-prediction explanations
- Model delay *duration* (regression) in addition to delay probability (classification)
- Add live port congestion data via an API for real-time scoring
