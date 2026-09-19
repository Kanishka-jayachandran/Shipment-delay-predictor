"""
Supply Chain Shipment Delay Prediction
========================================
Trains and compares three models on the delay classification task:
  1. Logistic Regression      (baseline, interpretable)
  2. Gradient Boosting        (strong tabular ensemble - sklearn's GBM,
                                the same family of algorithm as XGBoost/LightGBM)
  3. MLP Neural Network       (deep learning baseline)

Then runs permutation-importance based explainability on the best model.
"""

import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_curve, classification_report
)
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

DATA_PATH = "/home/claude/supply_chain_delay_predictor/data/shipments.csv"
PLOTS_DIR = "/home/claude/supply_chain_delay_predictor/plots"
MODELS_DIR = "/home/claude/supply_chain_delay_predictor/models"

# ---------------------------------------------------------------------------
# 1. Load + basic EDA
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} shipments, {df['is_delayed'].mean():.1%} delayed overall\n")

# EDA plot: delay rate by mode and by carrier
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
df.groupby("mode")["is_delayed"].mean().sort_values().plot(kind="barh", ax=axes[0], color="#4C72B0")
axes[0].set_title("Delay Rate by Shipping Mode")
axes[0].set_xlabel("Delay Rate")

df.groupby("carrier")["is_delayed"].mean().sort_values().plot(kind="barh", ax=axes[1], color="#DD8452")
axes[1].set_title("Delay Rate by Carrier")
axes[1].set_xlabel("Delay Rate")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/eda_delay_rates.png", dpi=140)
plt.close()
print("Saved EDA plot -> plots/eda_delay_rates.png")

# ---------------------------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------------------------
df["congestion_gap"] = (df["origin_congestion_idx"] - df["dest_congestion_idx"]).abs()
df["total_congestion"] = df["origin_congestion_idx"] + df["dest_congestion_idx"]
df["transit_per_1000km"] = df["planned_transit_days"] / (df["distance_km"] / 1000 + 1e-6)
df["log_shipment_value"] = np.log1p(df["shipment_value_usd"])
df["route_experience"] = np.log1p(df["num_prior_shipments_route"])

numeric_features = [
    "distance_km", "planned_transit_days", "origin_congestion_idx",
    "dest_congestion_idx", "customs_complexity", "num_customs_stops",
    "weather_risk", "carrier_reliability_score", "log_shipment_value",
    "is_hazmat", "route_experience", "congestion_gap", "total_congestion",
    "transit_per_1000km",
]
categorical_features = ["carrier", "mode", "origin_port", "dest_port", "season"]

X = df[numeric_features + categorical_features]
y = df["is_delayed"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}\n")

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
])

# ---------------------------------------------------------------------------
# 3. Train three models
# ---------------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "Gradient Boosting (XGBoost-style ensemble)": GradientBoostingClassifier(
        n_estimators=250, learning_rate=0.07, max_depth=3, subsample=0.9, random_state=42
    ),
    "Neural Network (MLP)": MLPClassifier(
        hidden_layer_sizes=(64, 32), activation="relu", alpha=1e-3,
        max_iter=500, random_state=42, early_stopping=True
    ),
}

results = {}
fitted_pipelines = {}

plt.figure(figsize=(6, 5.5))
for name, clf in models.items():
    pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = pipe.predict(X_test)

    auc = roc_auc_score(y_test, proba)
    f1 = f1_score(y_test, pred)
    prec = precision_score(y_test, pred)
    rec = recall_score(y_test, pred)

    results[name] = {"roc_auc": auc, "f1": f1, "precision": prec, "recall": rec}
    fitted_pipelines[name] = pipe

    fpr, tpr, _ = roc_curve(y_test, proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

    print(f"--- {name} ---")
    print(classification_report(y_test, pred, target_names=["On-time", "Delayed"]))

plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves — Model Comparison")
plt.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/roc_comparison.png", dpi=140)
plt.close()
print("Saved ROC comparison -> plots/roc_comparison.png")

# ---------------------------------------------------------------------------
# 4. Save metrics + best model
# ---------------------------------------------------------------------------
with open(f"{MODELS_DIR}/metrics.json", "w") as f:
    json.dump(results, f, indent=2)

best_name = max(results, key=lambda k: results[k]["roc_auc"])
best_pipe = fitted_pipelines[best_name]
joblib.dump(best_pipe, f"{MODELS_DIR}/best_model.joblib")
joblib.dump({"numeric_features": numeric_features, "categorical_features": categorical_features},
            f"{MODELS_DIR}/feature_schema.joblib")
print(f"\nBest model: {best_name} (ROC-AUC={results[best_name]['roc_auc']:.3f}) -> saved to models/best_model.joblib")

# ---------------------------------------------------------------------------
# 5. Explainability: permutation importance on best model
# ---------------------------------------------------------------------------
perm = permutation_importance(
    best_pipe, X_test, y_test, n_repeats=8, random_state=42, scoring="roc_auc", n_jobs=-1
)
# Map back to original column names (permutation importance works on raw X columns
# since it's computed through the full pipeline)
imp_df = pd.DataFrame({
    "feature": X_test.columns,
    "importance_mean": perm.importances_mean,
    "importance_std": perm.importances_std,
}).sort_values("importance_mean", ascending=False)

plt.figure(figsize=(7, 6))
top = imp_df.head(12).iloc[::-1]
plt.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"], color="#55A868")
plt.title(f"Permutation Feature Importance — {best_name}\n(drop in ROC-AUC when feature is shuffled)")
plt.xlabel("Importance (AUC drop)")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/feature_importance.png", dpi=140)
plt.close()
print("Saved feature importance plot -> plots/feature_importance.png")

imp_df.to_csv(f"{MODELS_DIR}/feature_importance.csv", index=False)

print("\n=== SUMMARY ===")
for name, m in results.items():
    print(f"{name:45s} ROC-AUC={m['roc_auc']:.3f}  F1={m['f1']:.3f}  Prec={m['precision']:.3f}  Rec={m['recall']:.3f}")
print(f"\nTop 5 delay drivers: {imp_df.head(5)['feature'].tolist()}")
