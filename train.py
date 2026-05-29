import pandas as pd
import numpy as np
import pickle, json, warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from xgboost import XGBClassifier

# ── Load ──────────────────────────────────────────────────────
df = pd.read_csv("data/inventory_clean.csv", parse_dates=["purchase_date","expiry_date"])
print(f"Loaded: {df.shape[0]} rows")

# ── Feature Engineering ───────────────────────────────────────
df["stock_expiry_gap"]      = df["days_of_stock"] - df["days_until_expiry"]
df["stock_to_expiry_ratio"] = df["days_of_stock"] / df["days_until_expiry"].replace(0, 0.5)
df["will_waste"]            = (df["days_of_stock"] > df["days_until_expiry"]).astype(int)
df["wastage_norm"]          = df["wastage_history_pct"] / 100.0
df["is_perishable"]         = df["category"].isin(["Meat","Dairy","Vegetable","Fruit"]).astype(int)

# ── Encode Categoricals ───────────────────────────────────────
le = LabelEncoder()
df["category_enc"] = le.fit_transform(df["category"])

# ── Features & Target ─────────────────────────────────────────
FEATURES = [
    "quantity_kg", "daily_consumption_kg", "days_until_expiry",
    "days_of_stock", "wastage_history_pct", "category_enc",
    "stock_expiry_gap", "stock_to_expiry_ratio", "will_waste",
    "wastage_norm", "is_perishable"
]

X = df[FEATURES]
y = df["wastage_risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# ── Train ─────────────────────────────────────────────────────
ratio = y.value_counts()[0] / y.value_counts()[1]

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=ratio,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42
)
model.fit(X_train, y_train)
print("Training complete.")

# ── Evaluate ──────────────────────────────────────────────────
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:,1]

print("Accuracy :", round(accuracy_score(y_test, y_pred), 4))
print("F1       :", round(f1_score(y_test, y_pred), 4))
print("ROC-AUC  :", round(roc_auc_score(y_test, y_proba), 4))
print(classification_report(y_test, y_pred, target_names=["Safe","At-Risk"]))

# ── Feature Importance ────────────────────────────────────────
feat_imp = pd.DataFrame({
    "feature":    FEATURES,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

print("\nTop Features:")
print(feat_imp.to_string(index=False))
feat_imp.to_csv("feature_importance.csv", index=False)

# ── Save Model ────────────────────────────────────────────────
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nSaved: model.pkl, feature_importance.csv")