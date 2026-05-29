import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

from xgboost import XGBClassifier
import joblib


# ---------------------------------------------------
# Load cleaned dataset
# ---------------------------------------------------

df = pd.read_csv("data/inventory_clean.csv")

# Save original names before encoding
original_names = df[["ingredient_name", "category"]].copy()

print("Dataset Loaded Successfully")
print(df.head())


# ---------------------------------------------------
# Encode categorical columns
# ML models need numerical values
# ---------------------------------------------------

label_encoders = {}

categorical_cols = [
    "ingredient_name",
    "category",
    "supplier",
    "storage_type"
]

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le


# ---------------------------------------------------
# Features and Target
# ---------------------------------------------------

features = [
    "ingredient_name",
    "category",
    "supplier",
    "storage_type",
    "quantity_kg",
    "daily_consumption_kg",
    "wastage_history_pct",
    "days_until_expiry",
    "days_of_stock"
]

X = df[features]
y = df["wastage_risk"]


# ---------------------------------------------------
# Train Test Split
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ---------------------------------------------------
# Train XGBoost Model
# ---------------------------------------------------

model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)

model.fit(X_train, y_train)

print("\nModel Training Complete")


# ---------------------------------------------------
# Predictions
# ---------------------------------------------------

y_pred = model.predict(X_test)


# ---------------------------------------------------
# Evaluation
# ---------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# ---------------------------------------------------
# Feature Importance
# ---------------------------------------------------

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:\n")
print(importance_df)


# ---------------------------------------------------
# Save model
# ---------------------------------------------------

joblib.dump(model, "model.pkl")

print("\nModel saved as model.pkl")


# ---------------------------------------------------
# Save predictions CSV (for Streamlit dashboard)
# ---------------------------------------------------

# Restore original ingredient and category names
df["ingredient_name"] = original_names["ingredient_name"]
df["category"]        = original_names["category"]

df["risk_prob_high"]  = model.predict_proba(X)[:, 1]
df["risk_label_name"] = df["risk_prob_high"].apply(
    lambda x: "High" if x >= 0.6 else "Medium" if x >= 0.3 else "Low"
)
df["explanation"] = df.apply(lambda row:
    "Expires soon" if row["days_until_expiry"] <= 3
    else "High wastage history" if row["wastage_history_pct"] > 25
    else "Overstocked" if row["days_of_stock"] > row["days_until_expiry"]
    else "Within safe parameters", axis=1
)

df.to_csv("data/predictions.csv", index=False)
print("Predictions saved to data/predictions.csv")