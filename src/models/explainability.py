import pandas as pd
import shap
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# ---------------------------------------------------
# Load Data
# ---------------------------------------------------

df = pd.read_csv("data/inventory_clean.csv")

categorical_cols = [
    "ingredient_name",
    "category",
    "supplier",
    "storage_type"
]

# Encode categorical variables
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))


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
# Train-test split
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ---------------------------------------------------
# Load trained model
# ---------------------------------------------------

model = joblib.load("model.pkl")


# ---------------------------------------------------
# SHAP Explainer
# ---------------------------------------------------

explainer = shap.Explainer(model)

shap_values = explainer(X_test)


# ---------------------------------------------------
# Explain one prediction
# ---------------------------------------------------

sample_index = 0

print("\nPrediction Explanation:\n")

for feature, value in zip(
    features,
    shap_values[sample_index].values
):
    print(f"{feature}: {round(value, 4)}")


# ---------------------------------------------------
# Global Feature Importance
# ---------------------------------------------------

print("\nGenerating SHAP summary plot...")

shap.summary_plot(shap_values, X_test)