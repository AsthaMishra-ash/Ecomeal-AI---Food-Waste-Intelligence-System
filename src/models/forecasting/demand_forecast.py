import pandas as pd
import numpy as np

# Load cleaned inventory data
df = pd.read_csv("data/inventory_clean.csv")

print("Dataset Loaded\n")

# -----------------------------
# Demand Forecast Logic
# -----------------------------

# Group by ingredient
forecast_df = df.groupby("ingredient_name").agg({
    "daily_consumption_kg": ["mean", "max", "min"],
    "quantity_kg": "mean"
})

# Flatten column names
forecast_df.columns = [
    "avg_daily_consumption",
    "max_daily_consumption",
    "min_daily_consumption",
    "avg_stock"
]

forecast_df = forecast_df.reset_index()

# -----------------------------
# Forecast Future Demand
# -----------------------------
# Explanation:
# We simulate future demand using
# rolling growth assumptions.

forecast_df["forecast_next_7_days"] = (
    forecast_df["avg_daily_consumption"] * 7 * 1.10
).round(2)

# -----------------------------
# Demand Trend Analysis
# -----------------------------

forecast_df["demand_trend"] = np.where(
    forecast_df["max_daily_consumption"] >
    forecast_df["avg_daily_consumption"] * 1.2,

    "Increasing",

    np.where(
        forecast_df["min_daily_consumption"] <
        forecast_df["avg_daily_consumption"] * 0.8,

        "Decreasing",

        "Stable"
    )
)

# -----------------------------
# Overstock Detection
# -----------------------------

forecast_df["overstock_risk"] = np.where(
    forecast_df["avg_stock"] >
    forecast_df["forecast_next_7_days"] * 1.5,

    "High",

    "Low"
)

# -----------------------------
# Shortage Risk Detection
# -----------------------------

forecast_df["shortage_risk"] = np.where(
    forecast_df["forecast_next_7_days"] >
    forecast_df["avg_stock"] * 1.2,

    "High",

    "Low"
)
# -----------------------------
# Procurement Suggestions
# -----------------------------

def procurement_action(row):

    # Overstock takes higher priority
    if row["overstock_risk"] == "High":
        return "Reduce Procurement"

    # Strong shortage condition
    elif (
        row["shortage_risk"] == "High"
        and row["demand_trend"] == "Increasing"
    ):
        return "Restock Soon"

    else:
        return "Stock Level OK"

# -----------------------------
# Save Forecast
# -----------------------------

forecast_df.to_csv(
    "data/demand_forecast.csv",
    index=False
)

print("Demand Forecast Generated\n")

print(forecast_df.head())