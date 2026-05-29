import pandas as pd
import numpy as np

def clean_inventory(path="data/inventory_raw.csv"):
    df = pd.read_csv(path, parse_dates=["purchase_date", "expiry_date"])

    # 1. Drop rows with no ingredient name (unusable)
    df = df[df["ingredient_name"].str.strip() != ""]
    df = df[df["ingredient_name"].notna()]

    # 2. Fix invalid negatives — treat as missing then impute
    df.loc[df["daily_consumption_kg"] < 0, "daily_consumption_kg"] = np.nan

    # 3. Impute missing numerics with category median
    # Interview Q: "Why median not mean?" → median is robust to outliers
    for col in ["quantity_kg", "daily_consumption_kg"]:
        df[col] = df.groupby("category")[col].transform(
            lambda x: x.fillna(x.median())
        )

    # 4. Fix missing expiry_date — fill with category average shelf life
    avg_expiry = df.groupby("category")["days_until_expiry"].transform("median")
    df["expiry_date"] = df["expiry_date"].fillna(
        pd.Timestamp.today() + pd.to_timedelta(avg_expiry, unit="D")
    )

    # 5. Recompute derived columns (they may have been affected by fixes)
    today = pd.Timestamp.today().normalize()
    df["days_until_expiry"] = (df["expiry_date"] - today).dt.days.clip(lower=0)
    df["days_of_stock"] = (df["quantity_kg"] / df["daily_consumption_kg"]).round(2)

    # 6. Remove duplicates
    df = df.drop_duplicates()

    df.to_csv("data/inventory_clean.csv", index=False)
    print(f"Clean dataset: {len(df)} rows")
    print(f"Nulls remaining:\n{df.isnull().sum()}")
    return df

if __name__ == "__main__":
    clean_inventory()