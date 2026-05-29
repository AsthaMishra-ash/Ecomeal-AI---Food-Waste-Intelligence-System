import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta

fake = Faker()
np.random.seed(42)       # makes results reproducible — interviewer Q: "why seed 42?" → reproducibility
random.seed(42)

# ── Configuration ────────────────────────────────────────────────────────────
N = 1200   # slightly over 1000 to be safe

CATEGORIES   = ["Vegetable", "Fruit", "Dairy", "Meat", "Grain", "Spice", "Beverage"]
STORAGE      = ["Refrigerated", "Frozen", "Dry Storage", "Room Temperature"]
SUPPLIERS    = ["FreshFarm Co", "Metro Wholesale", "GreenLeaf Suppliers",
                "Daily Fresh", "AgriDirect", "SpiceRoute India"]

# Realistic ingredients per category
INGREDIENTS = {
    "Vegetable": ["Tomato","Spinach","Potato","Onion","Capsicum","Carrot","Mushroom",
                  "Broccoli","Cabbage","Zucchini"],
    "Fruit":     ["Lemon","Mango","Apple","Banana","Papaya","Watermelon","Orange"],
    "Dairy":     ["Paneer","Curd","Butter","Cheese","Milk","Cream"],
    "Meat":      ["Chicken","Mutton","Fish","Prawns","Eggs"],
    "Grain":     ["Rice","Wheat Flour","Semolina","Oats","Chickpeas","Lentils"],
    "Spice":     ["Turmeric","Cumin","Coriander Powder","Red Chilli","Garam Masala"],
    "Beverage":  ["Orange Juice","Mango Pulp","Coconut Water","Soda"],
}

# ── Helper: realistic expiry windows per category ────────────────────────────
# Interview Q: "Why different expiry windows?" → different shelf lives in real kitchens
EXPIRY_DAYS = {
    "Vegetable": (2, 10),
    "Fruit":     (3, 14),
    "Dairy":     (2, 7),
    "Meat":      (1, 5),
    "Grain":     (30, 180),
    "Spice":     (90, 365),
    "Beverage":  (7, 30),
}

def random_date_within(days_min, days_max):
    """Return a future date between days_min and days_max from today."""
    delta = random.randint(days_min, days_max)
    return datetime.today() + timedelta(days=delta)

# ── Generate rows ────────────────────────────────────────────────────────────
rows = []
today = datetime.today()

for _ in range(N):
    category   = random.choice(CATEGORIES)
    ingredient = random.choice(INGREDIENTS[category])
    storage    = random.choice(STORAGE)
    supplier   = random.choice(SUPPLIERS)

    expiry_min, expiry_max = EXPIRY_DAYS[category]
    expiry_date   = random_date_within(expiry_min, expiry_max)
    purchase_date = today - timedelta(days=random.randint(0, expiry_min))

    days_until_expiry = (expiry_date - today).days   # key engineered feature

    quantity_kg          = round(random.uniform(0.5, 50.0), 2)
    daily_consumption_kg = round(random.uniform(0.1, quantity_kg * 0.4), 2)
    wastage_history_pct  = round(random.uniform(0, 40), 2)   # past % wasted

    # ── Target variable logic (rule-based label — honest to explain in interview) ──
    # Interview Q: "How did you label the target?" →
    # "I used domain rules: if expiry < 3 days OR wastage history > 25% OR
    #  quantity far exceeds consumption → likely waste. Then added noise."
    risk_score = 0
    if days_until_expiry <= 3:
        risk_score += 3
    elif days_until_expiry <= 7:
        risk_score += 1

    if wastage_history_pct > 25:
        risk_score += 2

    days_of_stock = quantity_kg / daily_consumption_kg if daily_consumption_kg > 0 else 999
    if days_of_stock > days_until_expiry * 1.5:   # overstocked relative to expiry
        risk_score += 2

    if category in ["Meat", "Dairy"]:
        risk_score += 1   # naturally higher-risk categories

    wastage_risk = 1 if risk_score >= 3 else 0
    # Add a small % of noise to avoid perfectly rule-derived data
    if random.random() < 0.05:
        wastage_risk = 1 - wastage_risk

    rows.append({
        "ingredient_name":      ingredient,
        "category":             category,
        "supplier":             supplier,
        "storage_type":         storage,
        "quantity_kg":          quantity_kg,
        "daily_consumption_kg": daily_consumption_kg,
        "wastage_history_pct":  wastage_history_pct,
        "purchase_date":        purchase_date.date(),
        "expiry_date":          expiry_date.date(),
        "days_until_expiry":    days_until_expiry,
        "days_of_stock":        round(days_of_stock, 2),
        "wastage_risk":         wastage_risk,        # 0 = safe, 1 = at risk
    })

df = pd.DataFrame(rows)

# ── Intentionally inject messy data (shows real-world awareness) ─────────────
# Interview Q: "Why did you add noise?" → "Real datasets are never clean.
# I simulated common issues to test my pipeline's robustness."
mess_idx = df.sample(frac=0.03).index   # ~3% rows get corrupted
df.loc[mess_idx[:10], "quantity_kg"]          = np.nan
df.loc[mess_idx[10:20], "daily_consumption_kg"] = -99   # invalid negative
df.loc[mess_idx[20:30], "expiry_date"]         = None
df.loc[mess_idx[30:35], "ingredient_name"]     = ""     # blank name

# ── Save ──────────────────────────────────────────────────────────────────────
df.to_csv("data/inventory_raw.csv", index=False)
print(f"Dataset saved: {len(df)} rows, {df.columns.tolist()}")
print(f"\nClass balance:\n{df['wastage_risk'].value_counts()}")
print(f"\nNull counts:\n{df.isnull().sum()}")