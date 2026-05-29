import ollama
import pandas as pd

# Load inventory data
df = pd.read_csv("data/inventory_clean.csv")

# Filter risky ingredients
risky_items = df[df["wastage_risk"] == 1]

# Select top risky ingredients
ingredients = risky_items["ingredient_name"].unique()[:10]

ingredient_list = ", ".join(ingredients)

print("\nGenerating AI Recommendations...\n")

# Prompt for local LLM
prompt = f"""
You are an expert restaurant chef and inventory manager.

The following ingredients are nearing expiry:

{ingredient_list}

Suggest:
1. 5 chef special dishes
2. Waste reduction strategies
3. Inventory optimization suggestions

Keep suggestions practical for a restaurant.
"""

# Generate response from Ollama
response = ollama.chat(
    model="phi3",   # change if using another model
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

# Extract response text
result = response["message"]["content"]

print(result)

# Save recommendations
with open("data/chef_recommendations.txt", "w", encoding="utf-8") as f:
    f.write(result)

print("\nRecommendations saved successfully.")