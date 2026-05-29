# 🍽️ Ecomeal AI — Food Waste Intelligence System

An AI-powered food waste prediction and recommendation system for restaurants.
Built to reduce inventory waste, predict expiry risks, forecast demand,
and generate smart chef specials using expiring ingredients.


## 📁 Project Structure

Ecomeal AI/
├── data/
│   ├── inventory_raw.csv
│   ├── inventory_clean.csv
│   ├── demand_forecast.csv
│   └── predictions.csv
├── src/models/
│   ├── train_wastage_model.py
│   ├── explainability.py
│   ├── forecasting/demand_forecast.py
│   └── recommendation/chef_specials.py
├── app.py
├── data_cleaner.py
├── generate_dataset.py
├── model.pkl
├── feature_importance.csv
├── requirements.txt
└── README.md


## 🚀 How to Run

### 1. Install dependencies
pip install -r requirements.txt

### 2. Generate dataset
python generate_dataset.py

### 3. Clean data
python data_cleaner.py

### 4. Train model
python src/models/train_wastage_model.py

### 5. Generate demand forecast
python src/models/forecasting/demand_forecast.py

### 6. Start Ollama
ollama serve
ollama pull tinyllama

### 7. Run dashboard
streamlit run app.py

## 📊 Dataset

- 1200 simulated restaurant inventory records
- Features: ingredient name, category, quantity, expiry date,
  daily consumption, wastage history, supplier, storage type
- Intentionally injected missing values, duplicates, negative
  values and malformed dates to simulate real-world data

## ⚙️ Feature Engineering

| Feature | Description |
|---------|-------------|
| stock_expiry_gap | Days of stock minus days until expiry |
| stock_to_expiry_ratio | How much stock relative to shelf life |
| will_waste | Binary flag if stock outlasts expiry |
| wastage_norm | Normalised wastage history (0–1) |
| is_perishable | High risk categories like Meat, Dairy |

## 🤖 Model

- Algorithm: XGBoost Classifier
- Handles class imbalance via scale_pos_weight
- Train/Test split: 80/20 with stratification

### Results
| Metric | Score |
|--------|-------|
| Accuracy | 0.9037 |
| F1 Score | 0.9486 |
| ROC-AUC | 0.9639 |

### Top Features
1. days_until_expiry (54%)
2. stock_expiry_gap (12.6%)
3. wastage_norm (9.6%)



## 🧠 AI Integration

- Used Ollama (local LLM) with tinyllama model
- Generates Chef Special dish suggestions from expiring ingredients
- Runs fully offline — no API key needed
- Input: comma separated expiring ingredients
- Output: 3 dish suggestions with preparation method


## 🔮 Future Improvements

- Connect to real POS system for live consumption data
- Add SHAP visualisation in dashboard
- Retrain model automatically when new data arrives
- Deploy on cloud (AWS/GCP) with Docker
- Add multi-restaurant support

## 🛠️ Tech Stack

- Python, Pandas, NumPy
- XGBoost, Scikit-learn
- Streamlit, Plotly
- Ollama (tinyllama)




