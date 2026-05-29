import streamlit as st
import pandas as pd
import plotly.express as px
import requests

st.set_page_config(
    page_title="Ecomeal AI",
    page_icon="🍽️",
    layout="wide"
)

# ── Load Data ─────────────────────────────────────────────────
@st.cache_data
def load_data():
    inventory   = pd.read_csv("data/inventory_clean.csv")
    forecast    = pd.read_csv("data/demand_forecast.csv")
    predictions = pd.read_csv("data/predictions.csv")
    return inventory, forecast, predictions

inventory, forecast, predictions = load_data()

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.title("🍽️ Ecomeal AI")
st.sidebar.markdown("Food Waste Intelligence System")
st.sidebar.markdown("---")
st.sidebar.markdown(f"📦 **Total Items:** {len(inventory)}")
st.sidebar.markdown(f"⚠️ **High Risk:** {len(predictions[predictions['risk_label_name']=='High'])}")
st.sidebar.markdown(f"🔄 **Last Refreshed:** Just now")

# ── Title ─────────────────────────────────────────────────────
st.title("🍽️ Ecomeal — AI Food Waste Intelligence Dashboard")
st.markdown("Predict waste · Forecast demand · Reduce losses · Generate Chef Specials")
st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview",
    "⚠️ Risk Items",
    "📈 Demand Forecast",
    "🍽️ Chef Specials"
])

# ════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Inventory Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Items",    len(inventory))
    col2.metric("High Risk",      len(predictions[predictions["risk_label_name"] == "High"]))
    col3.metric("Overstock",      int((forecast["overstock_risk"] == "High").sum()))
    col4.metric("Expiring Today", int((inventory["days_until_expiry"] <= 1).sum()))

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Risk Distribution by Category")
        risk_cat = predictions.groupby(
            ["category","risk_label_name"]
        ).size().reset_index(name="count")
        fig1 = px.bar(
            risk_cat, x="category", y="count", color="risk_label_name",
            color_discrete_map={"High":"red","Medium":"orange","Low":"green"},
            labels={"count":"Items","category":"Category","risk_label_name":"Risk"}
        )
        st.plotly_chart(fig1, width='stretch')

    with col2:
        st.subheader("Waste Risk Breakdown")
        risk_counts = predictions["risk_label_name"].value_counts().reset_index()
        risk_counts.columns = ["Risk Level","Count"]
        fig2 = px.pie(
            risk_counts, names="Risk Level", values="Count",
            color="Risk Level",
            color_discrete_map={"High":"red","Medium":"orange","Low":"green"}
        )
        st.plotly_chart(fig2, width='stretch')

    st.subheader("Top 10 Highest Risk Items")
    top10 = predictions.sort_values("risk_prob_high", ascending=False).head(10)
    st.dataframe(
        top10[["ingredient_name","category","days_until_expiry",
               "risk_label_name","risk_prob_high","explanation"]],
        width='stretch'
    )

# ════════════════════════════════════════════════════════════
# TAB 2 — RISK ITEMS
# ════════════════════════════════════════════════════════════
with tab2:
    st.subheader("⚠️ At-Risk Inventory")

    col1, col2 = st.columns(2)
    cat_filter  = col1.selectbox("Filter by Category",
                                  ["All"] + sorted(predictions["category"].unique().tolist()))
    risk_filter = col2.selectbox("Filter by Risk Level",
                                  ["All","High","Medium","Low"])

    filtered = predictions.copy()
    if cat_filter  != "All":
        filtered = filtered[filtered["category"] == cat_filter]
    if risk_filter != "All":
        filtered = filtered[filtered["risk_label_name"] == risk_filter]

    st.markdown(f"Showing **{len(filtered)}** items")
    st.dataframe(
        filtered[["ingredient_name","category","days_until_expiry",
                  "risk_label_name","risk_prob_high","explanation"]]
        .sort_values("risk_prob_high", ascending=False),
        width='stretch'
    )

    st.subheader("Risk by Category")
    fig3 = px.histogram(
        filtered, x="category", color="risk_label_name",
        color_discrete_map={"High":"red","Medium":"orange","Low":"green"},
        labels={"category":"Category","risk_label_name":"Risk Level"}
    )
    st.plotly_chart(fig3, width='stretch')

# ════════════════════════════════════════════════════════════
# TAB 3 — DEMAND FORECAST
# ════════════════════════════════════════════════════════════
with tab3:
    st.subheader("📈 7-Day Demand Forecast")

    col1, col2, col3 = st.columns(3)
    col1.metric("Overstock Items", int((forecast["overstock_risk"] == "High").sum()))
    col2.metric("Shortage Items",  int((forecast["shortage_risk"] == "High").sum()))
    col3.metric("Reorder Needed",  int((forecast["overstock_risk"] == "High").sum() +
                                       (forecast["shortage_risk"] == "High").sum()))

    st.markdown("---")

    ingredient = st.selectbox(
        "Select Ingredient to view forecast",
        sorted(forecast["ingredient_name"].unique().tolist())
    )

    row = forecast[forecast["ingredient_name"] == ingredient].iloc[0]

    forecast_data = pd.DataFrame({
        "Day": [f"Day {i}" for i in range(1, 8)],
        "Forecasted Consumption": [row["forecast_next_7_days"] / 7] * 7
    })

    fig4 = px.line(
        forecast_data, x="Day", y="Forecasted Consumption",
        markers=True,
        title=f"7-Day Forecast: {ingredient}"
    )
    st.plotly_chart(fig4, width='stretch')

    st.subheader("Full Forecast Table")
    st.dataframe(
        forecast[["ingredient_name","avg_daily_consumption",
                  "forecast_next_7_days","demand_trend",
                  "overstock_risk","shortage_risk"]]
        .sort_values("forecast_next_7_days", ascending=False),
        width='stretch'
    )

# ════════════════════════════════════════════════════════════
# TAB 4 — CHEF SPECIALS
# ════════════════════════════════════════════════════════════
with tab4:
    st.subheader("🍽️ AI Chef Specials Generator")
    st.markdown("Enter expiring ingredients and get AI-powered dish suggestions to minimise waste.")

    # Auto-suggest expiring ingredients
    expiring = predictions[predictions["days_until_expiry"] <= 3]["ingredient_name"].unique()
if len(expiring) > 0:
    st.info(f"🚨 Expiring soon: **{', '.join([str(x) for x in expiring[:5]])}**")
    
    ingredients = st.text_area(
        "Enter expiring ingredients (comma separated)",
        placeholder="e.g. spinach, paneer, tomatoes, mushrooms",
        height=80
    )

    if st.button("Generate Chef Specials 🤖", type="primary"):
        if ingredients.strip():
            with st.spinner("Asking AI for suggestions..."):
                try:
                    response = requests.post(
                      "http://localhost:11434/api/generate",
    json={
        "model": "tinyllama", 
        "prompt": f"""You are a professional chef assistant for a restaurant.
         These ingredients are expiring soon and must be used today: {ingredients}

          Suggest 3 creative dishes using these ingredients to minimize food waste.
          For each dish provide:
          1. Dish name
          2. Ingredients used from the list
          3. Brief preparation method (2-3 lines)
          4. Why this reduces waste

         Be practical and realistic for a restaurant kitchen.""",
          "stream": False
    },
    timeout=120
)
                    result = response.json()["response"]
                    st.markdown("### 👨‍🍳 Chef Suggestions")
                    st.markdown(result)

                except requests.exceptions.ConnectionError:
                    st.error("❌ Ollama is not running. Start it with: `ollama serve`")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        else:
            st.warning("Please enter some ingredients first.")