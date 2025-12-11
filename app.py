import streamlit as st
import pandas as pd
import pickle
import os
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# -------------------------------------------
# 1. SETUP & CONFIGURATION
# -------------------------------------------
st.set_page_config(page_title="Amsterdam Host Advisor", page_icon="", layout="wide")

# PROFESSIONAL STYLING SETUP
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 9,              # Smaller font
    'axes.titlesize': 12,        # Smaller titles
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.titlesize': 14
})

COLOR_FIXED = "#94A3B8"  # Slate Grey
COLOR_AI = "#4F46E5"     # Royal Blue
COLOR_ACCENT = "#10B981" # Emerald

# -------------------------------------------
# 2. LOAD THE TRAINED MODEL
# -------------------------------------------
@st.cache_resource
def load_model():
    model_path = os.path.join('models', 'xgboost_occupancy.pkl')
    if not os.path.exists(model_path):
        return None
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

if model is None:
    st.error("⚠️ System Error: Model file missing. Please check your folder structure.")
    st.stop()

# -------------------------------------------
# 3. SIDEBAR INPUTS
# -------------------------------------------
st.sidebar.header("1. Pricing Strategy")

ref_price = st.sidebar.number_input(
    "Reference Price (Standard) €", 
    min_value=10, max_value=2000, value=200, step=10
)

target_price = st.sidebar.number_input(
    "Target Price (Test) €", 
    min_value=10, max_value=2000, value=180, step=10
)

if target_price < ref_price:
    st.sidebar.caption(f"📉 **Discount:** Testing a €{ref_price - target_price} drop.")
elif target_price > ref_price:
    st.sidebar.caption(f"📈 **Hike:** Testing a €{target_price - ref_price} increase.")
else:
    st.sidebar.caption("⚖️ **Stable:** No price change.")

st.sidebar.markdown("---")
st.sidebar.header("2. Your Listing")

reviews = st.sidebar.slider("Total Reviews", 0, 500, 50)

star_options = {
    "⭐⭐⭐⭐⭐ (Excellent)": 0.95,
    "⭐⭐⭐⭐ (Good)": 0.6,
    "⭐⭐⭐ (Average)": 0.0,
    "⭐⭐ (Poor)": -0.5,
    "⭐ (Terrible)": -0.9
}
star_selection = st.sidebar.selectbox("Guest Rating", options=list(star_options.keys()), index=1)
sentiment_score = star_options[star_selection]

topic_display = {0: "Standard / General", 1: "Great Location", 2: "Great Hospitality"}
topic = st.sidebar.selectbox("Main Selling Point", options=[0, 1, 2], format_func=lambda x: topic_display[x])

st.sidebar.markdown("---")
st.sidebar.header("3. Select Date")

today = datetime.date.today()
selected_date = st.sidebar.date_input("Check-in Date", value=today + datetime.timedelta(days=7), min_value=today)

input_month = selected_date.month
input_day_of_week = selected_date.weekday()

weather_defaults = {
    1: (3, 20, False), 2: (3, 20, False), 3: (6, 18, True),
    4: (9, 15, False), 5: (13, 13, False), 6: (15, 12, False),
    7: (18, 12, False), 8: (18, 11, False), 9: (15, 13, True),
    10: (11, 15, True), 11: (7, 17, True), 12: (4, 20, True)
}
d_temp, d_wind, d_rain = weather_defaults[input_month]

with st.sidebar.expander(f"Weather Settings ({selected_date.strftime('%B')})"):
    temp = st.slider("Temp (°C)", -5, 35, value=d_temp)
    wind = st.slider("Wind (km/h)", 0, 50, value=d_wind)
    rain = st.checkbox("Rain Forecasted?", value=d_rain)

# -------------------------------------------
# 4. MAIN PAGE: LIVE ANALYSIS (No Buttons)
# -------------------------------------------
st.title(" Amsterdam Host Strategy Tool")
st.markdown(f"###  Forecast for {selected_date.strftime('%A, %d %B %Y')}")

# --- 1. RUN PREDICTION INSTANTLY ---
scenarios = {
    "Target": target_price,
    "Discount": int(target_price * 0.8), 
    "Premium": int(target_price * 1.2)   
}

results = {}

for name, price_sim in scenarios.items():
    input_data = pd.DataFrame({
        'price': [price_sim],
        'price_7d_lag': [ref_price], 
        'Temp': [temp], 'Rain': [1.0 if rain else 0.0], 'Wind': [wind],
        'month': [input_month], 'day_of_week': [input_day_of_week],
        'avg_sentiment': [sentiment_score], 'total_reviews': [reviews], 'dominant_topic': [topic]
    })
    prob = model.predict_proba(input_data)[0][1]
    results[name] = prob

p_target = results["Target"]
p_prem = results["Premium"]
p_disc = results["Discount"]

# DISPLAY SCORECARD
col1, col2, col3 = st.columns([1, 1.3, 1])

with col1:
    color = "green" if p_target >= 0.7 else "orange" if p_target >= 0.4 else "red"
    st.markdown(f"""
    <div style="text-align: center; border: 2px solid #f0f2f6; padding: 15px; border-radius: 10px;">
        <h4 style="margin:0; color:gray;">Booking Chance</h4>
        <h1 style="color: {color}; font-size: 36px; margin:0;">{p_target:.0%}</h1>
        <p style="color: gray; margin:0; font-size: 12px;">at €{target_price}</p>
    </div>
    """, unsafe_allow_html=True)
    # FIX: Cast to float to prevent StreamlitAPIException
    st.progress(float(p_target))

with col2:
    st.write("#### 💡 AI Strategy")
    if p_prem >= 0.65:
        st.success(f"🚀 **Raise Price.** Demand is strong! You retain a **{p_prem:.0%}** chance even at €{scenarios['Premium']}.")
    elif p_target < 0.50 and p_disc >= 0.60:
        st.warning(f"🏷️ **Discount Needed.** Demand is soft. Dropping to €{scenarios['Discount']} boosts chance by **+{(p_disc - p_target):.0%}**.")
    elif p_target >= 0.50:
            st.success(f"✅ **Hold Steady.** Price is competitive. Changing it adds risk.")
    else:
        st.error("📉 **Low Demand.** Even with a discount, chance is low. Focus on photos/reviews.")

with col3:
    st.write("#### 📊 Sensitivity")
    data = {
        "Price": [f"€{scenarios['Premium']}", f"€{target_price}", f"€{scenarios['Discount']}"],
        "Chance": [f"{p_prem:.0%}", f"{p_target:.0%}", f"{p_disc:.0%}"],
    }
    st.dataframe(pd.DataFrame(data, index=["Premium", "Target", "Discount"]), use_container_width=True)

# -------------------------------------------
# 5. ANNUAL SIMULATION (AUTO-RUN)
# -------------------------------------------
st.divider()
st.subheader("💰 Annual Revenue Simulator")

# Run simulation automatically (It's fast enough)
monthly_data = []
months = range(1, 13)
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

test_prices = [int(ref_price * 0.7), int(ref_price * 0.85), ref_price, int(ref_price * 1.15), int(ref_price * 1.3)]
total_fixed_revenue = 0
total_opt_revenue = 0

for m in months:
    m_temp, m_wind, m_rain = weather_defaults[m]
    
    # FIXED Strategy
    fixed_input = pd.DataFrame({
        'price': [ref_price], 'price_7d_lag': [ref_price],
        'Temp': [m_temp], 'Rain': [1.0 if m_rain else 0.0], 'Wind': [m_wind],
        'month': [m], 'day_of_week': [5],
        'avg_sentiment': [sentiment_score], 'total_reviews': [reviews], 'dominant_topic': [topic]
    })
    prob_fixed = model.predict_proba(fixed_input)[0][1]
    exp_rev_fixed = ref_price * prob_fixed
    
    # OPTIMAL Strategy
    best_price = ref_price
    best_rev = 0
    
    batch_input = pd.DataFrame({
        'price': test_prices,
        'price_7d_lag': [ref_price]*5,
        'Temp': [m_temp]*5, 'Rain': [1.0 if m_rain else 0.0]*5, 'Wind': [m_wind]*5,
        'month': [m]*5, 'day_of_week': [5]*5,
        'avg_sentiment': [sentiment_score]*5, 'total_reviews': [reviews]*5, 'dominant_topic': [topic]*5
    })
    probs = model.predict_proba(batch_input)[:, 1]
    
    for i, p_val in enumerate(test_prices):
        rev = p_val * probs[i]
        if rev > best_rev:
            best_rev = rev
            best_price = p_val
    
    monthly_data.append({
        "Month": month_names[m-1],
        "Fixed Price": ref_price,
        "Optimal Price": best_price,
        "Revenue (Fixed)": exp_rev_fixed * 30, 
        "Revenue (Smart)": best_rev * 30       
    })
    
    total_fixed_revenue += (exp_rev_fixed * 30)
    total_opt_revenue += (best_rev * 30)

df_sim = pd.DataFrame(monthly_data)

# --- VISUALIZATION 1: COMPACT LINE CHART ---
col_graph1, col_graph2 = st.columns(2)

with col_graph1:
    st.write("**Optimal Pricing Curve**")
    fig, ax = plt.subplots(figsize=(7, 3)) # Compact Size
    sns.lineplot(data=df_sim, x="Month", y="Fixed Price", label="Fixed", 
                 linestyle="--", color=COLOR_FIXED, linewidth=2, ax=ax)
    sns.lineplot(data=df_sim, x="Month", y="Optimal Price", label="Smart", 
                 marker="o", markersize=6, color=COLOR_AI, linewidth=2.5, ax=ax)
    ax.fill_between(df_sim["Month"], df_sim["Fixed Price"], df_sim["Optimal Price"], 
                    where=(df_sim["Optimal Price"] > df_sim["Fixed Price"]),
                    interpolate=True, color=COLOR_AI, alpha=0.1)
    ax.set_ylabel("Price (€)")
    ax.set_xlabel("")
    sns.despine()
    st.pyplot(fig)

with col_graph2:
    st.write("**Annual Revenue Gap**")
    delta = total_opt_revenue - total_fixed_revenue
    st.caption(f"Potential Gain: **+€{delta:,.0f}** / year")
    
    fig2, ax2 = plt.subplots(figsize=(7, 3)) # Compact Size
    revenue_data = pd.DataFrame({
        "Strategy": ["Fixed", "Smart"],
        "Revenue": [total_fixed_revenue, total_opt_revenue]
    })
    sns.barplot(data=revenue_data, y="Strategy", x="Revenue", palette=[COLOR_FIXED, COLOR_ACCENT], ax=ax2)
    for i, v in enumerate([total_fixed_revenue, total_opt_revenue]):
        ax2.text(v * 0.95, i, f"€{v:,.0f}", color='white', fontweight='bold', ha='right', va='center')
    ax2.set_xlabel("Total €")
    ax2.set_ylabel("")
    sns.despine(left=True, bottom=True)
    ax2.set_xticks([])
    st.pyplot(fig2)
