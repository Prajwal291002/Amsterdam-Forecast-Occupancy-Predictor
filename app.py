import streamlit as st
import pandas as pd
import pickle
import os
import datetime

# -------------------------------------------
# 1. SETUP & CONFIGURATION
# -------------------------------------------
st.set_page_config(page_title="Amsterdam Host Advisor", page_icon="🌷", layout="wide")

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
# 3. HEADER
# -------------------------------------------
st.title("Amsterdam Host Strategy Tool")
st.markdown("""
**Will you get a booking?** This tool compares your **New Price** against your **Reference Price** to predict if a discount or hike will win the guest.
""")
st.divider()

# -------------------------------------------
# 4. SIDEBAR INPUTS
# -------------------------------------------
st.sidebar.header("1. Pricing Strategy")

# INPUT 1: The Anchor
ref_price = st.sidebar.number_input(
    "Reference Price (Standard) €", 
    min_value=10, max_value=2000, value=200, step=10,
    help="Your standard price or last week's price."
)

# INPUT 2: The Test Price
target_price = st.sidebar.number_input(
    "Target Price (Test) €", 
    min_value=10, max_value=2000, value=180, step=10,
    help="The price you want to test for this date."
)

# Visual Cue
if target_price < ref_price:
    st.sidebar.caption(f"📉 **Discount:** Testing a €{ref_price - target_price} drop.")
elif target_price > ref_price:
    st.sidebar.caption(f"📈 **Hike:** Testing a €{target_price - ref_price} increase.")
else:
    st.sidebar.caption("⚖️ **Stable:** No price change.")

st.sidebar.markdown("---")
st.sidebar.header("2. Your Listing")

reviews = st.sidebar.slider("Total Reviews", 0, 500, 50)

# --- UPDATED: STAR RATING DROPDOWN ---
star_options = {
    "⭐⭐⭐⭐⭐ (Excellent)": 0.95,
    "⭐⭐⭐⭐ (Good)": 0.6,
    "⭐⭐⭐ (Average)": 0.0,
    "⭐⭐ (Poor)": -0.5,
    "⭐ (Terrible)": -0.9
}

star_selection = st.sidebar.selectbox(
    "Guest Rating",
    options=list(star_options.keys()),
    index=1 # Default to 4 stars
)
sentiment_score = star_options[star_selection]

# Topic Selection
topic_display = {0: "Standard / General", 1: "Great Location", 2: "Great Hospitality"}
topic = st.sidebar.selectbox("Main Selling Point", options=[0, 1, 2], format_func=lambda x: topic_display[x])

st.sidebar.markdown("---")
st.sidebar.header("3. Select Date")

# Date Picker
today = datetime.date.today()
selected_date = st.sidebar.date_input(
    "Check-in Date", 
    value=today + datetime.timedelta(days=7),
    min_value=today
)

# Auto-Extract Month/Day
input_month = selected_date.month
input_day_of_week = selected_date.weekday()

# Holiday Visuals
if input_month == 12:
    st.sidebar.success("🎄 Holiday Season!")
elif input_month in [7, 8]:
    st.sidebar.success("☀️ Peak Summer!")

# Weather Defaults
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
# 5. ANALYSIS ENGINE
# -------------------------------------------

if st.button("🚀 Analyze Strategy", type="primary"):
    
    # --- A. PRICE SIMULATION ---
    # Compare against REFERENCE Price
    scenarios = {
        "Target": target_price,
        "Discount": int(target_price * 0.8), # 20% Cheaper
        "Premium": int(target_price * 1.2)   # 20% More Expensive
    }
    
    results = {}
    
    for name, price_sim in scenarios.items():
        input_data = pd.DataFrame({
            'price': [price_sim],
            'price_7d_lag': [ref_price], # Always anchor to Reference
            'Temp': [temp], 'Rain': [1.0 if rain else 0.0], 'Wind': [wind],
            'month': [input_month], 'day_of_week': [input_day_of_week],
            'avg_sentiment': [sentiment_score], 'total_reviews': [reviews], 'dominant_topic': [topic]
        })
        
        prob = model.predict_proba(input_data)[0][1]
        results[name] = prob

    # --- B. TOPIC SIMULATION ---
    topic_results = {}
    for t_code in [0, 1, 2]:
        if t_code == topic: continue 
        input_data_topic = pd.DataFrame({
            'price': [target_price],
            'price_7d_lag': [ref_price],
            'Temp': [temp], 'Rain': [1.0 if rain else 0.0], 'Wind': [wind],
            'month': [input_month], 'day_of_week': [input_day_of_week],
            'avg_sentiment': [sentiment_score], 'total_reviews': [reviews], 
            'dominant_topic': [t_code]
        })
        topic_results[t_code] = model.predict_proba(input_data_topic)[0][1]

    # -------------------------------------------
    # 6. RESULTS DASHBOARD
    # -------------------------------------------
    
    st.markdown(f"### 📅 Forecast for {selected_date.strftime('%A, %d %B %Y')}")
    
    # Extract Probabilities
    p_target = results["Target"]
    p_prem = results["Premium"]
    p_disc = results["Discount"]

    col1, col2, col3 = st.columns([1, 1.3, 1])
    
    # --- COLUMN 1: SCORECARD ---
    with col1:
        color = "green" if p_target >= 0.7 else "orange" if p_target >= 0.4 else "red"
        st.markdown(f"""
        <div style="text-align: center; border: 2px solid #f0f2f6; padding: 20px; border-radius: 10px;">
            <h2 style="margin:0; font-size:16px;">Booking Chance</h2>
            <h1 style="color: {color}; font-size: 45px; margin:0;">{p_target:.0%}</h1>
            <p style="color: gray; margin:0;">at €{target_price}</p>
        </div>
        """, unsafe_allow_html=True)

    # --- COLUMN 2: STRATEGY ADVICE (No Revenue, Just Probability) ---
    with col2:
        st.write("#### 💡 AI Strategy")
        
        # 1. CONTENT CHECK (Topic)
        best_alt_topic = max(topic_results, key=topic_results.get) if topic_results else None
        if best_alt_topic is not None and topic_results[best_alt_topic] > p_target + 0.05:
            t_name = {0: "Standard", 1: "Location", 2: "Hospitality"}[best_alt_topic]
            diff = topic_results[best_alt_topic] - p_target
            st.info(f"✨ **Edit Description:** Switch focus to **'{t_name}'** to boost chance by **+{diff:.0%}**.")

        # 2. PRICE CHECK (Probability Based)
        
        # Case A: HIGH DEMAND (Safe to Raise?)
        if p_prem >= 0.65:
            st.success(f"🚀 **Raise Price to €{scenarios['Premium']}.**\n\nDemand is very strong! You still have a high **{p_prem:.0%}** chance of booking even at the higher price.")
            
        # Case B: PRICE SENSITIVE (Discount helps?)
        # Current is 'meh' (<50%), but discount makes it 'good' (>60%)
        elif p_target < 0.50 and p_disc >= 0.60:
            diff = p_disc - p_target
            st.warning(f"🏷️ **Drop Price to €{scenarios['Discount']}.**\n\nDemand is sensitive here. A discount boosts your booking probability by **+{diff:.0%}**. Secure the booking!")
            
        # Case C: HOLD (Discount doesn't help enough, or Target is fine)
        elif p_target >= 0.50:
             st.success(f"✅ **Hold Steady.**\n\nYour price is competitive. You have a solid **{p_target:.0%}** chance. Changing price might add unnecessary risk.")
             
        # Case D: COLD MARKET (Nothing helps)
        else:
            st.error("📉 **Market is Cold.**\n\nEven with a discount, the booking chance is low. Focus on getting better reviews or better photos.")

    # --- COLUMN 3: COMPARISON TABLE ---
    with col3:
        st.write("#### 📊 Price vs. Chance")
        
        data = {
            "Strategy": ["Premium (+20%)", "Your Target", "Discount (-20%)"],
            "Price": [f"€{scenarios['Premium']}", f"€{target_price}", f"€{scenarios['Discount']}"],
            "Booking Chance": [f"{p_prem:.0%}", f"{p_target:.0%}", f"{p_disc:.0%}"],
        }
        df_sens = pd.DataFrame(data)
        st.dataframe(df_sens, hide_index=True, use_container_width=True)