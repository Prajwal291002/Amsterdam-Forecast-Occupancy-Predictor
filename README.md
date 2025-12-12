

# 🏙️ The Amsterdam Forecast: A Multi-Modal Occupancy Predictor

A machine learning framework that helps Airbnb hosts in Amsterdam maximize revenue by predicting booking probability using price, weather, and guest sentiment. 

---

## 📖 About the Project

In the competitive short-term rental market, many hosts set prices by guesswork and overlook signals such as upcoming weather or patterns in guest reviews.     
The Amsterdam Forecast fuses Economics (calendar data), Environment (weather API), and Voice of the Customer (NLP on reviews) into a single predictive engine using an XGBoost classifier and an interactive Streamlit dashboard.   

Link to the Project : https://amsterdam-forecast.streamlit.app/ 

---

## ⭐ Key Features

- **Multi-Modal Learning**  
  Combines structured time-series data (calendar, price, weather) with unstructured textual data (guest reviews).

- **Advanced NLP**  
  Uses VADER for sentiment scoring and LDA (Latent Dirichlet Allocation) to extract latent topics from guest reviews.

- **Explainable AI**  
  Uses SHAP (SHapley Additive exPlanations) to interpret model decisions and highlight drivers of low or high occupancy predictions.

- **Strategic Deployment**  
  Provides a “What-If” simulator that calculates expected revenue and recommends optimal pricing strategies (Discount vs Premium).

---

## 💡 Usage Example: Meet “Sarah”

To see how the app works, consider **Sarah**, a host with an apartment near the canals who has an empty Saturday coming up with rain in the forecast.

1. **The Input**

   Sarah opens the app and enters:
   - Reference Price: €200 (standard rate).  
   - Target Price: €180 (test rate).  
   - Weather: Auto-detected “Rainy & 12°C”.     
   - Review Score: 4 Stars (good, but not perfect).

2. **The AI Analysis (What the App Displays)**

   The app runs multiple simulations and shows:

   - **Score Card**  
     - Booking Chance: 35% (red).  
     - Estimated Revenue: €63 (computed as €180 × 0.35).  
     - Meaning: At €180, the night is likely to remain unbooked.

3. **💡 AI Strategy Advice**

   Example guidance:

   > “⚠️ Drop price to around €144. Your current chance is low. A 20% discount could boost booking probability to about 75%, raising expected revenue from €63 to €108.”

4. **The Outcome**

   Sarah lowers the price to €145, secures a booking shortly after, and avoids earning €0 for that night.

---

## 📂 Project Structure
```text
Amsterdam_Forecast/
│
├── data/
│ ├── raw/ # Original CSVs (Calendar, Reviews, Listings, Weather)
│ └── processed/ # Cleaned & merged datasets (01_structured, 02_text)
│
├── models/
│ └── xgboost_occupancy.pkl # Trained Model (Serialized)
│
├── notebooks/
│ ├── 01_Structured_Pipeline.ipynb # Cleaning, Imputation & Weather Merge
│ ├── 02_Text_Pipeline.ipynb # NLP: VADER Sentiment & LDA Topic Modeling
│ └── 03_Integration_and_Modeling.ipynb # Fusion, Training, SHAP & Evaluation
│
├── output/ # Generated plots (Confusion Matrix, SHAP, etc.)
├── app.py # Streamlit Deployment Application
├── config.py # Global path variables
├── requirements.txt # Dependency list
└── README.md # Project Documentation
```

---

## ⚙️ Installation & Setup

This project uses a dedicated environment to avoid version conflicts between `xgboost`, `shap`, and `numpy`.   

### Prerequisites

- Anaconda or Miniconda installed.  
- Git (optional).

### Step 1: Clone the Repository

git clone https://github.com/Prajwal291002/Amsterdam-Forecast-Occupancy-Predictor.git
cd Amsterdam_Forecast


### Step 2: Create a Clean Environment

Use a fresh Conda environment (especially recommended on Windows to reduce DLL conflicts):

1. Create environment with Python 3.10
conda create -n amsterdam_env python=3.10 -y

2. Activate the environment
conda activate amsterdam_env

3. Install Core Libraries via Conda (Safe & Stable)
conda install -c conda-forge numpy=1.26.4 pandas scikit-learn matplotlib seaborn jupyter notebook -y

4. Install ML & NLP Libraries via Pip
pip install -r requirements.txt

---

## 🚀 How to Run

### 1. Reproduce the Analysis (Run Notebooks)

To explore the data processing and model training pipeline, launch Jupyter:

jupyter notebook

Then run, in order:

1. `01_Structured_Pipeline.ipynb` – Prepares calendar and weather data.      
2. `02_Text_Pipeline.ipynb` – Extracts sentiment and topics from reviews.  
3. `03_Integration_and_Modeling.ipynb` – Merges datasets, trains XGBoost, and creates evaluation plots.

### 2. Launch the App (Deployment)

Run the Host Strategy Tool with Streamlit:

streamlit run app.py


This starts a local server, typically available at:

http://localhost:8501

where the interactive dashboard can be accessed.
---

## 🧠 Methodology

### Phase 1: Structured Data Pipeline

- **Imputation:** Missing prices in the calendar for booked dates are filled using `base_price` from the listings dataset.  
- **Enrichment:** Historical weather data (rain, wind, temperature) is joined from a weather API such as Meteostat.      
- **Feature Engineering:** Features like `price_7d_lag` capture price volatility and discounting behavior over time.

### Phase 2: Text Pipeline (NLP)

- **Filtering:** Reviews with fewer than 40 words are removed to reduce noise.  
- **Sentiment:** VADER assigns each review a sentiment score between -1 (very negative) and +1 (very positive).  
- **Topic Modeling:** LDA identifies three latent themes:
  - Topic 0: General/Standard.  
  - Topic 1: Location (canals, city center).  
  - Topic 2: Hospitality (host, service).

### Phase 3: Integration & Modeling

- **Fusion Strategy:** Aggregates review-level features (mean sentiment, mode topic) by `listing_id` and broadcasts them across the temporal calendar data.  
- **Algorithm:** Trains an XGBoost classifier tailored for tabular data with non-linear feature interactions.  
- **Validation:** Achieves roughly 70.5% accuracy, outperforming a Random Forest baseline at 65.7%.

---

## 📊 Results & Insights

| Aspect             | Insight                                                                 |
|--------------------|-------------------------------------------------------------------------|
| Price Elasticity   | Price is the strongest predictor; higher prices lower booking odds in a non-linear way. |
| Reputation Matters | Listings with high sentiment (> 0.8) and “Location” topics show higher occupancy.       |
| Seasonality        | Month and day-of-week capture Amsterdam’s tourism peaks and troughs.                   |

---

## 👥 Authors

- **Prajwal Bhandarkar** – Structured Pipeline & Modeling  
- **Tushar Puntambekar** – NLP Pipeline & Deployment  

---

## 📜 License

This project is for academic purposes. Data is provided by the InsideAirbnb initiative.  
