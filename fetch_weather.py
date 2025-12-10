# fetch_weather.py
import pandas as pd
from datetime import datetime
from meteostat import Daily
import os
import config  # Importing your config.py

def fetch_amsterdam_weather():
    print("🌤️  Fetching Amsterdam Weather Data...")

    # 1. Define Location: Schiphol Airport Station (ID: 06240)
    # This is the standard WMO ID for Amsterdam
    station_id = '06240' 
    
    # 2. Define Timeframe
    # We fetch a wide range to ensure coverage for all calendar dates
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2025, 12, 31)

    # 3. Fetch Data via API
    data = Daily(station_id, start_date, end_date)
    data = data.fetch()

    # 4. Clean and Save
    # Reset index to make 'time' a column
    df = data.reset_index()
    
    # Rename columns to match our project specs
    # tavg = Temp, prcp = Rain, wspd = Wind
    df = df[['time', 'tavg', 'prcp', 'wspd']]
    df.columns = ['Date', 'Temp', 'Rain', 'Wind']
    
    # Save to raw data folder
    output_path = os.path.join(config.DATA_RAW, 'weather_amsterdam.csv')
    df.to_csv(output_path, index=False)
    
    print(f"✅ Weather data saved to: {output_path}")
    print(f"   Rows fetched: {len(df)}")

if __name__ == "__main__":
    fetch_amsterdam_weather()