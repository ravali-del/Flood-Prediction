import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# 1. Generate Calibrated Data (30,000 samples)
np.random.seed(42)
n_samples = 30000 
data = {
    'Rainfall_mm': np.random.uniform(0, 500, n_samples),
    'Temperature_C': np.random.uniform(10, 45, n_samples),
    'Humidity_pct': np.random.uniform(10, 100, n_samples),
    'Elevation_m': np.random.uniform(0, 1500, n_samples),
    'River_Discharge_m3_s': np.random.uniform(0, 1200, n_samples)
}
df = pd.DataFrame(data)

def calculate_flood_risk(row):
    rain, elev, hum = row['Rainfall_mm'], row['Elevation_m'], row['Humidity_pct'] / 100
    # Rain impact grows exponentially
    rain_score = (rain ** 1.6) * 1.5
    # High ground (like Hyderabad) protects better unless rain is extreme
    elev_protection = max(0, elev - 300) * 0.05
    raw = (rain_score * (1 + hum)) - elev_protection + (row['River_Discharge_m3_s'] * 0.1)
    return raw

df['Raw_Score'] = df.apply(calculate_flood_risk, axis=1)
min_s, max_s = df['Raw_Score'].min(), df['Raw_Score'].max()
df['Flood_Risk_Pct'] = ((df['Raw_Score'] - min_s) / (max_s - min_s)) * 100
df['Flood_Risk_Pct'] = df['Flood_Risk_Pct'].clip(6.5, 98.0)

# 2. Add Derived Features
feature_cols = ['Rainfall_mm', 'Temperature_C', 'Humidity_pct', 'Elevation_m', 
                'River_Discharge_m3_s', 'Rain_Saturation', 'Slope_Risk']
df['Rain_Saturation'] = df['Rainfall_mm'] * (df['Humidity_pct'] / 100)
df['Slope_Risk'] = df['River_Discharge_m3_s'] / (df['Elevation_m'] + 1)

# 3. Train and Save
X, y = df[feature_cols], df['Flood_Risk_Pct']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42)
model.fit(X_scaled, y)

joblib.dump(model, "flood_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("✅ Intelligence Saved: flood_model.pkl & scaler.pkl")