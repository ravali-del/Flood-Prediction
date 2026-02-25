import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# 1. Generate Synthetic Data with higher sensitivity
np.random.seed(42)
n_samples = 12000 

data = {
    'Rainfall_mm': np.random.uniform(0, 400, n_samples),
    'Temperature_C': np.random.uniform(10, 45, n_samples),
    'Humidity_pct': np.random.uniform(20, 100, n_samples),
    'Elevation_m': np.random.uniform(0, 1200, n_samples),
    'River_Discharge_m3_s': np.random.uniform(0, 1000, n_samples)
}
df = pd.DataFrame(data)

# 2. UPDATED LOGIC: Heavy weighting on Humidity and Elevation
# Even with 0 rain, high humidity + low elevation should equal ~25-30% risk
humidity_risk = df['Humidity_pct'] * 0.2
elevation_risk = (1200 - df['Elevation_m']) * 0.05
weather_risk = (df['Rainfall_mm'] * 0.6) + (df['River_Discharge_m3_s'] * 0.4)

raw_score = weather_risk + humidity_risk + elevation_risk
df['Flood_Risk_Pct'] = (raw_score - raw_score.min()) / (raw_score.max() - raw_score.min()) * 100

X = df.drop('Flood_Risk_Pct', axis=1)
y = df['Flood_Risk_Pct']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42)
model.fit(X_train_scaled, y_train)

joblib.dump(model, "flood_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(list(X.columns), "model_features.pkl")
print("✅ Sensitive Regressor Model Saved.")