import requests
import math
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Configuration - Replace with your actual OWM Key
OWM_API_KEY = "545c69b2976f421ac39cf670425bab85"

class CityInput(BaseModel):
    city: str
    simulation: bool = False

def get_location_name(lat, lon):
    url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json&accept-language=en"
    headers = {"User-Agent": "FloodGuard_V5"}
    try:
        res = requests.get(url, headers=headers).json()
        addr = res.get("address", {})
        return (addr.get("suburb") or addr.get("neighbourhood") or 
                addr.get("town") or addr.get("village") or "Safe High Ground")
    except: 
        return "Safe Zone"

def get_elevation(lat, lon):
    try:
        url = f"https://api.open-meteo.com/v1/elevation?latitude={lat}&longitude={lon}"
        res = requests.get(url, timeout=3).json()
        return res['elevation'][0]
    except:
        return 0

def get_geo_data(city):
    clean_city = city.strip()
    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={clean_city}&count=1&language=en&format=json"
    try:
        res = requests.get(geo_url, timeout=5).json()
        if "results" in res and len(res["results"]) > 0:
            data = res["results"][0]
            return data["latitude"], data["longitude"], data.get("elevation", 0)
        return None, None, 0
    except: 
        return None, None, 0

def get_hybrid_weather(lat, lon):
    """Combines OpenWeatherMap and Open-Meteo for high-accuracy flood monitoring."""
    rain_owm, rain_om = 0.0, 0.0
    temp, hum = 25.0, 50.0

    # 1. OpenWeatherMap Call (Primary for Temp/Humidity)
    try:
        owm_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OWM_API_KEY}&units=metric"
        owm_data = requests.get(owm_url, timeout=3).json()
        temp = owm_data['main']['temp']
        hum = owm_data['main']['humidity']
        rain_owm = owm_data.get('rain', {}).get('1h', 0.0)
    except:
        pass # Fallback to defaults or Open-Meteo

    # 2. Open-Meteo Call (Primary for High-Resolution Precipitation)
    try:
        om_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,precipitation&forecast_days=1"
        om_data = requests.get(om_url, timeout=3).json()
        rain_om = om_data['current']['precipitation']
        # If OWM failed, use OM for temp/hum
        if temp == 25.0: temp = om_data['current']['temperature_2m']
        if hum == 50.0: hum = om_data['current']['relative_humidity_2m']
    except:
        pass

    # Safety logic: Use the highest recorded rain value between both services
    final_rain = max(rain_owm, rain_om)
    return final_rain, temp, hum

def get_safety_instructions(risk_score):
    if risk_score >= 70:
        return {
            "sedan": "🚨 CRITICAL: Low-lying areas detected. High risk of hydroplaning.",
            "suv": "⚠️ WARNING: Proceed only via elevated detours. Avoid underpasses.",
            "bus": "🚨 ALERT: Route submerged. Exhaust clearance compromised.",
            "pedestrian": "🚨 DANGER: Fast moving water. Move to second floor/high ground.",
            "train": "🚨 HALTED: Tracks likely submerged.",
            "metro": "⚠️ DELAYS: Underground sections under monitoring."
        }
    return {k: "✅ SAFE: No significant risk detected." for k in ["sedan", "suv", "bus", "pedestrian", "train", "metro"]}

@app.post("/predict-flood")
def predict(data: CityInput):
    search_query = data.city.strip()
    lat, lon, current_elev = get_geo_data(search_query)
    
    if lat is None: 
        return {"error": f"City '{search_query}' not found."}
    
    # Check Simulation vs Real Hybrid Weather
    rain, temp, hum = (105.5, 22.0, 94.0) if data.simulation else get_hybrid_weather(lat, lon)
    
    # Risk calculation logic
    rain_impact = math.log1p(rain) * 22 
    elev_impact = (600 - current_elev) * 0.05 
    sat_impact = (hum / 100) * 10 if rain > 0 else (hum / 100) * 2
    risk = round(max(2, min(98, 5 + rain_impact + elev_impact + sat_impact)), 1)
    
    # --- INTELLIGENT REROUTING LOGIC ---
    test_points = [
        (lat + 0.03, lon), (lat - 0.03, lon), 
        (lat, lon + 0.03), (lat, lon - 0.03)
    ]
    
    best_lat, best_lon, max_elev = lat, lon, current_elev
    
    for t_lat, t_lon in test_points:
        t_elev = get_elevation(t_lat, t_lon)
        if t_elev > max_elev:
            max_elev = t_elev
            best_lat, best_lon = t_lat, t_lon

    safe_name = get_location_name(best_lat, best_lon)

    return {
        "city": search_query.capitalize(), 
        "risk": risk, 
        "rain_mm": round(rain, 1),
        "lat": lat, "lon": lon, 
        "temp": temp, "hum": hum, "elev": current_elev,
        "is_high_flood": risk >= 70,
        "safe_route_name": safe_name,
        "safe_lat": best_lat, "safe_lon": best_lon,
        "safe_elev": max_elev,
        "elev_gain": round(max_elev - current_elev, 1),
        "safety_alerts": get_safety_instructions(risk),
        "data_sources": "Dual-Sync (OpenWeatherMap + OpenMeteo)"
    }