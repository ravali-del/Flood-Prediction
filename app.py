import requests
import math
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MIRRORS = ["https://overpass-api.de/api/interpreter", "https://lz4.overpass-api.de/api/interpreter"]

class GPSInput(BaseModel):
    lat: float
    lon: float
    simulation: bool = False
    sim_rain: Optional[float] = None

def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371
    dlat, dlon = math.radians(lat2-lat1), math.radians(lon2-lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return round(R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a)), 2)

def calculate_polygon_area(coords):
    """
    Calculates the area of a polygon using the Shoelace Formula.
    coords: List of (lat, lon) tuples
    Returns: Area in square meters
    """
    if len(coords) < 3: return 0
    
    # Local approximation for meters per degree
    lat_avg = math.radians(sum(p[0] for p in coords) / len(coords))
    x = [p[1] * 111000 * math.cos(lat_avg) for p in coords]
    y = [p[0] * 111000 for p in coords]
    
    # Shoelace Formula (Surveyor's Formula)
    area = 0.5 * abs(sum(x[i] * y[i+1] - x[i+1] * y[i] for i in range(len(coords)-1)))
    return area

def estimate_capacity(element):
    """Fallback capacity based on amenity type if geometry is missing"""
    tags = element.get('tags', {})
    amenity = tags.get('amenity', 'building')
    base_caps = {
        "stadium": 5000, "university": 2500, "hospital": 1200, 
        "school": 800, "community_centre": 400
    }
    return base_caps.get(amenity, 300)

def get_safe_zones(lat, lon):
    # 'out geom' provides the full coordinate set for area calculation
    query = f"""[out:json][timeout:25];(
      node["amenity"~"university|college|school|community_centre|stadium|hospital"](around:5000,{lat},{lon});
      way["amenity"~"university|college|school|community_centre|stadium|hospital"](around:5000,{lat},{lon});
    );out geom;"""
    
    for url in MIRRORS:
        try:
            r = requests.post(url, data={'data': query}, timeout=15).json()
            zones = []
            for el in r.get('elements', []):
                tags = el.get('tags', {})
                if tags.get('amenity') in ['clinic', 'place_of_worship']: continue
                
                area = 0
                if el.get('type') == 'way' and 'geometry' in el:
                    # Calculate center point for mapping
                    z_lat = sum(p['lat'] for p in el['geometry']) / len(el['geometry'])
                    z_lon = sum(p['lon'] for p in el['geometry']) / len(el['geometry'])
                    
                    # Calculate actual area using Shoelace Formula
                    coords = [(p['lat'], p['lon']) for p in el['geometry']]
                    area = calculate_polygon_area(coords)
                    
                    # Capacity: 1 person per 3.5sqm (International Standard)
                   # Change 3.5 to a higher number like 6.0 to reduce the PAX count
                    capacity = int(area / 6.0) if area > 0 else estimate_capacity(el)
                else:
                    z_lat, z_lon = el.get('lat'), el.get('lon')
                    capacity = estimate_capacity(el)

                amenity_type = tags.get('amenity', 'CENTER')
                zones.append({
                    "name": (tags.get('name') or f"GOVT {amenity_type.upper()}").upper(),
                    "lat": z_lat, 
                    "lon": z_lon,
                    "distance": calculate_distance(lat, lon, z_lat, z_lon),
                    "capacity": max(capacity, 100) # Minimum safety floor of 100 PAX
                })
            
            if zones: return sorted(zones, key=lambda x: x['distance'])[:6]
        except: continue
    return []

@app.post("/predict-flood-gps")
def predict_gps(data: GPSInput):
    try:
        w = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={data.lat}&longitude={data.lon}&current=temperature_2m,relative_humidity_2m,precipitation", timeout=5).json()["current"]
        e = requests.get(f"https://api.open-meteo.com/v1/elevation?latitude={data.lat}&longitude={data.lon}", timeout=5).json()["elevation"][0]
        
        rain = data.sim_rain if (data.simulation and data.sim_rain is not None) else w["precipitation"]
        
        if rain <= 0.2: 
            risk = 0.0
        else: 
            risk = round(min(99.9, (rain * 6.5) + (w["relative_humidity_2m"] / 15)), 1)
        
        safe_havens = get_safe_zones(data.lat, data.lon) if risk > 35 else []
        
        return {
            "lat": data.lat, "lon": data.lon, "risk": risk,
            "status": "CRITICAL" if risk > 65 else "WARNING" if risk > 35 else "STABLE",
            "telemetry": {"temp": w["temperature_2m"], "rain": round(rain, 1), "hum": w["relative_humidity_2m"], "elev": e},
            "safe_zones": safe_havens
        }
    except Exception as ex: return {"error": str(ex)}

@app.post("/predict-flood")
def predict_area(data: dict):
    try:
        res = requests.get(f"https://nominatim.openstreetmap.org/search?q={data['query']}&format=json&limit=1&countrycodes=in", headers={"User-Agent":"PralayKavach"}).json()
        if not res: return {"error": "Location Not Found"}
        return predict_gps(GPSInput(lat=float(res[0]["lat"]), lon=float(res[0]["lon"]), simulation=data.get('simulation', False), sim_rain=data.get('sim_rain')))
    except: return {"error": "Sat-Nav Busy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
