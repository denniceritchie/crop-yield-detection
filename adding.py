import pandas as pd
import random
import numpy as np

# Load dataset
df = pd.read_csv(r"C:\Users\Balaji\Documents\dumps\crop yield recommend and prediction new concept\SmartCrop-Dataset.csv")

# Expanded list of Indian agricultural locations with base Latitude & Longitude
locations = [
    {"Location": "Punjab", "Latitude": 30.7333, "Longitude": 76.7794},
    {"Location": "Haryana", "Latitude": 29.0588, "Longitude": 76.0856},
    {"Location": "Uttar Pradesh", "Latitude": 26.8467, "Longitude": 80.9462},
    {"Location": "Madhya Pradesh", "Latitude": 23.2599, "Longitude": 77.4126},
    {"Location": "Rajasthan", "Latitude": 26.9124, "Longitude": 75.7873},
    {"Location": "Bihar", "Latitude": 25.5941, "Longitude": 85.1376},
    {"Location": "West Bengal", "Latitude": 22.5726, "Longitude": 88.3639},
    {"Location": "Maharashtra", "Latitude": 19.7515, "Longitude": 75.7139},
    {"Location": "Tamil Nadu", "Latitude": 11.1271, "Longitude": 78.6569},
    {"Location": "Karnataka", "Latitude": 15.3173, "Longitude": 75.7139},
    {"Location": "Kerala", "Latitude": 10.8505, "Longitude": 76.2711},
    {"Location": "Odisha", "Latitude": 20.9517, "Longitude": 85.0985},
    {"Location": "Gujarat", "Latitude": 22.2587, "Longitude": 71.1924},
    {"Location": "Assam", "Latitude": 26.2006, "Longitude": 92.9376},
    {"Location": "Andhra Pradesh", "Latitude": 15.9129, "Longitude": 79.7400}
]

# Ensure we have 2200 unique locations by repeating and modifying lat/lon slightly
expanded_locations = []
for i in range(len(df)):
    loc = random.choice(locations)
    new_lat = loc["Latitude"] + np.random.uniform(-0.05, 0.05)  # Small variation
    new_lon = loc["Longitude"] + np.random.uniform(-0.05, 0.05)  # Small variation
    expanded_locations.append({"Location": loc["Location"], "Latitude": round(new_lat, 5), "Longitude": round(new_lon, 5)})

# Assign new location data to dataset
df["Location"] = [loc["Location"] for loc in expanded_locations]
df["Latitude"] = [loc["Latitude"] for loc in expanded_locations]
df["Longitude"] = [loc["Longitude"] for loc in expanded_locations]

# Save updated dataset
df.to_csv("updated_dataset.csv", index=False)

print("Dataset updated with diverse Indian locations, latitude, and longitude!")
