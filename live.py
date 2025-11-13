import requests
import folium
import webbrowser
import time

def get_location():
    """Get current location based on IP address"""
    try:
        response = requests.get("https://ipinfo.io/json").json()
        loc = response["loc"].split(",")
        latitude = float(loc[0])
        longitude = float(loc[1])
        return latitude, longitude
    except Exception as e:
        print("Error fetching location:", e)
        return None, None

def show_location_on_map(lat, lon):
    """Generate and open map with current location"""
    map_ = folium.Map(location=[lat, lon], zoom_start=13)
    folium.Marker([lat, lon], popup="You are here!").add_to(map_)
    map_.save("live_location.html")
    webbrowser.open("live_location.html")

# Update every 30 seconds
while True:
    lat, lon = get_location()
    if lat and lon:
        print(f"📍 Current location: {lat}, {lon}")
        show_location_on_map(lat, lon)
    else:
        print("Could not get location.")
    time.sleep(30)
