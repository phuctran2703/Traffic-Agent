import os
from dotenv import load_dotenv
import requests
from pydantic import BaseModel
from langchain.tools import Tool, StructuredTool
from datetime import datetime
from zoneinfo import ZoneInfo

# Load API keys from .env
load_dotenv()
google_api_key = os.getenv("GOOGLE_MAPS_API_KEY")
tomtom_api_key = os.getenv("TOMTOM_API_KEY")
openweathermap_api_key = os.getenv("OPENWEATHERMAP_API_KEY")

# =========================
# Geocode Tool
# =========================

class GeocodeInput(BaseModel):
    query: str

def geocode_with_google(address, city="Ho Chi Minh City", country="Vietnam"):
    """
    Geocode an address using Google Maps API.
    """
    query = f"{address}, {city}, {country}"
    url = f"https://maps.googleapis.com/maps/api/geocode/json?address={query}&key={google_api_key}"
    response = requests.get(url)
    data = response.json()

    try:
        result = data['results'][0]
        lat = result['geometry']['location']['lat']
        lon = result['geometry']['location']['lng']
        formatted_address = result['formatted_address']
        return (
            f"Formatted Address: {formatted_address}\n"
            f"Latitude: {lat}\n"
            f"Longitude: {lon}"
        )
    except (KeyError, IndexError):
        return f"Could not find location for: {address}"

geocode_address_tool = Tool(
    name="geocode_address",
    description="Get coordinates from a given address or place name. Input is a address or place name.",
    func=geocode_with_google,
    args_schema=GeocodeInput
)

# =========================
# Traffic Tool
# =========================

class TrafficInput(BaseModel):
    latitude: float
    longitude: float

def get_traffic_info(latitude: float, longitude: float):
    """
    Get traffic information using latitude and longitude.
    """
    url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/relative0/10/json?point={latitude},{longitude}&key={tomtom_api_key}"
    response = requests.get(url)

    if response.status_code != 200:
        return f"Error: Received status code {response.status_code} from TomTom API."

    try:
        data = response.json()
        flow = data['flowSegmentData']
        return (
            f"Current Speed: {flow['currentSpeed']} km/h\n"
            f"Free Flow Speed: {flow['freeFlowSpeed']} km/h\n"
            f"Travel Time: {flow['currentTravelTime']} seconds\n"
            f"Confidence: {flow['confidence']}"
        )
    except (KeyError, IndexError, ValueError):
        return f"Could not retrieve traffic data for coordinates: ({latitude}, {longitude}). Please check the input or API response."

get_traffic_status_tool = StructuredTool.from_function(
    name="get_traffic_status",
    description="Get traffic data using latitude and longitude via TomTom API.",
    func=get_traffic_info,
    args_schema=TrafficInput
)

# =========================
# Weather Tool
# =========================

class WeatherInput(BaseModel):
    latitude: float
    longitude: float

def get_weather_info(latitude: float, longitude: float) -> str:
    """
    Get current weather information using latitude and longitude.
    """
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={latitude}&lon={longitude}&appid={openweathermap_api_key}&units=metric"
    response = requests.get(url)
    
    if response.status_code != 200:
        return f"Error: Received status code {response.status_code} from OpenWeatherMap API."

    try:
        data = response.json()
        weather = data['weather'][0]
        main = data['main']
        temp = main['temp']
        humidity = main['humidity']
        description = weather['description']
        city = data.get("name", "Unknown Location")

        return (
            f"Weather at {city} (lat: {latitude}, lon: {longitude}):\n"
            f"Description: {description}\n"
            f"Temperature: {temp}°C\n"
            f"Humidity: {humidity}%"
        )
    except (KeyError, IndexError):
        return f"Could not retrieve weather data for coordinates: ({latitude}, {longitude})"

get_weather_tool = StructuredTool.from_function(
    name="get_weather",
    description="Get current weather information using latitude and longitude.",
    func=get_weather_info,
    args_schema=WeatherInput
)

# =========================
# Time Tool
# =========================

def get_current_time() -> str:
    """
    Get the current time in Vietnam timezone in the format 'YYYY-MM-DD HH:MM:SS'
    """
    vn_timezone = ZoneInfo("Asia/Ho_Chi_Minh")
    current_time = datetime.now(vn_timezone).strftime("%Y-%m-%d %H:%M:%S")
    return f"Current time in Vietnam: {current_time}"

get_current_time_tool = StructuredTool.from_function(
    name="get_current_time",
    description="Get the current time in Vietnam timezone in the format 'YYYY-MM-DD HH:MM:SS'.",
    func=get_current_time
)