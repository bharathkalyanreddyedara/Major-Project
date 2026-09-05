import requests
from backend.app.config import settings
from typing import Dict, Any

class WeatherService:
    @staticmethod
    def get_weather(city: str = "Hyderabad", lat: float = None, lon: float = None) -> Dict[str, Any]:
        api_key = settings.OPENWEATHER_API_KEY
        
        if api_key:
            try:
                if lat is not None and lon is not None:
                    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
                else:
                    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
                
                resp = requests.get(url, timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    return {
                        "temperature": data["main"]["temp"],
                        "humidity": data["main"]["humidity"],
                        "rainfall": data.get("rain", {}).get("1h", 0.0) * 24, # estimate 24h
                        "pressure": data["main"]["pressure"],
                        "weather_condition": data["weather"][0]["description"],
                        "wind_speed": data["wind"]["speed"],
                        "city": data.get("name", city),
                        "is_live": True
                    }
            except Exception as e:
                print(f"OpenWeather API fetch error: {e}")

        # Intelligent Climatological Fallback for Telangana / South Asia
        return {
            "temperature": 28.5,
            "humidity": 62.0,
            "rainfall": 110.0,
            "pressure": 1012,
            "weather_condition": "Partly Cloudy with Moderate Humidity",
            "wind_speed": 3.2,
            "city": city or "Regional Farm Location",
            "is_live": False
        }

weather_service = WeatherService()
