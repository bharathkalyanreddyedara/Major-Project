from datetime import datetime
from typing import List, Dict, Any
from backend.app.services.weather_service import weather_service

class NotificationService:
    def __init__(self):
        self.notifications_store = [
            {
                "id": 1,
                "title": "Welcome to AI Farm Assistant",
                "category": "System",
                "severity": "info",
                "message": "Enter your manual soil test values or upload a soil photo to get personalized crop recommendations.",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "read": False
            }
        ]

    def get_notifications(self, crop_name: str = None, growth_stage: str = None, city: str = "Hyderabad", **kwargs) -> List[Dict[str, Any]]:
        # Check live weather for automated warnings
        weather = weather_service.get_weather(city=city)
        results = list(self.notifications_store)

        rainfall = weather.get("rainfall", 0.0) or 0.0
        temp = weather.get("temperature", 25.0) or 25.0
        humidity = weather.get("humidity", 50.0) or 50.0
        wind = weather.get("wind_speed", 10.0) or 10.0

        if rainfall > 15.0:
            results.insert(0, {
                "id": len(results) + 101,
                "title": "Severe Rainstorm & Flood Alert",
                "category": "Weather Hazard",
                "severity": "critical",
                "message": f"Precipitation estimated at {rainfall} mm with wind speeds of {wind} km/h in {city}.",
                "action_required": "Halt all chemical and foliar fertilizer sprays. Clear peripheral field drainage trenches immediately to prevent root waterlogging.",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "read": False
            })
        elif rainfall > 5.0:
            results.insert(0, {
                "id": len(results) + 101,
                "title": "Moderate Rain Advisory",
                "category": "Weather Alert",
                "severity": "warning",
                "message": f"Rainfall expected ({rainfall} mm). Soil moisture levels will remain elevated.",
                "action_required": "Delay planned irrigation rounds by 48-72 hours to save power and water.",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "read": False
            })

        if temp > 36.0:
            results.insert(0, {
                "id": len(results) + 102,
                "title": "Extreme Heatwave & Thermal Stress Warning",
                "category": "Thermal Hazard",
                "severity": "warning",
                "message": f"Ambient temperature reaching {temp}°C in {city}, creating high evapotranspiration demand.",
                "action_required": "Provide light evening sprinkler irrigation and apply straw/organic mulch to reduce soil surface evaporation.",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "read": False
            })
        elif temp < 10.0:
            results.insert(0, {
                "id": len(results) + 103,
                "title": "Cold Snap & Frost Risk Advisory",
                "category": "Thermal Hazard",
                "severity": "warning",
                "message": f"Night temperature dropping to {temp}°C in {city}.",
                "action_required": "Schedule early-evening light flood irrigation to maintain soil temperature above freezing point.",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "read": False
            })

        if humidity > 80.0 and temp > 22.0:
            results.insert(0, {
                "id": len(results) + 104,
                "title": "High Fungal & Blast Infection Risk",
                "category": "Pathogen Alert",
                "severity": "warning",
                "message": f"High relative humidity ({humidity}%) and warm weather create optimal conditions for fungal spores.",
                "action_required": f"Inspect {crop_name or 'crop'} canopy for foliar leaf spots, sheath blight, or powdery mildew. Keep bio-fungicides ready.",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "read": False
            })

        if crop_name and growth_stage:
            results.append({
                "id": len(results) + 105,
                "title": f"{crop_name} Management Alert: {growth_stage}",
                "category": "Agronomic Stage",
                "severity": "info",
                "message": f"Your {crop_name} is in the {growth_stage} developmental window.",
                "action_required": "Ensure timely top-dressing of nitrogen and monitor root zone soil moisture.",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "read": False
            })

        return results

    def add_notification(self, title: str, message: str, category: str = "Timeline", severity: str = "info", action_required: str = "Check dashboard"):
        self.notifications_store.insert(0, {
            "id": len(self.notifications_store) + 1,
            "title": title,
            "category": category,
            "severity": severity,
            "message": message,
            "action_required": action_required,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "read": False
        })

notification_service = NotificationService()
