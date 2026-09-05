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

    def get_notifications(self, crop_name: str = None, city: str = "Hyderabad") -> List[Dict[str, Any]]:
        # Check live weather for automated warnings
        weather = weather_service.get_weather(city=city)
        results = list(self.notifications_store)

        if weather.get("rainfall", 0) > 10.0:
            results.insert(0, {
                "id": len(results) + 101,
                "title": "Weather Alert: Heavy Rain Expected",
                "category": "Weather",
                "severity": "warning",
                "message": f"Precipitation estimated at {weather['rainfall']} mm. Delay foliar fertilizer spraying and ensure drainage channels are open.",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "read": False
            })
        elif weather.get("temperature", 25) > 36.0:
            results.insert(0, {
                "id": len(results) + 102,
                "title": "Heatwave Warning",
                "category": "Weather",
                "severity": "warning",
                "message": f"High temperatures ({weather['temperature']} C). Provide light evening irrigation to avoid wilting.",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "read": False
            })

        return results

    def add_notification(self, title: str, message: str, category: str = "Timeline", severity: str = "info"):
        self.notifications_store.insert(0, {
            "id": len(self.notifications_store) + 1,
            "title": title,
            "category": category,
            "severity": severity,
            "message": message,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "read": False
        })

notification_service = NotificationService()
