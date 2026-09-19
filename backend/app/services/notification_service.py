from datetime import datetime, timedelta
from typing import List, Dict, Any
from backend.app.services.weather_service import weather_service

class NotificationService:
    def __init__(self):
        self.notifications_store = [
            {
                "id": 1,
                "title": "Welcome to Agro-AI Farm Intelligence",
                "category": "System Telemetry",
                "severity": "info",
                "message": "Enter your manual soil test values or upload a soil photo to get personalized crop recommendations and dynamic lifecycle timelines.",
                "action_required": "Review soil parameters and select active crop for season.",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "read": False
            }
        ]

    def get_notifications(
        self,
        crop_name: str = None,
        growth_stage: str = None,
        sowing_date: str = None,
        city: str = "Hyderabad",
        **kwargs
    ) -> List[Dict[str, Any]]:
        # Import lazily to avoid circular imports
        from backend.app.services.timeline_service import timeline_service

        # Check live weather for automated warnings
        weather = weather_service.get_weather(city=city)
        results = list(self.notifications_store)

        rainfall = weather.get("rainfall", 0.0) or 0.0
        temp = weather.get("temperature", 25.0) or 25.0
        humidity = weather.get("humidity", 50.0) or 50.0
        wind = weather.get("wind_speed", 10.0) or 10.0
        condition = weather.get("weather_condition", "Clear")

        crop_clean = (crop_name or "Rice").strip().title()

        # If sowing_date is available, calculate real dynamic stage
        dynamic_stage_info = None
        if sowing_date:
            try:
                t_resp = timeline_service.generate_timeline(crop_name=crop_clean, sowing_date=sowing_date)
                growth_stage = t_resp.get("current_stage", growth_stage)
                curr_day = t_resp.get("current_day", 0)
                stages = t_resp.get("stages", [])
                
                for stg in stages:
                    if stg.get("status") == "current":
                        dynamic_stage_info = stg
                        break
                    
                # Add timeline upcoming stage notifications
                for t_notif in t_resp.get("active_notifications", []):
                    results.append({
                        "id": len(results) + 200,
                        "title": t_notif["title"],
                        "category": "Timeline Schedule",
                        "severity": t_notif.get("severity", "info"),
                        "message": t_notif["message"],
                        "action_required": "Procure inputs and schedule field operations.",
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "read": False
                    })
            except Exception as e:
                print(f"[NotificationService] Timeline calculation note: {e}")

        # 1. Severe Weather Hazards
        if rainfall > 15.0:
            results.insert(0, {
                "id": len(results) + 101,
                "title": f"🚨 Severe Rainstorm & Waterlogging Hazard ({city})",
                "category": "Meteorological Hazard",
                "severity": "critical",
                "message": f"Precipitation estimated at {rainfall} mm with wind speeds of {wind} km/h in {city}.",
                "action_required": f"Halt all pesticide and fertilizer sprays on {crop_clean}. Clear peripheral field drainage trenches immediately to prevent root asphyxiation.",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "read": False
            })
        elif rainfall > 5.0:
            results.insert(0, {
                "id": len(results) + 101,
                "title": f"🌧️ Moderate Rain Advisory ({city})",
                "category": "Weather Alert",
                "severity": "warning",
                "message": f"Rainfall expected ({rainfall} mm). Root zone soil moisture will remain elevated.",
                "action_required": "Delay scheduled irrigation by 48-72 hours to save energy and prevent fertilizer leaching.",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "read": False
            })

        if temp > 36.0:
            results.insert(0, {
                "id": len(results) + 102,
                "title": f"☀️ Extreme Heatwave & Thermal Stress Warning ({temp}°C)",
                "category": "Thermal Hazard",
                "severity": "warning",
                "message": f"High ambient temperatures in {city} will accelerate crop evapotranspiration.",
                "action_required": f"Provide light evening sprinkler irrigation for {crop_clean} and apply straw mulch to protect root crown.",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "read": False
            })
        elif temp < 10.0:
            results.insert(0, {
                "id": len(results) + 103,
                "title": f"❄️ Frost & Low Temperature Advisory ({temp}°C)",
                "category": "Thermal Hazard",
                "severity": "warning",
                "message": f"Night temperatures dropping to {temp}°C in {city}.",
                "action_required": "Schedule early-evening light flood irrigation to maintain soil heat capacity above freezing point.",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "read": False
            })

        if humidity > 80.0 and temp > 22.0:
            results.insert(0, {
                "id": len(results) + 104,
                "title": f"🍄 High Pathogen & Fungal Spore Risk ({humidity}% RH)",
                "category": "Pathology Alert",
                "severity": "warning",
                "message": f"High humidity ({humidity}%) and warm weather create optimal incubation for foliar fungi in {crop_clean}.",
                "action_required": f"Scout {crop_clean} canopy for blast, blight, or powdery mildew. Apply preventative bio-fungicide (*Trichoderma* / *Pseudomonas*).",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "read": False
            })

        # 2. Crop & Active Growth Stage Dynamic Notification
        if crop_name and dynamic_stage_info:
            results.append({
                "id": len(results) + 105,
                "title": f"🌾 {crop_clean} Active Stage: {dynamic_stage_info.get('stage_name')}",
                "category": "Agronomic Protocol",
                "severity": "info",
                "message": f"Field Activities: {', '.join(dynamic_stage_info.get('activities', []))}. Irrigation: {dynamic_stage_info.get('irrigation_schedule')}",
                "action_required": f"Fertilizer: {dynamic_stage_info.get('fertilizer_advice')}. Watch: {dynamic_stage_info.get('pest_disease_watch')}",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "read": False
            })
        elif crop_name and growth_stage:
            results.append({
                "id": len(results) + 105,
                "title": f"🌾 {crop_clean} Stage Alert: {growth_stage}",
                "category": "Agronomic Protocol",
                "severity": "info",
                "message": f"Your {crop_clean} is in the {growth_stage} phase.",
                "action_required": "Ensure timely top-dressing of nutrients and monitor root zone soil moisture.",
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
