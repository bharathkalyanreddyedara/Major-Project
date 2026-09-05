from datetime import datetime, timedelta
from typing import List, Dict, Any
from backend.app.schemas.models import TimelineStage, CropTimelineResponse

class TimelineService:
    def __init__(self):
        # Stage distribution percentages and instructions for crops
        self.stage_templates = {
            "Rice": [
                {
                    "name": "Land Preparation & Nursery Sowing",
                    "pct_start": 0, "pct_end": 15,
                    "activities": ["Ploughing, puddling, and levelling field", "Seed treatment with Carbendazim / Trichoderma", "Nursery raising for 20-25 days"],
                    "irrigation": "Maintain 2-3 cm shallow standing water in nursery beds.",
                    "fertilizer": "Apply 10 tonnes Farmyard Manure + full P & K as basal dose.",
                    "pest_watch": "Watch for stem borer and damping off in nursery beds.",
                    "notes": "Ensure level field for uniform seedling establishment."
                },
                {
                    "name": "Transplanting & Early Vegetative",
                    "pct_start": 15, "pct_end": 40,
                    "activities": ["Transplant 2-3 seedlings per hill at 20x15 cm spacing", "First manual or mechanical weeding (15-20 DAT)", "Gap filling within 7 days"],
                    "irrigation": "Maintain 3-5 cm water depth. Do not allow soil cracking.",
                    "fertilizer": "First top-dressing: Apply 25% Nitrogen (Urea) at 20-25 days after transplanting.",
                    "pest_watch": "Monitor for yellow stem borer and gall midge.",
                    "notes": "Avoid deep water standing during early tillering."
                },
                {
                    "name": "Panicle Initiation & Tillering",
                    "pct_start": 40, "pct_end": 65,
                    "activities": ["Second weeding and aeration", "Field inspection for nutrient deficiencies (Zinc / Iron)", "Monitor tiller density"],
                    "irrigation": "Critical watering stage; maintain constant shallow water (5 cm).",
                    "fertilizer": "Second top-dressing: Apply 25% Urea + 25 kg Zinc Sulphate / ha if leaves show chlorosis.",
                    "pest_watch": "Leaf folder and brown planthopper (BPH) monitoring. Spray neem oil if early infestation.",
                    "notes": "Maintain bunds to prevent nutrient leaching during monsoon."
                },
                {
                    "name": "Flowering & Grain Filling (Milky Stage)",
                    "pct_start": 65, "pct_end": 85,
                    "activities": ["Bird scaring and lodging prevention", "Check for blast or sheath blight symptoms", "Keep field free of rodents"],
                    "irrigation": "Keep soil saturated; avoid water stress as it causes chaffy grains.",
                    "fertilizer": "Final light foliar spray of 1% Potassium Nitrate (KNO3) if needed.",
                    "pest_watch": "Gundhi bug (spraying Malathion dust if threshold exceeded), False smut.",
                    "notes": "Drain excess water 10-14 days before harvest."
                },
                {
                    "name": "Maturity & Harvesting",
                    "pct_start": 85, "pct_end": 100,
                    "activities": ["Complete field drainage 10 days prior", "Harvest when 85% of grains turn golden yellow", "Threshing, cleaning, and moisture reduction to 12-14%"],
                    "irrigation": "Complete dry field to facilitate mechanical or manual harvesting.",
                    "fertilizer": "No fertilizer application.",
                    "pest_watch": "Protect harvested sheaves from moisture and post-harvest pests.",
                    "notes": "Proper sun-drying prevents fungal growth and mill breakage."
                }
            ],
            "Wheat": [
                {
                    "name": "Field Prep & Sowing (CRI Stage)",
                    "pct_start": 0, "pct_end": 20,
                    "activities": ["Fine tilth seedbed preparation", "Seed treatment with Thiram / Chlorpyrifos", "Line sowing with seed-cum-fertilizer drill"],
                    "irrigation": "First irrigation (Crown Root Initiation stage) at 21 days after sowing - crucial for yield!",
                    "fertilizer": "Full P (DAP) & K (MOP) + 50% Nitrogen as basal application.",
                    "pest_watch": "Termites and early seedling rot.",
                    "notes": "Sow at optimum depth (4-5 cm) to avoid poor germination."
                },
                {
                    "name": "Tillering & Jointing",
                    "pct_start": 20, "pct_end": 50,
                    "activities": ["First manual or chemical weeding (Sulfosulfuron)", "Monitor tiller count per plant", "Inspect soil moisture"],
                    "irrigation": "Second irrigation at late tillering (40-45 DAS).",
                    "fertilizer": "Top-dress remaining 50% Nitrogen (Urea) prior to second irrigation.",
                    "pest_watch": "Aphids and armyworms on tender leaves.",
                    "notes": "Light hoeing improves soil aeration."
                },
                {
                    "name": "Booting & Heading (Ear emergence)",
                    "pct_start": 50, "pct_end": 75,
                    "activities": ["Monitor flag leaf health", "Disease scouting for yellow/brown rust", "Check for hot dry winds"],
                    "irrigation": "Third irrigation at heading / flowering stage.",
                    "fertilizer": "Foliar spray of 2% Urea or micronutrient mixture if yellowing appears.",
                    "pest_watch": "Wheat rust (stripe/brown rust), Powdery mildew (Propiconazole if needed).",
                    "notes": "High temperature during heading can cause terminal heat stress."
                },
                {
                    "name": "Milking, Dough & Ripening",
                    "pct_start": 75, "pct_end": 100,
                    "activities": ["Final irrigation at soft dough stage (avoid during strong winds to prevent lodging)", "Inspect grain hardness", "Harvest when straw turns yellow and dry"],
                    "irrigation": "Stop irrigation 15 days before harvest.",
                    "fertilizer": "None.",
                    "pest_watch": "Storage grain pests, ear cockle nematode.",
                    "notes": "Harvesting at 14% moisture minimizes shatter losses."
                }
            ]
        }

    def generate_timeline(self, crop_name: str, sowing_date_str: str, soil_type: str = "Alluvial", location: str = "Farm") -> CropTimelineResponse:
        crop_clean = crop_name.strip().title()
        
        try:
            sowing_date = datetime.strptime(sowing_date_str, "%Y-%m-%d")
        except Exception:
            sowing_date = datetime.now()

        # Total duration based on crop type
        duration_map = {
            "Rice": 120, "Wheat": 130, "Barley": 115, "Cotton": 165,
            "Maize": 105, "Sugarcane": 360, "Groundnuts": 110, "Millets": 90,
            "Pomegranate": 210, "Pulses": 95
        }
        total_days = duration_map.get(crop_clean, 120)
        harvest_date = sowing_date + timedelta(days=total_days)

        current_day = max(0, (datetime.now() - sowing_date).days)
        
        # Pick or construct stages
        template = self.stage_templates.get(crop_clean, self.stage_templates["Rice"])
        stages: List[TimelineStage] = []
        current_stage_name = "Preparation / Sowing"
        notifications = []

        for idx, t in enumerate(template, 1):
            start_d = int((t["pct_start"] / 100.0) * total_days)
            end_d = int((t["pct_end"] / 100.0) * total_days)

            status = "upcoming"
            if current_day > end_d:
                status = "completed"
            elif start_d <= current_day <= end_d:
                status = "current"
                current_stage_name = t["name"]
            
            # Construct proactive stage notification
            if status == "current":
                notifications.append({
                    "title": f"Active Stage: {t['name']}",
                    "severity": "info",
                    "message": f"Crop is at Day {current_day}. Recommended action: {t['activities'][0]}. Irrigation advice: {t['irrigation']}",
                    "date": datetime.now().strftime("%Y-%m-%d")
                })
            elif status == "upcoming" and (start_d - current_day) <= 5:
                notifications.append({
                    "title": f"Upcoming Stage in {start_d - current_day} days: {t['name']}",
                    "severity": "warning",
                    "message": f"Prepare for upcoming tasks: {t['fertilizer']}",
                    "date": (datetime.now() + timedelta(days=max(1, start_d - current_day))).strftime("%Y-%m-%d")
                })

            stages.append(TimelineStage(
                stage_id=idx,
                stage_name=t["name"],
                start_day=start_d,
                end_day=end_d,
                status=status,
                activities=t["activities"],
                irrigation_schedule=t["irrigation"],
                fertilizer_advice=t["fertilizer"],
                pest_disease_watch=t["pest_watch"],
                critical_notes=t["notes"]
            ))

        return CropTimelineResponse(
            crop_name=crop_clean,
            sowing_date=sowing_date.strftime("%Y-%m-%d"),
            expected_harvest_date=harvest_date.strftime("%Y-%m-%d"),
            total_duration_days=total_days,
            current_day=current_day,
            current_stage=current_stage_name,
            stages=stages,
            active_notifications=notifications
        )

timeline_service = TimelineService()
