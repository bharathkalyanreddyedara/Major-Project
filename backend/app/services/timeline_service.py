from datetime import datetime, timedelta
from typing import List, Dict, Any
from backend.app.schemas.models import TimelineStage, CropTimelineResponse

class TimelineService:
    def __init__(self):
        # Specific agronomic stages and activity templates for all crops
        self.stage_templates = {
            "Rice": [
                {
                    "name": "Land Preparation & Sowing",
                    "pct_start": 0, "pct_end": 18,
                    "activities": ["Puddling, bunding and field levelling", "Seed treatment with Trichoderma (10g/kg)", "Nursery sowing or direct drum seeding"],
                    "irrigation": "Maintain 2-3 cm shallow standing water.",
                    "fertilizer": "Apply 10 tonnes FYM + 100% P (DAP) & K (MOP) as basal dose.",
                    "pest_watch": "Monitor for yellow stem borer moths and damping off.",
                    "notes": "Ensure weed-free nursery bed for healthy seedling vigor."
                },
                {
                    "name": "Transplanting & Active Tillering",
                    "pct_start": 18, "pct_end": 45,
                    "activities": ["Transplant 2-3 seedlings/hill at 20x15 cm spacing", "First manual or cono-weeding at 20 DAT", "Check tiller count per hill (target 15-20)"],
                    "irrigation": "Maintain 3-5 cm water layer; prevent soil cracking.",
                    "fertilizer": "First top-dressing: 25% Nitrogen (Urea) at 20-25 DAT.",
                    "pest_watch": "Scout for leaf folders (folded leaves) and gall midge.",
                    "notes": "Keep bunds intact to avoid fertilizer runoff."
                },
                {
                    "name": "Panicle Initiation & Booting",
                    "pct_start": 45, "pct_end": 70,
                    "activities": ["Second weeding and rogueing off-types", "Inspect for sheath blight and leaf blast symptoms", "Ensure uninterrupted water supply"],
                    "irrigation": "Critical watering stage! Water stress at this stage severely reduces yield.",
                    "fertilizer": "Second top-dressing: 25% Nitrogen (Urea) + 25 kg/ha Zinc Sulphate if yellowing.",
                    "pest_watch": "Brown Planthopper (BPH) monitoring at base of plant clumps.",
                    "notes": "Do not allow field to dry during panicle emergence."
                },
                {
                    "name": "Flowering & Grain Filling (Milky to Dough)",
                    "pct_start": 70, "pct_end": 88,
                    "activities": ["Bird scaring in early morning/evening", "Rodent control around field borders", "Foliar health monitoring"],
                    "irrigation": "Maintain shallow water (2-3 cm) until 10 days before harvest.",
                    "fertilizer": "Foliar spray of 1% Potassium Nitrate (13:0:45) for grain luster.",
                    "pest_watch": "Gundhi bug scouting during milky grain stage (spray Malathion if >2 bugs/hill).",
                    "notes": "Stop all pesticide sprays 15 days before harvest."
                },
                {
                    "name": "Maturity & Harvesting",
                    "pct_start": 88, "pct_end": 100,
                    "activities": ["Complete field drainage 10 days before harvest", "Harvest when 85-90% grains turn golden yellow", "Threshing, winnowing, and sun-drying to 12% moisture"],
                    "irrigation": "Completely drain field to facilitate harvesting machinery.",
                    "fertilizer": "No fertilizer application.",
                    "pest_watch": "Protect harvested sheaves from moisture and storage weevils.",
                    "notes": "Proper drying prevents aflatoxin contamination and grain breakage."
                }
            ],
            "Wheat": [
                {
                    "name": "Field Prep & Sowing (CRI Stage)",
                    "pct_start": 0, "pct_end": 20,
                    "activities": ["Fine tilth seedbed preparation", "Seed treatment with Thiram / Chlorpyrifos", "Line sowing with seed-cum-fertilizer drill at 20 cm spacing"],
                    "irrigation": "First irrigation at 21 DAS (Crown Root Initiation stage) - Most critical for root establishment!",
                    "fertilizer": "Full P (DAP) & K (MOP) + 50% Nitrogen as basal application.",
                    "pest_watch": "Termites and early seedling rot in dry patches.",
                    "notes": "Sow at optimum depth (4-5 cm) for uniform germination."
                },
                {
                    "name": "Tillering & Jointing",
                    "pct_start": 20, "pct_end": 50,
                    "activities": ["First weeding (Sulfosulfuron at 30-35 DAS)", "Monitor tiller count per square meter", "Inter-cultivation for soil aeration"],
                    "irrigation": "Second irrigation at late tillering stage (40-45 DAS).",
                    "fertilizer": "Top-dress remaining 50% Nitrogen (Urea) prior to second irrigation.",
                    "pest_watch": "Aphids on young shoots and armyworm leaf feeding.",
                    "notes": "Avoid excessive nitrogen to prevent crop lodging."
                },
                {
                    "name": "Booting & Heading (Ear Emergence)",
                    "pct_start": 50, "pct_end": 75,
                    "activities": ["Disease scouting for yellow/brown rust on flag leaves", "Monitor canopy temperature", "Erect wind barriers if strong winds expected"],
                    "irrigation": "Third irrigation at heading / flowering stage.",
                    "fertilizer": "Foliar spray of 2% Urea or micronutrient mixture if leaves show chlorosis.",
                    "pest_watch": "Wheat rust (stripe rust) - spray Propiconazole 25% EC if yellow pustules appear.",
                    "notes": "High temperature during heading can cause terminal heat stress."
                },
                {
                    "name": "Milking, Dough & Ripening",
                    "pct_start": 75, "pct_end": 100,
                    "activities": ["Final irrigation at soft dough stage (avoid on windy days)", "Inspect grain firmness", "Harvest when straw turns golden yellow and dry"],
                    "irrigation": "Stop all irrigation 15 days before harvest.",
                    "fertilizer": "None.",
                    "pest_watch": "Storage grain borers and ear cockle.",
                    "notes": "Harvesting at 14% moisture minimizes shattering losses."
                }
            ],
            "Cotton": [
                {
                    "name": "Land Preparation & Sowing",
                    "pct_start": 0, "pct_end": 20,
                    "activities": ["Deep summer ploughing to eradicate resting pupae", "Form ridges and furrows at 90 cm spacing", "Dibble seeds at 2.5-3 cm depth"],
                    "irrigation": "Light pre-sowing irrigation to ensure uniform germination.",
                    "fertilizer": "Apply 10 t FYM + 50% P & K as basal dose.",
                    "pest_watch": "Sucking pests (thrips, jassids) on cotyledon leaves.",
                    "notes": "Maintain optimum plant population (55,000 plants/ha for Bt hybrids)."
                },
                {
                    "name": "Square Formation & Vegetative Growth",
                    "pct_start": 20, "pct_end": 50,
                    "activities": ["First & second weeding/hoeing at 25 & 45 DAS", "Monopodial branch trimming if overgrown", "Nipping terminal shoot at 90 DAS"],
                    "irrigation": "Irrigate at 12-15 day intervals depending on soil type.",
                    "fertilizer": "First top-dressing: 33% Nitrogen + 25 kg Zinc Sulphate/ha.",
                    "pest_watch": "Pink bollworm monitoring using pheromone traps (ETL: 8 moths/trap/night).",
                    "notes": "Avoid waterlogging; cotton is sensitive to standing water."
                },
                {
                    "name": "Flowering & Boll Development",
                    "pct_start": 50, "pct_end": 80,
                    "activities": ["Scout for boll shedding and flower rot", "Foliar spray of Planofix (NAA) at 4.5 ml/10L to prevent boll drop", "Maintain clean inter-row spaces"],
                    "irrigation": "Critical moisture stage; maintain regular furrow irrigation.",
                    "fertilizer": "Second top-dressing: 33% Nitrogen + 2% DAP foliar spray.",
                    "pest_watch": "Whitefly (spray Flonicamid 50 WG if above threshold), American bollworm.",
                    "notes": "Boll retention directly determines final cotton yield."
                },
                {
                    "name": "Boll Bursting & Picking",
                    "pct_start": 80, "pct_end": 100,
                    "activities": ["Defoliation if required", "First picking when 60% bolls burst open cleanly", "Second picking 15-20 days later", "Sun-dry seed cotton to 8% moisture"],
                    "irrigation": "Withhold irrigation 20 days prior to first picking.",
                    "fertilizer": "None.",
                    "pest_watch": "Stainers and pink bollworm entry holes in green bolls.",
                    "notes": "Pick cotton in dry morning hours without leaf/bract contamination."
                }
            ],
            "Maize": [
                {
                    "name": "Sowing & Seedling Establishment",
                    "pct_start": 0, "pct_end": 20,
                    "activities": ["Form ridges and furrows at 60 cm spacing", "Seed treatment with Imidacloprid (4ml/kg)", "Dibble single seed at 4 cm depth"],
                    "irrigation": "Initial light irrigation for germination.",
                    "fertilizer": "Apply full P & K + 25% N as basal dose.",
                    "pest_watch": "Fall Armyworm (FAW) egg masses and early pinhole leaf damage.",
                    "notes": "Early scouting for FAW is vital in first 30 days."
                },
                {
                    "name": "Knee-High & Tasseling Stage",
                    "pct_start": 20, "pct_end": 60,
                    "activities": ["Inter-cultivation and earthing-up at knee-high stage", "Whorl application of Neem seed kernel extract / Chlorantraniliprole for FAW", "Weeding"],
                    "irrigation": "Critical irrigation at tasseling/silking stage.",
                    "fertilizer": "Top-dress 50% Nitrogen (Urea) at knee-high stage before irrigation.",
                    "pest_watch": "Stem borer and Fall Armyworm whorl feeding.",
                    "notes": "Moisture stress at silking causes poor kernel setting."
                },
                {
                    "name": "Cob Development, Grain Filling & Harvest",
                    "pct_start": 60, "pct_end": 100,
                    "activities": ["Inspect cob filling and husk cover", "Harvest when husk leaves turn straw-yellow and dry", "De-husking, shelling, and moisture drying to 13%"],
                    "irrigation": "Stop irrigation 10 days before harvesting.",
                    "fertilizer": "Top-dress remaining 25% Nitrogen at early grain formation.",
                    "pest_watch": "Cob borers and grain rot fungi.",
                    "notes": "Proper drying prevents aflatoxin mold during storage."
                }
            ]
        }

        self.crop_durations = {
            "Rice": 120, "Wheat": 130, "Barley": 115, "Cotton": 160,
            "Maize": 105, "Sugarcane": 360, "Groundnuts": 110, "Millets": 90,
            "Pomegranate": 210, "Pulses": 95, "Chickpea": 100, "Kidneybeans": 90,
            "Pigeonpeas": 160, "Mothbeans": 80, "Mungbean": 75, "Blackgram": 80,
            "Lentil": 110, "Coffee": 270, "Jute": 125, "Coconut": 365,
            "Apple": 180, "Orange": 240, "Papaya": 270, "Banana": 330,
            "Mango": 150, "Grapes": 140, "Watermelon": 85, "Muskmelon": 80
        }

    def generate_timeline(self, crop_name: str, sowing_date_str: str, soil_type: str = "Alluvial", location: str = "Farm") -> CropTimelineResponse:
        crop_clean = crop_name.strip().title()
        
        try:
            sowing_date = datetime.strptime(sowing_date_str, "%Y-%m-%d")
        except Exception:
            sowing_date = datetime.now()

        total_days = self.crop_durations.get(crop_clean, 120)
        harvest_date = sowing_date + timedelta(days=total_days)

        current_day = max(0, (datetime.now() - sowing_date).days)
        
        # Select appropriate stage template or adapt
        template = self.stage_templates.get(crop_clean, self.stage_templates.get("Rice"))
        
        stages: List[TimelineStage] = []
        current_stage_name = "Preparation / Sowing"
        notifications = []

        for idx, t in enumerate(template, 1):
            start_d = int((t["pct_start"] / 100.0) * total_days)
            end_d = int((t["pct_end"] / 100.0) * total_days)

            # Calculate real calendar dates for every stage
            stage_start_date = (sowing_date + timedelta(days=start_d)).strftime("%Y-%m-%d")
            stage_end_date = (sowing_date + timedelta(days=end_d)).strftime("%Y-%m-%d")

            status = "upcoming"
            if current_day > end_d:
                status = "completed"
            elif start_d <= current_day <= end_d:
                status = "current"
                current_stage_name = t["name"]
            
            # Construct dynamic notification
            if status == "current":
                notifications.append({
                    "title": f"Active Stage: {t['name']} (Day {current_day})",
                    "severity": "info",
                    "message": f"Recommended action: {t['activities'][0]}. Irrigation: {t['irrigation']}",
                    "date": stage_start_date
                })
            elif status == "upcoming" and (start_d - current_day) <= 7 and (start_d - current_day) > 0:
                notifications.append({
                    "title": f"Upcoming in {start_d - current_day} days: {t['name']}",
                    "severity": "warning",
                    "message": f"Prepare fertilizers and supplies: {t['fertilizer']}",
                    "date": stage_start_date
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
                critical_notes=f"Scheduled: {stage_start_date} to {stage_end_date}. {t['notes']}"
            ))

        return CropTimelineResponse(
            crop_name=crop_clean,
            sowing_date=sowing_date.strftime("%Y-%m-%d"),
            expected_harvest_date=harvest_date.strftime("%Y-%m-%d"),
            total_duration_days=total_days,
            current_day=min(total_days, current_day),
            current_stage=current_stage_name,
            stages=stages,
            active_notifications=notifications
        )

timeline_service = TimelineService()
