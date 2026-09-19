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
                    "fertilizer": "Second top-dressing: 33% Nitrogen + 2% DAP foliar spray + 1% MgSO4 against leaf reddening.",
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
            ],
            "Sugarcane": [
                {
                    "name": "Land Prep & Sett Planting (Germination Stage)",
                    "pct_start": 0, "pct_end": 15,
                    "activities": ["Deep ploughing and furrow formation at 90-120 cm spacing", "Two or three budded sett treatment in Carbendazim (1g/L) for 15 mins", "Sett placement in furrows and light soil covering"],
                    "irrigation": "Immediate germination irrigation followed by second light watering at 7-10 days.",
                    "fertilizer": "Basal: 25% Nitrogen + 100% P2O5 (SSP/DAP) + 25 tonnes FYM/ha.",
                    "pest_watch": "Early shoot borer (dead hearts) and termite damage on cut setts.",
                    "notes": "Use healthy, disease-free seed canes of 8-10 months age."
                },
                {
                    "name": "Tillering & Formative Phase",
                    "pct_start": 15, "pct_end": 35,
                    "activities": ["First weeding and light earthing up at 45 DAP", "Trash mulching in alternate furrows to conserve soil moisture", "Trash shedding and shoot count monitoring (target 100,000 canes/ha)"],
                    "irrigation": "Irrigate at 8-10 day intervals during peak summer.",
                    "fertilizer": "First top-dressing: 25% Nitrogen + 25 kg/ha Zinc Sulphate at 45 DAP.",
                    "pest_watch": "Early shoot borer and internode borer.",
                    "notes": "Tillering determines millable cane population."
                },
                {
                    "name": "Grand Growth & Cane Elongation Phase",
                    "pct_start": 35, "pct_end": 75,
                    "activities": ["Heavy earthing up at 90-120 DAP to prevent lodging", "De-trashing lower dried leaves at 150 & 210 DAP for aeration", "Propping / wrapping canes together to resist cyclonic winds"],
                    "irrigation": "Regular irrigation at 10-12 days interval. High water requirement phase.",
                    "fertilizer": "Second & third top-dressings: 50% Nitrogen in 2 split doses by 120 DAP.",
                    "pest_watch": "Top borer, pyrilla, and red rot symptoms (drying of crown leaves).",
                    "notes": "Avoid late nitrogen application which reduces juice sugar content."
                },
                {
                    "name": "Ripening, Sugar Accumulation & Harvest",
                    "pct_start": 75, "pct_end": 100,
                    "activities": ["Brix testing with hand refractometer (target 18-20% Brix)", "Withhold irrigation 15-20 days prior to harvest", "Base level cane cutting close to ground without stubs"],
                    "irrigation": "Stop irrigation 20 days before harvest to enhance sucrose crystallization.",
                    "fertilizer": "No fertilizer. Foliar spray of Potassium Schoenite (1%) can aid maturity.",
                    "pest_watch": "Rodent damage and wild animal fencing.",
                    "notes": "Crush harvested cane within 24-48 hours to prevent sugar inversion."
                }
            ],
            "Groundnut": [
                {
                    "name": "Sowing & Seedling Establishment",
                    "pct_start": 0, "pct_end": 20,
                    "activities": ["Seedbed preparation with fine loose tilth", "Kernel treatment with Trichoderma (10g/kg) + Rhizobium inoculant", "Sowing at 30x10 cm spacing at 5 cm depth"],
                    "irrigation": "Pre-sowing irrigation; avoid standing water.",
                    "fertilizer": "Basal: 25 kg N : 50 kg P2O5 : 40 kg K2O/ha + 10 t FYM.",
                    "pest_watch": "Collar rot, seedling damping off, and thrips/aphids.",
                    "notes": "Use bold, unbroken certified kernels."
                },
                {
                    "name": "Vegetative & Flowering Stage",
                    "pct_start": 20, "pct_end": 40,
                    "activities": ["Hand weeding and inter-cultivation at 20-25 DAS", "Scout for yellow flower flush across field canopy", "Do NOT hoe or disturb soil after 45 DAS!"],
                    "irrigation": "Irrigate at 10-12 day intervals.",
                    "fertilizer": "Foliar spray of 0.5% Ferrous Sulphate + 0.1% Citric acid if chlorosis occurs.",
                    "pest_watch": "Spodoptera litura (leaf miner) and leaf spot (Tikka disease).",
                    "notes": "Loose soil surface facilitates seamless peg penetration."
                },
                {
                    "name": "Pegging & Pod Development (Critical Stage)",
                    "pct_start": 40, "pct_end": 75,
                    "activities": ["Soil application of Gypsum @ 400 kg/ha at pegging (40-45 DAS)", "Light earthing up to support pegs entering soil", "Strictly avoid mechanical weeding that snaps pegs"],
                    "irrigation": "Most critical irrigation stage! Moisture stress causes empty shells (pops).",
                    "fertilizer": "Gypsum provides essential Calcium (for shell hardening) and Sulphur (for oil synthesis).",
                    "pest_watch": "Tikka leaf spot (spray Carbendazim 12% + Mancozeb 63% WP @ 2 g/L) and rust.",
                    "notes": "Calcium is directly absorbed by developing pods from surrounding soil."
                },
                {
                    "name": "Pod Maturity & Harvesting",
                    "pct_start": 75, "pct_end": 100,
                    "activities": ["Sample 5 plants: mature pods show dark black inner shell lining", "Harvesting / uprooting by tractor digger or manual pulling", "Pod stripping, field curing and sun-drying to 8% moisture"],
                    "irrigation": "Give one light pre-harvest irrigation 2 days prior to ease digging.",
                    "fertilizer": "None.",
                    "pest_watch": "Aflatoxin mold (*Aspergillus flavus*) prevention during drying.",
                    "notes": "Dry pods thoroughly on clean tarpaulins before bagging."
                }
            ],
            "Tomato": [
                {
                    "name": "Pro-Tray Nursery & Field Prep",
                    "pct_start": 0, "pct_end": 22,
                    "activities": ["Raise seedlings in 98-cell pro-trays using sterilized coco-peat", "Solarize main field and lay raised beds with silver-black plastic mulch", "Install drip irrigation lines beneath mulch"],
                    "irrigation": "Nursery misting twice daily; field bed wetting before transplanting.",
                    "fertilizer": "Basal: 10 tonnes FYM + 50 kg DAP + 50 kg SOP per hectare.",
                    "pest_watch": "Damping off in nursery and Whitefly vector monitoring.",
                    "notes": "Hardening of seedlings 4 days before transplanting."
                },
                {
                    "name": "Transplanting & Staking / Trellising",
                    "pct_start": 22, "pct_end": 45,
                    "activities": ["Transplant 25-day old seedlings at 90x60 cm spacing", "Root dip in Imidacloprid (0.5 ml/L) to prevent vector transmission of ToLCV", "Erect bamboo stakes / trellis wire for plant support at 30 DAT"],
                    "irrigation": "Daily drip fertigation (1-2 hours depending on ET).",
                    "fertilizer": "Start fertigation: 19:19:19 @ 5 kg/ha every 4th day.",
                    "pest_watch": "Tomato pinworm (*Tuta absoluta*) and Leaf miner (serpentine mines).",
                    "notes": "Prune ground-touching lower suckers for canopy airflow."
                },
                {
                    "name": "Flowering & Fruit Set",
                    "pct_start": 45, "pct_end": 70,
                    "activities": ["Vibrate trellis wires or maintain bee colonies for cross pollination", "Foliar spray of Boron (0.1%) + Calcium Nitrate (0.5%) to prevent Blossom End Rot", "De-suckering side shoots on indeterminate hybrids"],
                    "irrigation": "Maintain constant soil moisture; fluctuation causes fruit cracking.",
                    "fertilizer": "Fertigation: Switch to 12:61:0 (Mono Ammonium Phosphate) + 0:52:34.",
                    "pest_watch": "Early blight (concentric leaf rings) and Fruit borer (*Helicoverpa armigera*).",
                    "notes": "Night temperatures between 16-20°C optimize pollen viability."
                },
                {
                    "name": "Fruit Sizing, Color Break & Multiple Pickings",
                    "pct_start": 70, "pct_end": 100,
                    "activities": ["Harvest fruits at 'Breaker' or 'Pink' stage for distant markets", "Harvest at full red ripe stage for local processing", "Grade fruits by size and pack in plastic crates"],
                    "irrigation": "Taper drip irrigation slightly to enhance sugar and TSS content.",
                    "fertilizer": "Fertigation: Potassium Nitrate (13:0:45) @ 6 kg/ha to enhance fruit redness and firmness.",
                    "pest_watch": "Late blight (*Phytophthora*) during overcast humid spells and fruit rot.",
                    "notes": "Handle harvested tomatoes gently without stacking more than 4 crate layers."
                }
            ],
            "Potato": [
                {
                    "name": "Sprouting, Planting & Emergence",
                    "pct_start": 0, "pct_end": 25,
                    "activities": ["Take out cold-stored seed tubers 10-14 days prior to break dormancy and sprout", "Treat seed tubers with Mancozeb (2.5g/L)", "Plant on ridges at 60x20 cm spacing"],
                    "irrigation": "First light irrigation immediately after planting; avoid ridge submergence.",
                    "fertilizer": "Basal: 50% N + 100% P + 100% K (preferably SOP - Sulphate of Potash) + 20 t FYM.",
                    "pest_watch": "Black scurf (seed-borne) and cutworms.",
                    "notes": "Slightly acidic to neutral soil (pH 5.5-6.5) prevents potato common scab."
                },
                {
                    "name": "Stolon Initiation & Earthing-Up",
                    "pct_start": 25, "pct_end": 50,
                    "activities": ["First earthing-up when plants reach 15-20 cm height", "Weeding and loose soil mounding around plant base", "Top-dress remaining 50% Nitrogen (Urea)"],
                    "irrigation": "Irrigate at 7-10 day intervals; maintain loose, friable ridge soil.",
                    "fertilizer": "Top-dress Urea before earthing up; spray 0.2% Zinc Sulphate.",
                    "pest_watch": "Potato aphids (*Myzus persicae*) - crucial vector for potato viruses.",
                    "notes": "Loose soil allows stolons to hook downwards and initiate tubers."
                },
                {
                    "name": "Tuberization & Bulking Phase",
                    "pct_start": 50, "pct_end": 80,
                    "activities": ["Second light earthing-up to cover any exposed tubers (prevents greening/solanine)", "Inspect canopy for late blight water-soaked spots with white fungal margin", "Maintain uniform moisture"],
                    "irrigation": "Critical water demand! Moisture deficit severely reduces tuber size.",
                    "fertilizer": "Foliar spray of 1% Potassium Nitrate (13:0:45) + 0.2% Borax for skin finish.",
                    "pest_watch": "Late blight (*Phytophthora infestans*) - spray Cymoxanil + Mancozeb @ 2.5 g/L.",
                    "notes": "Night temperature below 20°C is mandatory for rapid tuber bulking."
                },
                {
                    "name": "Dehaulming, Curing & Harvest",
                    "pct_start": 80, "pct_end": 100,
                    "activities": ["Dehaulming: Cut potato foliage at ground level 12-15 days before harvest", "Allow skin to cure and harden inside the soil", "Harvest with potato digger in dry soil conditions"],
                    "irrigation": "Stop all watering 15 days before harvest.",
                    "fertilizer": "None.",
                    "pest_watch": "Potato Tuber Moth (PTM) in field and storage bins.",
                    "notes": "Cure dug potatoes in shade for 7 days before cold storage."
                }
            ],
            "Chickpea": [
                {
                    "name": "Sowing & Seedling Establishment",
                    "pct_start": 0, "pct_end": 20,
                    "activities": ["Seedbed preparation with adequate residual soil moisture", "Seed treatment with Rhizobium (20g/kg) + PSB + Trichoderma (5g/kg)", "Line sowing at 30x10 cm at 6-8 cm depth"],
                    "irrigation": "Pre-sowing irrigation; avoid waterlogging at all costs.",
                    "fertilizer": "Basal: 20 kg N : 50 kg P2O5 : 20 kg K2O/ha + 20 kg/ha Sulphur.",
                    "pest_watch": "Collar rot and Fusarium wilt on seedlings.",
                    "notes": "Deep sowing utilizes receding monsoon moisture."
                },
                {
                    "name": "Vegetative Branching & Nipping",
                    "pct_start": 20, "pct_end": 45,
                    "activities": ["Nipping / detopping terminal shoots at 35-40 DAS to promote profuse lateral branching", "Hand weeding or hoeing at 30 DAS", "Inspect root nodules (should be healthy pink inside)"],
                    "irrigation": "First protective irrigation at pre-flowering stage (45 DAS) if soil is dry.",
                    "fertilizer": "No nitrogen needed; Rhizobia fix atmospheric nitrogen.",
                    "pest_watch": "Cutworms and early Helicoverpa pod borer larvae on leaves.",
                    "notes": "Nipping increases pod-bearing secondary branches by 30-40%."
                },
                {
                    "name": "Flowering & Pod Formation",
                    "pct_start": 45, "pct_end": 75,
                    "activities": ["Install pheromone traps @ 5/acre for *Helicoverpa armigera*", "Erect bird perches (T-shaped wooden poles @ 20/acre) for natural predation", "Foliar spray of 2% Urea at flowering to prevent flower drop"],
                    "irrigation": "Second irrigation at early pod filling stage.",
                    "fertilizer": "Foliar spray of 2% DAP or Potassium Nitrate (1%).",
                    "pest_watch": "Gram pod borer (*Helicoverpa*) - spray Chlorantraniliprole 18.5 SC @ 0.3 ml/L at ETL (1 larva/meter row).",
                    "notes": "Do NOT irrigate during peak full bloom; it causes flower drop and vegetative overgrowth."
                },
                {
                    "name": "Pod Filling & Harvesting",
                    "pct_start": 75, "pct_end": 100,
                    "activities": ["Inspect pods: grains turn hard and rattle inside dry yellow-brown pods", "Harvest early morning to prevent pod shattering", "Threshing and sun-drying to 9-10% moisture"],
                    "irrigation": "Withhold irrigation during pod drying.",
                    "fertilizer": "None.",
                    "pest_watch": "Pulse beetle (*Callosobruchus*) in storage - mix neem oil (5 ml/kg seed).",
                    "notes": "Store in airtight bins with dry neem leaves."
                }
            ],
            "Banana": [
                {
                    "name": "Pit Preparation & Sucker / Tissue Culture Planting",
                    "pct_start": 0, "pct_end": 20,
                    "activities": ["Dig pits of 60x60x60 cm at 1.8x1.8 m spacing", "Fill pits with 10 kg FYM + 250g Neem cake + 20g Carbofuran", "Plant certified sword suckers or primary-hardened Tissue Culture (TC) plantlets"],
                    "irrigation": "Immediate flood/drip irrigation; keep root zone continuously moist.",
                    "fertilizer": "Basal: 50g DAP + 100g MOP per plant.",
                    "pest_watch": "Banana pseudostem weevil and Rhizome weevil.",
                    "notes": "TC plants give uniform flowering and higher bunch weight."
                },
                {
                    "name": "Vegetative Pseudostem Growth & Sucker Pruning",
                    "pct_start": 20, "pct_end": 55,
                    "activities": ["De-suckering: Remove all side suckers until bunch emergence to channel nutrients to mother plant", "Earthing up around pseudostem at 3rd and 5th month", "Weed-free ring basins"],
                    "irrigation": "Drip irrigation @ 15-20 litres/plant/day.",
                    "fertilizer": "Monthly fertigation: 50g Urea + 60g MOP per plant in split monthly doses.",
                    "pest_watch": "Sigatoka leaf spot (spray Propiconazole @ 1 ml/L with mineral oil).",
                    "notes": "Healthy banana requires 30-35 functional green leaves."
                },
                {
                    "name": "Shooting, Bunch Emergence & Propping",
                    "pct_start": 55, "pct_end": 80,
                    "activities": ["Propping: Support heavy bunch-bearing trees with double bamboo poles", "Denavelling: Remove male bud (heart) 15 days after last hand opens", "Sleeve bunches with 100-gauge blue perforated polythene bags"],
                    "irrigation": "High water demand! 25-30 litres/plant/day.",
                    "fertilizer": "High Potassium feeding: Apply 100g SOP / MOP + 10g Borax per tree.",
                    "pest_watch": "Bunchy top virus (aphid vector *Pentalonia nigronervosa*) and flower thrips.",
                    "notes": "Bunch covering protects fingers from sunscald and thrip scarring."
                },
                {
                    "name": "Bunch Maturation & Harvesting",
                    "pct_start": 80, "pct_end": 100,
                    "activities": ["Harvest when bunch finger ridges turn round and angularity disappears (75-80% maturity)", "Cut bunch leaving 30 cm peduncle handle", "Dehanding, washing in alum water, and packing in CFB cartons"],
                    "irrigation": "Reduce irrigation 10 days before bunch cutting.",
                    "fertilizer": "None.",
                    "pest_watch": "Post-harvest crown rot and anthracnose.",
                    "notes": "Transport bunches upright with cushioning padding."
                }
            ],
            "Mango": [
                {
                    "name": "Post-Harvest Pruning & Basin Preparation",
                    "pct_start": 0, "pct_end": 25,
                    "activities": ["Light center pruning of crossed, dead, and diseased twigs for canopy sunlight penetration", "Clean tree basin (drip line) and apply 50 kg well-decomposed FYM + 2 kg Neem cake per tree", "Paclobutrazol soil application (if regulated flowering needed)"],
                    "irrigation": "Withhold irrigation in Oct-Nov to induce physiological water stress (forces vegetative shoot to floral bud transition).",
                    "fertilizer": "Basal: 500g N : 250g P2O5 : 750g K2O per mature bearing tree.",
                    "pest_watch": "Stem borer and bark eating caterpillar.",
                    "notes": "Stress period is critical for flower panicle initiation."
                },
                {
                    "name": "Panicle Emergence & Flowering",
                    "pct_start": 25, "pct_end": 50,
                    "activities": ["Scout emerging panicles for mango hoppers and powdery mildew", "Keep honeybee hives in orchard to ensure cross-pollination", "Strictly avoid chemical sprays during 100% full bloom"],
                    "irrigation": "Resume light drip irrigation ONLY after fruit set (pea size stage).",
                    "fertilizer": "Foliar spray of 0.2% Boron (Solubor) + 1% Potassium Nitrate at panicle emergence.",
                    "pest_watch": "Mango hopper (*Amritodus atkinsoni*) - spray Thiamethoxam 25 WG @ 0.4 g/L before flowering opens; Powdery mildew (*Oidium mangiferae*).",
                    "notes": "High humidity or cloudy weather during flowering promotes powdery mildew."
                },
                {
                    "name": "Fruit Set, Marble Stage & Stone Hardening",
                    "pct_start": 50, "pct_end": 80,
                    "activities": ["Fruit thinning to remove misshapen clusters", "Spray 20 ppm NAA (Planofix @ 4.5 ml/10L) to prevent heavy fruit drop at marble stage", "Bagging individual premium fruits (Alphonso/Kesar) with paper bags"],
                    "irrigation": "Regular irrigation at 7-10 day intervals until 15 days before harvest.",
                    "fertilizer": "Second dose: 500g Urea + 250g SOP per tree around root drip line.",
                    "pest_watch": "Fruit fly (*Bactrocera dorsalis*) - install methyl eugenol pheromone traps @ 10/acre.",
                    "notes": "Stone hardening stage requires adequate soil moisture."
                },
                {
                    "name": "Fruit Maturity & Harvesting",
                    "pct_start": 80, "pct_end": 100,
                    "activities": ["Harvest when fruits develop characteristic shoulder expansion and olive green to yellowish color break", "Harvest with pole harvesters with 8-10 mm pedicel attached", "De-sapping on washing tables to prevent sap burning of skin"],
                    "irrigation": "Stop irrigation 12 days before harvest to improve shelf life and sweetness.",
                    "fertilizer": "None.",
                    "pest_watch": "Anthracnose and stem end rot during ripening.",
                    "notes": "Ripen fruits in ethylene chambers @ 100 ppm at 20-22°C."
                }
            ],
            "Grapes": [
                {
                    "name": "Foundation (April) Pruning & Cane Development",
                    "pct_start": 0, "pct_end": 30,
                    "activities": ["Back pruning to 1-2 basal buds in April", "Train new canes onto Bower / Y-trellis system", "Sub-cane pinching at 5-7 leaf stage to develop fruitful buds"],
                    "irrigation": "Regular drip irrigation matching ET demand.",
                    "fertilizer": "Apply 40% total N + 100% P + 25% K + FYM.",
                    "pest_watch": "Flea beetle and mealybugs on new shoots.",
                    "notes": "Good cane thickness (8-10 mm) is mandatory for floral bud fertility."
                },
                {
                    "name": "Forward (Oct) Pruning, Sprouting & Bloom",
                    "pct_start": 30, "pct_end": 60,
                    "activities": ["Forward pruning to 4-6 buds in October", "Apply Hydrogen Cyanamide (Dormex) @ 30-40 ml/L to cut buds for uniform bud break", "Panicle dipping in GA3 (10-15 ppm) for bunch elongation"],
                    "irrigation": "Light irrigation after Dormex application.",
                    "fertilizer": "Fertigate with 12:61:0 (MAP) + Phosphoric acid + Zinc EDTA.",
                    "pest_watch": "Downy mildew (oil spots) - spray Metalaxyl + Mancozeb; Thrips on flower caps.",
                    "notes": "Downy mildew is devastating if rain occurs during sprouting."
                },
                {
                    "name": "Berry Set, Thinning & Veraison (Color Break)",
                    "pct_start": 60, "pct_end": 85,
                    "activities": ["Berry thinning: Remove 50-60% berries per cluster for uniform berry size (target 18-20 mm)", "GA3 berry dipping (40 ppm) at 4mm and 6mm stage", "Girdling the main trunk to enhance sugar translocation"],
                    "irrigation": "Carefully regulated deficit irrigation to avoid berry splitting.",
                    "fertilizer": "Switch to 0:0:50 (SOP) + Calcium Nitrate + Micronutrients.",
                    "pest_watch": "Powdery mildew (powdery white coating on berries) and Mealybug cluster infesting.",
                    "notes": "Proper canopy thinning ensures dappled sunlight reaching bunches."
                },
                {
                    "name": "Ripening & Harvesting",
                    "pct_start": 85, "pct_end": 100,
                    "activities": ["Test TSS with refractometer (target 18-20° Brix and acidity 0.6%)", "Harvest bunches during cool morning hours using sharp scissors", "Pre-cooling to 2°C within 4 hours of harvest followed by SO2 sheet packaging"],
                    "irrigation": "Stop irrigation 8-10 days before harvesting.",
                    "fertilizer": "None.",
                    "pest_watch": "Berry rot and bird damage (use orchard nets).",
                    "notes": "Cold chain maintenance (0-1°C, 95% RH) gives 60 days storage."
                }
            ],
            "Pomegranate": [
                {
                    "name": "Bahar Treatment, Defoliation & Pruning",
                    "pct_start": 0, "pct_end": 20,
                    "activities": ["Stress induction: Stop irrigation for 40-50 days before Bahar (Hasth / Ambe / Mrig Bahar)", "Chemical defoliation with Ethrel @ 2-2.5 ml/L + 2% DAP", "Pruning dead twigs and water shoots followed by 1% Bordeaux paste on cuts"],
                    "irrigation": "Resume light irrigation after defoliation to trigger synchronized flowering.",
                    "fertilizer": "Basal: 25 kg FYM + 2 kg Neem cake + 250g DAP + 250g SOP per plant basin.",
                    "pest_watch": "Stem borer, shot hole borer, and bacterial blight (*Xanthomonas axonopodis* pv. *punicae*).",
                    "notes": "Synchronized flowering ensures single uniform harvest."
                },
                {
                    "name": "Flowering & Fruit Set",
                    "pct_start": 20, "pct_end": 45,
                    "activities": ["Pollination management: Conserve natural pollinators", "Thin hermaphrodite flower clusters to single vigorous fruit per spur", "Spray 0.2% Solubor (Boron) to reduce flower and pinhead drop"],
                    "irrigation": "Drip irrigation @ 15-20 litres/plant/day; avoid moisture fluctuation.",
                    "fertilizer": "Fertigation: 19:19:19 + Calcium Nitrate (5 kg/ha/week).",
                    "pest_watch": "Anar butterfly (*Virachola isocrates*) - spray Spinosad 45 SC @ 0.3 ml/L; Thrips.",
                    "notes": "Regular pruning of interior canopy gives light access."
                },
                {
                    "name": "Fruit Development, Bagging & Sizing",
                    "pct_start": 45, "pct_end": 80,
                    "activities": ["Bagging individual developing fruits with non-woven polypropylene or butter paper bags at 45 DAFS", "Spray Potassium Schoenite (1%) + Micronutrient mixture (Zn, Fe, Mn, B)", "Check for internal fruit cracking"],
                    "irrigation": "Uniform daily drip watering is critical to prevent fruit cracking.",
                    "fertilizer": "High Potassium fertigation: 0:0:50 (SOP) @ 6 kg/ha/week.",
                    "pest_watch": "Bacterial blight (oily spots on rind) - spray Streptocycline 0.5 g/L + Copper Oxychloride 2.5 g/L.",
                    "notes": "Fruit bagging prevents sunburn, thrips scratching, and borer entry."
                },
                {
                    "name": "Aril Color Development & Harvesting",
                    "pct_start": 80, "pct_end": 100,
                    "activities": ["Harvest when fruit calyx lobes curve inwards, skin turns glossy yellowish-red, and metallic sound upon tapping", "Clip fruits with secateurs without tearing bark", "Clean, grade by weight (>350g export grade), and pack in 4-kg boxes"],
                    "irrigation": "Maintain low soil moisture.",
                    "fertilizer": "None.",
                    "pest_watch": "Post-harvest fungal rot (*Aspergillus*, *Penicillium*).",
                    "notes": "Pomegranates are non-climacteric and must be harvested fully ripe."
                }
            ],
            "Apple": [
                {
                    "name": "Dormancy Break & Silver Tip to Pink Bud",
                    "pct_start": 0, "pct_end": 20,
                    "activities": ["Winter pruning of spur and lateral branches", "Tree basin clearing and application of 40 kg FYM per tree", "Horticultural Mineral Oil (HMO) 2% spray at silver tip to kill scale insects"],
                    "irrigation": "Irrigation during dormancy break if winter snowfall was deficit.",
                    "fertilizer": "Basal: Apply 50% N + 100% P2O5 + 50% K2O around canopy drip line.",
                    "pest_watch": "San Jose scale and Apple Scab (*Venturia inaequalis*) ascospore discharge.",
                    "notes": "Chilling requirement of 800-1200 hours below 7°C is essential for bud break."
                },
                {
                    "name": "Full Bloom & Fruit Set",
                    "pct_start": 20, "pct_end": 45,
                    "activities": ["Place 3-4 beehives per acre during 10-20% flowering", "Plant pollinizer varieties (e.g., Golden Delicious, Granny Smith) @ 20% ratio", "Spray Boric acid (0.1%) at pink bud stage to enhance pollen germination"],
                    "irrigation": "Light basin irrigation; avoid drought during flower opening.",
                    "fertilizer": "Foliar spray of 0.5% Urea + Zinc Sulphate (0.2%) after petal fall.",
                    "pest_watch": "Blossom thrips, European red mite, and Powdery mildew.",
                    "notes": "Strictly no insecticide spray during active bee foraging period."
                },
                {
                    "name": "Fruit Thinning & Sizing Phase (June Drop)",
                    "pct_start": 45, "pct_end": 80,
                    "activities": ["Hand thinning to retain single healthy 'King fruit' per spur (spacing 15-20 cm between fruits)", "Foliar spray of Calcium Chloride (0.3%) in 3 split sprays to prevent Bitter Pit disorder", "Summer pruning of water shoots to allow sunlight to lower canopy"],
                    "irrigation": "Regular irrigation at 10 day intervals during fruit cell expansion.",
                    "fertilizer": "Top-dress remaining 50% Nitrogen and Potassium.",
                    "pest_watch": "Woolly apple aphid, Codling moth, and Marssonina leaf blotch.",
                    "notes": "Calcium nutrition directly governs crispness and storage life."
                },
                {
                    "name": "Fruit Coloring, Starch Conversion & Harvest",
                    "pct_start": 80, "pct_end": 100,
                    "activities": ["Spread reflective silver mulch film on orchard floor 3 weeks before harvest for uniform red coloring", "Starch-iodine index test for picking maturity", "Careful two-stage hand picking with stalks intact"],
                    "irrigation": "Taper irrigation 10 days before harvest.",
                    "fertilizer": "None.",
                    "pest_watch": "Sooty blotch and flyspeck.",
                    "notes": "Pre-cooling to 4°C within 12 hours followed by Controlled Atmosphere (CA) storage (1°C, 1% O2, 1% CO2)."
                }
            ],
            "Soybean": [
                {
                    "name": "Sowing & Nodulation (VE - V2)",
                    "pct_start": 0, "pct_end": 20,
                    "activities": ["Broad bed furrow (BBF) or ridge-furrow planting at 45x5 cm spacing", "Seed inoculation with *Bradyrhizobium japonicum* + PSB (10g/kg)", "Sowing when soil is moist at 3-4 cm depth"],
                    "irrigation": "Pre-sowing irrigation; sensitive to waterlogging (BBF provides drainage).",
                    "fertilizer": "Basal: 25 kg N : 60 kg P2O5 : 40 kg K2O : 30 kg Sulphur/ha.",
                    "pest_watch": "Girdle beetle and stem fly on young seedlings.",
                    "notes": "Inoculation ensures heavy pink nodulation on tap roots."
                },
                {
                    "name": "Vegetative & Flowering (V3 - R2)",
                    "pct_start": 20, "pct_end": 50,
                    "activities": ["Inter-cultivation with wheel hoe or sweep at 20 & 35 DAS", "Monitor for yellow mosaic virus (whitefly vector)", "Scout for defoliating semilooper caterpillars"],
                    "irrigation": "Critical irrigation at flowering stage if monsoon dry spell exceeds 10 days.",
                    "fertilizer": "Foliar spray of 2% DAP or 19:19:19 (1%) at flower initiation.",
                    "pest_watch": "Tobacco caterpillar (*Spodoptera litura*) and Semilooper - spray Chlorantraniliprole 18.5 SC @ 0.3 ml/L.",
                    "notes": "Drought at flowering causes massive flower and pod abortion."
                },
                {
                    "name": "Pod Filling & Seed Development (R3 - R6)",
                    "pct_start": 50, "pct_end": 80,
                    "activities": ["Second critical moisture phase: provide life-saving protective irrigation", "Inspect for pod blight and charcoal rot", "Foliar spray of 1% Potassium Nitrate (13:0:45)"],
                    "irrigation": "Maintain adequate root zone moisture during seed filling.",
                    "fertilizer": "Sulphur applied at basal stage boosts oil content (18-20%) and protein (40%).",
                    "pest_watch": "Pod borer and Rust (*Phakopsora pachyrhizi*) - spray Hexaconazole 5% EC @ 1 ml/L.",
                    "notes": "Seed weight (100-seed weight) is determined in this stage."
                },
                {
                    "name": "Maturity & Harvesting (R7 - R8)",
                    "pct_start": 80, "pct_end": 100,
                    "activities": ["Harvest when 95% leaves turn yellow and drop off, and pods turn greyish-brown", "Harvest when grain moisture is 14-16% to avoid pod shattering", "Thresh at low cylinder speed (400-500 rpm) to avoid cracking seed coat"],
                    "irrigation": "No irrigation.",
                    "fertilizer": "None.",
                    "pest_watch": "Storage bruchid beetle.",
                    "notes": "Store seed soybean below 10% moisture to maintain germination viability."
                }
            ],
            "Mustard": [
                {
                    "name": "Sowing & Rosette Stage",
                    "pct_start": 0, "pct_end": 22,
                    "activities": ["Seedbed preparation with fine tilth and preserved moisture", "Seed treatment with *Trichoderma* (10g/kg) + Apron 35 SD (for white rust)", "Line sowing at 30x10 cm with thinning at 15-20 DAS"],
                    "irrigation": "First irrigation at 25-30 DAS (Rosette stage) - very critical.",
                    "fertilizer": "Basal: 40 kg N + 40 kg P2O5 + 20 kg K2O + 40 kg Sulphur (Bentonite-S) per hectare.",
                    "pest_watch": "Sawfly larvae and Painted bug on young cotyledons.",
                    "notes": "Thinning to single plant per hill ensures vigorous branching."
                },
                {
                    "name": "Stem Elongation & Flowering",
                    "pct_start": 22, "pct_end": 55,
                    "activities": ["Second top-dressing of 40 kg N (Urea) before second irrigation", "Scout for mustard aphid colonies on inflorescence branches", "Conserve honeybees for cross-pollination"],
                    "irrigation": "Second irrigation at flowering stage (50-55 DAS).",
                    "fertilizer": "Top-dress remaining Nitrogen. Sulphur is vital for glucosinolate & oil synthesis.",
                    "pest_watch": "Mustard Aphid (*Lipaphis erysimi*) - spray Dimethoate 30 EC @ 1.5 ml/L or Thiamethoxam 25 WG @ 0.4 g/L at ETL (25 aphids/10cm central shoot).",
                    "notes": "Aphids suck sap and cause curling and stunted pod formation."
                },
                {
                    "name": "Pod Formation (Siliqua) & Seed Filling",
                    "pct_start": 55, "pct_end": 85,
                    "activities": ["Third light irrigation at early siliqua development stage (75 DAS)", "Inspect for White Rust (white pustules on leaves) and Alternaria blight (concentric leaf spots)", "Foliar spray of 1% Potassium Nitrate"],
                    "irrigation": "Light irrigation on calm windless days to prevent crop lodging.",
                    "fertilizer": "Foliar spray of Mancozeb 75 WP (2 g/L) against Alternaria blight.",
                    "pest_watch": "White rust (*Albugo candida*) and Sclerotinia stem rot.",
                    "notes": "Frost risk during December-January can be mitigated by light evening irrigation."
                },
                {
                    "name": "Maturity & Harvesting",
                    "pct_start": 85, "pct_end": 100,
                    "activities": ["Harvest when 75-80% pods (siliquae) turn golden yellow and seeds rattle inside", "Harvest during early morning hours to minimize pod shattering losses", "Bundle sheaves, dry in sun for 5-7 days, and thresh"],
                    "irrigation": "Withhold all water 20 days prior to harvest.",
                    "fertilizer": "None.",
                    "pest_watch": "Protect harvested threshing floor from dew and birds.",
                    "notes": "Clean and dry seeds to 8% moisture before oil milling."
                }
            ],
            "Watermelon": [
                {
                    "name": "Raised Bed Sowing & Nursery",
                    "pct_start": 0, "pct_end": 20,
                    "activities": ["Form raised beds of 1.2 m width with 2.5 m channel spacing", "Lay 25-micron silver-black plastic mulch and punch holes at 60 cm spacing", "Dibble 2 seeds/hole at 2.5 cm depth"],
                    "irrigation": "Initial drip wetting of beds before sowing.",
                    "fertilizer": "Basal: 10 tonnes FYM + 50 kg DAP + 50 kg SOP per hectare.",
                    "pest_watch": "Red pumpkin beetle on emerging cotyledon leaves.",
                    "notes": "Mulching conserves soil moisture and prevents fruit rotting on bare soil."
                },
                {
                    "name": "Vine Extension & Secondary Branching",
                    "pct_start": 20, "pct_end": 45,
                    "activities": ["Train vines towards bed edges across mulch", "Nipping main vine tip at 6-8 leaf stage to trigger vigorous fruitful lateral branches", "Weeding furrow spaces"],
                    "irrigation": "Daily drip irrigation (1-2 hours) as canopy expands.",
                    "fertilizer": "Fertigation: 19:19:19 @ 4 kg/ha every 3rd day + Magnesium Sulphate (2 kg/ha).",
                    "pest_watch": "Thrips and Cucumber mosaic virus (aphid vector).",
                    "notes": "Lateral branches produce higher proportion of female flowers."
                },
                {
                    "name": "Flowering, Fruit Setting & Sizing",
                    "pct_start": 45, "pct_end": 75,
                    "activities": ["Bee activity or hand pollination in early morning hours (6-9 AM)", "Retain only 2-3 uniform fruits per vine and thin out deformed ones", "Place dried straw under fruits resting near furrows"],
                    "irrigation": "Maintain steady uniform soil moisture. Avoid sudden heavy watering which cracks fruits.",
                    "fertilizer": "Fertigation: 0:52:34 (Mono Potassium Phosphate) + 13:0:45 (Potassium Nitrate) @ 5 kg/ha.",
                    "pest_watch": "Fruit fly (*Bactrocera cucurbitae*) - install Cue-lure pheromone traps @ 10/acre; Powdery mildew.",
                    "notes": "High potassium promotes deep red aril color and high TSS sugar."
                },
                {
                    "name": "Sugar Accumulation & Harvest",
                    "pct_start": 75, "pct_end": 100,
                    "activities": ["Check harvest indicators: Ground spot turns creamy yellow, tendril opposite fruit stem dries completely, and dull thud sound on tapping", "Cut fruits with 3 cm stem intact using sharp knife", "Stack gently in transport vehicles with straw cushioning"],
                    "irrigation": "Completely stop irrigation 5-7 days before harvest to maximize Brix sugar levels.",
                    "fertilizer": "None.",
                    "pest_watch": "Fruit rot during post-harvest transit.",
                    "notes": "Harvest in cool morning hours; never expose cut fruit to direct sun."
                }
            ]
        }

        # Add fallback mappings for allied pulses, vegetables, and fruit crops
        self.stage_templates["Muskmelon"] = self.stage_templates["Watermelon"]
        self.stage_templates["Groundnuts"] = self.stage_templates["Groundnut"]
        self.stage_templates["Blackgram"] = self.stage_templates["Chickpea"]
        self.stage_templates["Mungbean"] = self.stage_templates["Chickpea"]
        self.stage_templates["Pigeonpeas"] = self.stage_templates["Chickpea"]
        self.stage_templates["Kidneybeans"] = self.stage_templates["Chickpea"]
        self.stage_templates["Lentil"] = self.stage_templates["Chickpea"]
        self.stage_templates["Mothbeans"] = self.stage_templates["Chickpea"]
        self.stage_templates["Pulses"] = self.stage_templates["Chickpea"]
        self.stage_templates["Orange"] = self.stage_templates["Mango"]
        self.stage_templates["Papaya"] = self.stage_templates["Banana"]
        self.stage_templates["Coconut"] = self.stage_templates["Sugarcane"]
        self.stage_templates["Coffee"] = self.stage_templates["Mango"]
        self.stage_templates["Jute"] = self.stage_templates["Rice"]
        self.stage_templates["Barley"] = self.stage_templates["Wheat"]
        self.stage_templates["Millets"] = self.stage_templates["Maize"]
        self.stage_templates["Sorghum"] = self.stage_templates["Maize"]

        self.crop_durations = {
            "Rice": 120, "Wheat": 130, "Barley": 115, "Cotton": 160,
            "Maize": 105, "Sugarcane": 360, "Groundnut": 110, "Groundnuts": 110,
            "Millets": 90, "Sorghum": 110, "Pomegranate": 210, "Pulses": 95,
            "Chickpea": 100, "Kidneybeans": 90, "Pigeonpeas": 160, "Mothbeans": 80,
            "Mungbean": 75, "Blackgram": 80, "Lentil": 110, "Coffee": 270,
            "Jute": 125, "Coconut": 365, "Apple": 180, "Orange": 240,
            "Papaya": 270, "Banana": 330, "Mango": 150, "Grapes": 140,
            "Watermelon": 85, "Muskmelon": 80, "Tomato": 115, "Potato": 100,
            "Mustard": 110, "Soybean": 105
        }

    def generate_timeline(self, crop_name: str, sowing_date: str = None, sowing_date_str: str = None, soil_type: str = "Alluvial", location: str = "Farm", **kwargs) -> Dict[str, Any]:
        crop_clean = crop_name.strip().title()
        date_input = sowing_date or sowing_date_str or datetime.now().strftime("%Y-%m-%d")
        
        try:
            sowing_date_dt = datetime.strptime(str(date_input), "%Y-%m-%d")
        except Exception:
            sowing_date_dt = datetime.now()

        total_days = self.crop_durations.get(crop_clean, 120)
        harvest_date = sowing_date_dt + timedelta(days=total_days)

        current_day = max(0, (datetime.now() - sowing_date_dt).days)
        
        # Select appropriate stage template
        template = self.stage_templates.get(crop_clean, self.stage_templates.get("Rice"))
        
        stages: List[Dict[str, Any]] = []
        current_stage_name = template[0]["name"] if template else "Preparation / Sowing"
        notifications = []

        for idx, t in enumerate(template, 1):
            start_d = int((t["pct_start"] / 100.0) * total_days)
            end_d = int((t["pct_end"] / 100.0) * total_days)

            # Calculate real calendar dates for every stage
            stage_start_date = (sowing_date_dt + timedelta(days=start_d)).strftime("%Y-%m-%d")
            stage_end_date = (sowing_date_dt + timedelta(days=end_d)).strftime("%Y-%m-%d")

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

            stages.append({
                "stage_id": idx,
                "stage_name": t["name"],
                "start_day": start_d,
                "end_day": end_d,
                "start_date": stage_start_date,
                "end_date": stage_end_date,
                "status": status,
                "activities": t["activities"],
                "irrigation_schedule": t["irrigation"],
                "fertilizer_advice": t["fertilizer"],
                "pest_disease_watch": t["pest_watch"],
                "critical_notes": f"Scheduled: {stage_start_date} to {stage_end_date}. {t['notes']}"
            })

        return {
            "crop_name": crop_clean,
            "sowing_date": sowing_date_dt.strftime("%Y-%m-%d"),
            "expected_harvest_date": harvest_date.strftime("%Y-%m-%d"),
            "estimated_harvest_date": harvest_date.strftime("%Y-%m-%d"),
            "total_duration_days": total_days,
            "current_day": min(total_days, current_day),
            "current_stage": current_stage_name,
            "stages": stages,
            "active_notifications": notifications
        }

timeline_service = TimelineService()
