import os

KNOWLEDGE_DIR = "backend/data/knowledge"

guides = {
    "crops/chilli.md": """# Chilli (Hot Pepper) Complete Cultivation & IPM Guide
*Authoritative Reference: ICAR-IIHR & AgricultureGuruji Protocols*

## 1. Agro-Climatic Requirements & Soil
- **Climate:** Tropical to subtropical warm climate. Optimal daytime temperature: 24°C - 32°C; night temperature: 18°C - 22°C. High temperature with dry winds triggers flower and pin-head bud drop.
- **Soil:** Well-drained sandy loam, clay loam, or black vertisol rich in organic matter with pH 6.0 - 7.5. Avoid waterlogged or saline soils (EC < 1.0 dS/m).

## 2. Nursery Raising & Transplanting
- **Seed Rate:** 150 - 200 g/acre for high-yielding hybrids (Teja, Byadgi, G4, Armoor).
- **Pro-Tray Nursery:** Raise in 98-cell pro-trays with sterilized coco-peat + vermicompost + Trichoderma viride. Ready for transplanting at 30-35 days (4-5 true leaves).
- **Spacing:** Raised beds with 25-micron silver-black mulch. Spacing: 120 cm row-to-row, 45 cm plant-to-plant (approx. 7,400 plants/acre).

## 3. Fertilizer & Fertigation Regimen (kg/acre)
- **Basal Soil Dose:** FYM: 8-10 tons, SSP: 100 kg, MOP: 25 kg, Neem cake: 150 kg, Zinc Sulphate (21%): 10 kg, Borax: 4 kg.
- **Drip Fertigation Schedule:**
  - *Vegetative (Days 15 to 45):* 19:19:19 @ 3.0 kg/acre every 4 days + 12:61:0 MAP @ 2.0 kg/acre weekly.
  - *Flowering & Fruit Set (Days 46 to 90):* 0:52:34 MKP @ 3.5 kg/acre twice a week + Calcium Nitrate @ 2.5 kg/acre weekly.
  - *Fruit Sizing & Harvest Phase (Days 91 to 180):* 13:0:45 Potassium Nitrate @ 4.0 kg/acre every 4 days + 0:0:50 SOP @ 2.0 kg/acre for pungency (Capsaicin), deep red color, and fruit weight.

## 4. Pest & Disease Management (IPM)
- **Chilli Thrips (*Scirtothrips dorsalis* - Upward Leaf Curling):** Spinetoram 11.7% SC @ 1.0 ml/L or Fipronil 5% SC @ 2.0 ml/L. Install 20 blue sticky traps/acre.
- **Yellow Mite (*Polyphagotarsonemus latus* - Downward Leaf Curling):** Propargite 57% EC @ 2.5 ml/L or Diafenthiuron 50% WP @ 1.2 g/L.
- **Die-Back & Anthracnose Fruit Rot (*Colletotrichum capsici*):** Azoxystrobin 18.2% + Difenoconazole 11.4% SC @ 1.0 ml/L or Copper Oxychloride @ 3.0 g/L.
""",

    "crops/onion_and_garlic.md": """# Onion & Garlic Precision Production Guide
*Authoritative Reference: ICAR-DOGR Directives*

## 1. Varieties & Seasons
- **Kharif Onion:** Agrifound Dark Red, Baswant 780, Bhima Super. Transplanting: July-August; Harvest: November-December.
- **Rabi Onion:** Bhima Kiran, Bhima Shakti, N-2-4-1, Pusa Red. Transplanting: December-January; Harvest: April-May.
- **Garlic:** Yamuna Safed (G-1), G-50, Bhima Omkar. Sowing: October-November.

## 2. Soil & Nutrient Architecture
- **Soil:** Light sandy loam to clay loam with high organic carbon (> 0.8%) and pH 6.0 - 7.5.
- **NPK Requirements (per acre):** 50 kg N : 25 kg P2O5 : 35 kg K2O.
- **Sulphur is Mandatory:** Apply elemental Sulphur or Bentonite Sulphur @ 15 kg/acre at transplanting. Sulphur enhances bulb pungency (Allyl Propyl Disulphide), outer scale color, and extends storage life by 60 days.

## 3. Irrigation & Bulb Curing
- **Critical Irrigation Stages:** Bulb initiation (35-45 DAT) and bulb enlargement (60-80 DAT). Stop irrigation 10-15 days prior to harvest to prevent neck rot.
- **Field Curing:** After harvesting, windrow bulbs with foliage covering lower bulbs for 3-5 days in field, then shade cure for 10-12 days until neck is fully dry and papery.
""",

    "crops/turmeric_and_ginger.md": """# Turmeric & Ginger High-Yield Production Guide
*Authoritative Reference: ICAR-IISR (Indian Institute of Spices Research)*

## 1. Soil & Land Preparation
- **Soil:** Deep, well-drained loamy, alluvial or red soils with rich humus and pH 5.5 - 7.2. Sensitive to water stagnation (rhizome rot).
- **Raised Bed Method:** 120 cm bed width, 30 cm height, 40 cm furrow. Incorporate 12 tons FYM + 200 kg Neem cake + 2 kg Trichoderma per acre.

## 2. Seed Rhizome Treatment
- Treat healthy, bold mother rhizomes (30-40 g) with Mancozeb 75% WP @ 3 g/L + Imidacloprid 17.8% SL @ 0.5 ml/L for 30 minutes to prevent Pythium soft rot and scale insects.

## 3. Organic Mulching & Nutrition
- **Mulching:** Apply green leaf mulch @ 5 tons/acre immediately after planting, followed by 3 tons/acre at 45 and 90 days. Mulching conserves moisture, adds 2% organic matter, and keeps root-zone cool.
- **Nutrient Dose:** 60 kg N : 30 kg P2O5 : 60 kg K2O per acre. Apply 50% Potash at 90 days for rhizome bulk and high curcumin content (> 4.5%).
""",

    "organic_farming/natural_farming_jeevamrutham.md": """# Subhash Palekar Natural Farming (SPNF) & Bio-Formulations Guide
*Authoritative Reference: National Centre of Organic Farming (NCOF)*

## 1. Liquid Jeevamrutham (Microbial Bio-Inoculant)
- **Ingredients for 200 Litres (1 Acre):**
  - Desi Cow Dung (Fresh): 10 kg
  - Desi Cow Urine: 5 - 10 Litres
  - Organic Jaggery: 2 kg
  - Pulse Flour (Besan): 2 kg
  - Virgin Forest / Farm Boundary Soil: 1 handful (contains indigenous beneficial microbes)
  - Water: 200 Litres
- **Preparation:** Mix in shade, stir clockwise with wooden stick twice daily for 4-7 days.
- **Application:** Apply 200 L/acre with irrigation water or drip every 15 days. Rapidly multiplies rhizosphere bacterial and mycorrhizal populations.

## 2. Beejamrutham (Seed Treatment)
- 5 kg fresh cow dung + 5 L cow urine + 50 g slaked lime (Chuna) + 20 L water. Soak seeds for 15 minutes before sowing. Protects seedlings from damping-off and soil-borne pathogens.

## 3. Ghanajeevamrutham (Solid Bio-Fertilizer)
- 100 kg dried cow dung + 2 kg jaggery + 2 kg besan + 5 L cow urine. Dry under shade, pulverize and store. Apply 200 kg/acre during field preparation.

## 4. Dashaparni Kashayam (Organic Insecticide)
- Extract of 10 bitter/medicinal leaves (Neem, Karanj, Custard apple, Papaya, Castor, Datura, Calotropis, Guava, Lantana, Nerium) fermented in cow dung and urine for 30 days. Dilute 200 ml per 10 L water for broad-spectrum sucking pest and caterpillar knockdown.
""",

    "pest_and_disease/sucking_pests_management.md": """# Comprehensive Sucking Pest Diagnosis & IPM Protocol
*Authoritative Reference: ICAR-NCIPM (National Centre for Integrated Pest Management)*

## 1. Major Sucking Pests & Economic Threshold Levels (ETL)
- **Whiteflies (*Bemisia tabaci*):** ETL: 5-8 adults/leaf. Vector for Leaf Curl Virus.
- **Thrips (*Thrips tabaci*, *Scirtothrips*):** ETL: 5 thrips/leaf. Causes upward leaf curling, silvery sheen, and scar tissue on fruits.
- **Aphids (*Aphis gossypii*):** ETL: 10-15% infested plants. Excrete sticky honeydew causing black sooty mold (*Capnodium*).
- **Jassids / Leafhoppers (*Amrasca biguttula*):** ETL: 2-3 nymphs/leaf. Causes yellowing margins ("hopper burn").

## 2. Mechanical & Cultural Control
- **Sticky Traps:** Install 15 Yellow Sticky Traps/acre for whiteflies & aphids; 15 Blue Sticky Traps/acre for thrips.
- **Barrier Crops:** Plant 2-3 border rows of Maize, Sorghum, or Pearl Millet around fields to block wind-borne insect vectors.
- **Reflective Mulch:** 25-micron silver-black plastic mulch repels 85% of incoming winged thrips and aphids.

## 3. Biological & Chemical Interventions
- **Stage 1 (Bio-Agents):** Spray Neem Oil (10,000 ppm) @ 2 ml/L or *Verticillium lecanii* / *Beauveria bassiana* @ 5 g/L during early infestation.
- **Stage 2 (Targeted Chemistry):**
  - For Whiteflies/Jassids: Acetamiprid 20% SP @ 0.5 g/L or Dinotefuran 20% SG @ 0.4 g/L.
  - For Thrips: Spinetoram 11.7% SC @ 0.9 ml/L or Fipronil 5% SC @ 2.0 ml/L.
  - For Resistant Mixed Infestations: Pyriproxyfen 10% + Bifenthrin 10% EC @ 2 ml/L.
""",

    "fertilizers/micronutrient_deficiency_guide.md": """# Soil & Foliar Micronutrient Deficiency Diagnosis & Correction
*Authoritative Reference: ICAR-IISS (Indian Institute of Soil Science)*

## 1. Key Micronutrient Deficiencies & Diagnostic Symptoms
- **Zinc (Zn) Deficiency:**
  - *Symptoms:* "Khaira disease" in rice (rusty brown spots on lower leaves), "White bud" in maize, interveinal chlorosis in younger leaves of cotton and vegetables.
  - *Soil Correction:* Zinc Sulphate (21% Heptahydrate) @ 10-12 kg/acre basal.
  - *Foliar Emergency Spray:* Chelated Zinc (Zn-EDTA 12%) @ 1.0 g/L or Zinc Sulphate @ 5 g/L + 2.5 g Slaked Lime.
- **Boron (B) Deficiency:**
  - *Symptoms:* Hollow heart in groundnut, blossom end rot and fruit cracking in tomato/pomegranate, deformed fruit tips.
  - *Foliar Correction:* Solubor (Boron 20%) @ 1.0 - 1.5 g/L sprayed during pre-flowering and fruit set.
- **Iron (Fe) Deficiency:**
  - *Symptoms:* Complete whitening / ivory yellowing of new emerging leaves while veins remain green (common in alkaline soils pH > 7.8).
  - *Foliar Correction:* Ferrous Sulphate (FeSO4 19%) @ 5 g/L + Citric Acid @ 1 g/L or Fe-EDDHA (6%) @ 1.5 g/L.
- **Magnesium (Mg) Deficiency:**
  - *Symptoms:* Reddening / purple discoloration of leaves between veins in cotton and potato.
  - *Correction:* Magnesium Sulphate (9.6% Mg) @ 15 kg/acre basal or 5 g/L foliar.

## 2. Soil Reclamation Protocols
- **Alkaline / Sodic Soil (pH > 8.2, ESP > 15%):** Broadcast agricultural Gypsum (CaSO4·2H2O) @ 2 - 3 metric tons/acre, flood field with water for 48 hours and drain out displaced sodium salts.
- **Acidic Soil (pH < 5.5):** Broadcast Agricultural Limestone (CaCO3) or Dolomite @ 1 - 1.5 tons/acre 30 days prior to sowing.
""",

    "market_and_schemes/government_subsidies_and_schemes.md": """# Complete Guide to Indian Agricultural Schemes & Direct Farmer Subsidies
*Authoritative Reference: Ministry of Agriculture & Farmers Welfare, Govt of India*

## 1. PM-Kisan Samman Nidhi
- **Direct Income Support:** ₹6,000 per year transferred directly to farmer bank accounts in 3 equal four-monthly installments of ₹2,000 each.
- **Eligibility:** All landholding farmer families across India. Aadhaar eKYC mandatory on pmkisan.gov.in.

## 2. PM Krishi Sinchayee Yojana (PMKSY - Per Drop More Crop)
- **Drip & Sprinkler Irrigation Subsidy:**
  - Small & Marginal Farmers (< 5 acres): 55% subsidy on total installation cost.
  - Other Farmers (> 5 acres): 45% subsidy.
  - Certain states (Telangana, Andhra, Tamil Nadu, Gujarat) provide additional top-up up to 70-90% subsidy for SC/ST farmers.

## 3. Sub-Mission on Agricultural Mechanization (SMAM)
- **Equipment Subsidy:** 40% to 50% subsidy on purchase of Tractors, Power Tillers, Rotavators, Power Weeders, Happy Seeders, Drone Spraying units, and Laser Levelers.
- **Custom Hiring Centres (CHC):** Up to 40% financial assistance (max ₹10 Lakhs) to establish farm equipment custom hiring hubs.

## 4. PM Fasal Bima Yojana (PMFBY - Crop Insurance)
- **Premium Rates:** 2.0% for Kharif crops, 1.5% for Rabi foodgrains & oilseeds, and 5.0% for annual commercial/horticultural crops. Balance premium paid equally by Central and State Governments.
- **Claim Triggers:** Prevented sowing, mid-season adversity, localized natural calamities (hailstorm, cloudburst), post-harvest cyclone damage within 14 days of cutting.
""",

    "protected_cultivation/hydroponics_nft_dwc.md": """# Commercial Hydroponics & Soilless Cultivation Handbook
*Authoritative Reference: AgricultureGuruji & ICAR-CIAH Protocols*

## 1. Hydroponic Systems Comparison
- **Nutrient Film Technique (NFT):** Thin film of nutrient solution continuously pumped through food-grade PVC/UPVC gullies (slope 1:40). Best suited for leafy greens (Lettuce, Spinach, Basil, Mint, Kale) and Strawberries.
- **Deep Water Culture (DWC):** Plant roots submerged in oxygenated reservoir with continuous air-stone bubbler (Dissolved Oxygen > 6.0 ppm).
- **Dutch Bucket / Drip System:** Media-based (Coco-peat + Perlite 70:30) with individual drippers. Best suited for heavy vine crops (Tomato, Cucumber, Capsicum, Melon).

## 2. Water Quality & Nutrient Solution Parameters
- **pH Range:** 5.8 - 6.2 (optimal nutrient bio-availability).
- **Electrical Conductivity (EC):**
  - Leafy greens (Lettuce, Herbs): 1.2 - 1.6 dS/m
  - Strawberries: 1.0 - 1.4 dS/m
  - Tomato / Capsicum: 2.0 - 2.8 dS/m
- **Water Temperature:** Keep between 18°C - 22°C. Water temperature > 26°C causes root hypoxia and Pythium root rot.

## 3. Standard 2-Part Hydroponic Nutrient Recipe (per 1000 L RO Water)
- **Stock Tank A (Calcium & Iron):**
  - Calcium Nitrate: 800 g
  - Potassium Nitrate (13:0:45): 200 g
  - Fe-EDDHA / Fe-DTPA (6% Iron): 25 g
- **Stock Tank B (Phosphorus, Potassium, Sulphates & Micronutrients):**
  - Potassium Dihydrogen Phosphate (0:52:34 MKP): 250 g
  - Potassium Nitrate: 300 g
  - Magnesium Sulphate: 400 g
  - Micronutrient Mix (Zn, Mn, B, Cu, Mo): 30 g
"""
}

for rel_path, content in guides.items():
    full_path = os.path.join(KNOWLEDGE_DIR, rel_path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(content.strip() + "\n")
    print(f"Created: {full_path}")

print("All massive knowledge guides populated successfully!")
