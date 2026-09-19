import os
import io
import datetime
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

# Import Backend AI & Agronomic Services
from backend.app.services.soil_service import soil_service
from backend.app.services.crop_service import crop_service
from backend.app.services.fertilizer_service import fertilizer_service
from backend.app.services.timeline_service import timeline_service
from backend.app.services.notification_service import notification_service
from backend.app.services.rag_assistant_service import rag_assistant_service
from backend.app.services.weather_service import weather_service
from backend.app.schemas.models import ManualSoilProperties, CropRecommendationRequest, FertilizerRecommendationRequest

# -------------------------------------------------------------
# 1. Page Configuration & Custom CSS Styling
# -------------------------------------------------------------
st.set_page_config(
    page_title="Agri-AI Multimodal Decision System",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Main Theme & Typography */
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #14532d;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    
    /* Card Component */
    .agri-card {
        background: #ffffff;
        padding: 1.25rem 1.5rem;
        border-radius: 14px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.25rem;
    }
    
    .agri-card-green {
        background: #f0fdf4;
        border: 1.5px solid #86efac;
        padding: 1.25rem;
        border-radius: 12px;
        margin-bottom: 1rem;
    }
    
    .agri-card-alert {
        background: #fef2f2;
        border: 1.5px solid #fca5a5;
        padding: 1.25rem;
        border-radius: 12px;
        margin-bottom: 1rem;
    }
    
    /* Status Badges */
    .badge-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 700;
    }
    .badge-green { background: #dcfce7; color: #166534; }
    .badge-blue { background: #dbeafe; color: #1e40af; }
    .badge-amber { background: #fef3c7; color: #92400e; }
    .badge-red { background: #fee2e2; color: #991b1b; }
    
    /* Metric Card Grid */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
        gap: 0.75rem;
        margin-top: 0.5rem;
    }
    .metric-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.75rem;
        text-align: center;
    }
    .metric-box-title { font-size: 0.75rem; color: #64748b; font-weight: 600; text-transform: uppercase; }
    .metric-box-val { font-size: 1.15rem; font-weight: 700; color: #1e293b; margin-top: 2px; }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 2. Session State Initialization
# -------------------------------------------------------------
if "city" not in st.session_state:
    st.session_state.city = "Hyderabad"
if "soil_data" not in st.session_state:
    st.session_state.soil_data = {
        "nitrogen": 90.0,
        "phosphorus": 42.0,
        "potassium": 43.0,
        "ph": 6.5,
        "soil_type": "Black",
        "moisture": 45.0,
        "zinc": 1.2,
        "sulphur": 15.0,
        "electrical_conductivity": 0.8
    }
if "cnn_result" not in st.session_state:
    st.session_state.cnn_result = None
if "recommendations" not in st.session_state:
    st.session_state.recommendations = []
if "selected_crop" not in st.session_state:
    st.session_state.selected_crop = "Rice"
if "sowing_date" not in st.session_state:
    st.session_state.sowing_date = datetime.date.today()
if "timeline_data" not in st.session_state:
    st.session_state.timeline_data = None
if "fertilizer_data" not in st.session_state:
    st.session_state.fertilizer_data = None
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {
            "role": "assistant",
            "content": "👋 Hello! I am your AI Agricultural Assistant. Ask me anything about crop planning, stage-specific fertilizer doses, pest management, or current weather impacts."
        }
    ]

# -------------------------------------------------------------
# 3. Sidebar: Live Open-Meteo Weather & System Telemetry
# -------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/plant-under-rain.png", width=64)
    st.title("🌾 Farm Intelligence")
    
    st.subheader("📍 Real-Time Location & Weather")
    city_input = st.text_input("Enter Farm Location / City:", value=st.session_state.city)
    if city_input != st.session_state.city:
        st.session_state.city = city_input
        st.rerun()

    # Fetch 100% Live Open-Meteo Satellite Weather
    with st.spinner("Fetching live meteorological feed..."):
        weather = weather_service.get_weather(st.session_state.city)

    st.markdown(f"""
    <div class="agri-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <b style="font-size: 1rem; color: #1e293b;">{weather.get('city', st.session_state.city)}</b>
            <span class="badge-pill badge-green">● Live Satellite</span>
        </div>
        <div style="font-size: 0.85rem; color: #15803d; font-weight: 600; margin-bottom: 8px;">
            🌤️ {weather.get('weather_condition', 'Clear Sky')} (Feels like {weather.get('feels_like', weather.get('temperature', 28.0))}°C)
        </div>
        <div class="metric-grid">
            <div class="metric-box">
                <div class="metric-box-title">Temp</div>
                <div class="metric-box-val">{weather.get('temperature', 28.0)}°C</div>
            </div>
            <div class="metric-box">
                <div class="metric-box-title">Humidity</div>
                <div class="metric-box-val">{weather.get('humidity', 65.0)}%</div>
            </div>
            <div class="metric-box">
                <div class="metric-box-title">Rain (24h)</div>
                <div class="metric-box-val">{weather.get('rainfall', 0.0)} mm</div>
            </div>
            <div class="metric-box">
                <div class="metric-box-title">Pressure</div>
                <div class="metric-box-val">{weather.get('pressure', 1012.0)} hPa</div>
            </div>
            <div class="metric-box">
                <div class="metric-box-title">Wind</div>
                <div class="metric-box-val">{weather.get('wind_speed', 5.0)} km/h</div>
            </div>
            <div class="metric-box">
                <div class="metric-box-title">Rain Chance</div>
                <div class="metric-box-val">{weather.get('rain_probability', 0)}%</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption(f"🤖 **RAG Knowledge Base:** 59 Guides (306 Chunks)")
    st.caption(f"🧠 **Crop ML Model:** XGBoost + LightGBM (30 Crops)")
    st.caption(f"👁️ **Vision Model:** 133 Features + OOD Domain Guard")
    st.caption(f"🛰️ **Weather Model:** Open-Meteo Global High-Res")

# -------------------------------------------------------------
# 4. Main Dashboard Header
# -------------------------------------------------------------
st.markdown('<div class="main-header">🌱 Adaptive Agro-AI Decision & Crop Intelligence System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Multimodal Generative AI for Out-of-Distribution Soil Verification, Precision Crop Planning & Proactive Farm Intelligence</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📷 1. Soil Vision & Lab Test",
    "🌾 2. Crop Recommendation",
    "🧪 3. Fertilizer Deficit Plan",
    "📅 4. Lifecycle Calendar",
    "🔔 5. Proactive Alerts",
    "🤖 6. AI Agronomist (RAG)"
])

# =============================================================
# TAB 1: Soil Vision (CNN/Ensemble + OOD Detector) & Manual Lab Test
# =============================================================
with tab1:
    col_v1, col_v2 = st.columns([1, 1], gap="large")
    
    with col_v1:
        st.subheader("📷 Live Soil Photo Verification")
        st.write("Upload a field soil photo. The AI vision model extracts 133 multi-spectral color and spatial texture features, and runs a strict Out-of-Distribution (OOD) detector to reject non-soil photos (posters, portraits, solid graphics).")
        
        uploaded_file = st.file_uploader("Upload Soil Photo (JPG, PNG, WEBP):", type=["jpg", "jpeg", "png", "webp"])
        
        if uploaded_file is not None:
            image_bytes = uploaded_file.read()
            image = Image.open(io.BytesIO(image_bytes))
            st.image(image, caption="Uploaded Image Preview", use_container_width=True)
            
            if st.button("🔍 Analyze Soil Image", type="primary", use_container_width=True):
                with st.spinner("Extracting 133 visual features & verifying domain distribution..."):
                    result = soil_service.analyze_soil_image(image_bytes)
                    st.session_state.cnn_result = result
                    
                    if result["is_valid_soil"] and result["detected_soil_type"]:
                        clean_type = result["detected_soil_type"].replace(" Soil", "").strip()
                        st.session_state.soil_data["soil_type"] = clean_type
                        
                        # Real agronomic chemical baselines by soil classification
                        soil_baselines = {
                            "Black": {"ph": 7.8, "nitrogen": 90.0, "phosphorus": 42.0, "potassium": 48.0, "moisture": 45.0, "zinc": 1.2, "sulphur": 15.0, "electrical_conductivity": 0.8},
                            "Alluvial": {"ph": 7.2, "nitrogen": 110.0, "phosphorus": 52.0, "potassium": 55.0, "moisture": 40.0, "zinc": 1.5, "sulphur": 18.0, "electrical_conductivity": 0.6},
                            "Red": {"ph": 6.2, "nitrogen": 75.0, "phosphorus": 35.0, "potassium": 50.0, "moisture": 30.0, "zinc": 0.9, "sulphur": 12.0, "electrical_conductivity": 0.4},
                            "Laterite": {"ph": 5.2, "nitrogen": 55.0, "phosphorus": 22.0, "potassium": 35.0, "moisture": 28.0, "zinc": 0.6, "sulphur": 8.0, "electrical_conductivity": 0.3},
                            "Arid": {"ph": 8.4, "nitrogen": 40.0, "phosphorus": 18.0, "potassium": 65.0, "moisture": 15.0, "zinc": 0.5, "sulphur": 25.0, "electrical_conductivity": 1.8},
                            "Mountain": {"ph": 5.6, "nitrogen": 85.0, "phosphorus": 30.0, "potassium": 45.0, "moisture": 50.0, "zinc": 1.1, "sulphur": 14.0, "electrical_conductivity": 0.4},
                            "Yellow": {"ph": 6.0, "nitrogen": 65.0, "phosphorus": 28.0, "potassium": 40.0, "moisture": 35.0, "zinc": 0.8, "sulphur": 10.0, "electrical_conductivity": 0.5}
                        }
                        if clean_type in soil_baselines:
                            st.session_state.soil_data.update(soil_baselines[clean_type])
                        st.rerun()

        # Render Soil Vision Output
        if st.session_state.cnn_result:
            res = st.session_state.cnn_result
            if not res.get("is_valid_soil", True):
                st.markdown(f"""
                <div class="agri-card-alert">
                    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
                        <span style="font-size: 1.3rem;">⚠️</span>
                        <b style="font-size: 1rem; color: #991b1b;">Non-Soil Image Detected (Rejected)</b>
                    </div>
                    <p style="color: #7f1d1d; font-size: 0.85rem; margin: 0; line-height: 1.4;">
                        <b>Reason:</b> {res.get('rejection_reason', 'The uploaded photo does not match natural agricultural soil or field ground characteristics.')}
                    </p>
                    <p style="color: #991b1b; font-size: 0.78rem; margin-top: 6px; margin-bottom: 0;">
                        💡 <i>Please upload an authentic photo of agricultural field soil or use the manual soil entry form on the right.</i>
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="agri-card-green">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <div>
                            <span style="font-size: 1.2rem; font-weight: 800; color: #14532d;">🎯 Predicted Soil: {res.get('detected_soil_type')}</span>
                            <div style="font-size: 0.78rem; color: #166534;">Verified against 7 Agro-Soil Taxonomy Classes</div>
                        </div>
                        <span class="badge-pill badge-green" style="font-size: 0.95rem;">{res.get('confidence', 0.0)*100:.1f}% Confidence</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Visual Features
                vf = res.get("visual_features", {})
                if vf:
                    st.markdown(f"""
                    <div class="metric-grid">
                        <div class="metric-box">
                            <div class="metric-box-title">🎨 Mean Hue</div>
                            <div class="metric-box-val">{vf.get('mean_hue', 0)}</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-box-title">💡 Brightness</div>
                            <div class="metric-box-val">{vf.get('mean_brightness', 0)}</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-box-title">🌊 Visual Moisture</div>
                            <div class="metric-box-val">{vf.get('estimated_visual_moisture', 'Medium')}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Active Soil Test Summary Card
                st.markdown(f"""
                <div class="agri-card" style="margin-top: 12px; background: #f8fafc; border: 1.5px solid #cbd5e1;">
                    <b style="font-size: 0.92rem; color: #0f172a;">📊 Active Soil Test Profile (Auto-Synchronized):</b>
                    <div class="metric-grid" style="margin-top: 6px;">
                        <div class="metric-box">
                            <div class="metric-box-title">Soil Type</div>
                            <div class="metric-box-val" style="font-size: 1rem; color: #15803d;">{st.session_state.soil_data['soil_type']}</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-box-title">Nitrogen (N)</div>
                            <div class="metric-box-val">{st.session_state.soil_data['nitrogen']} kg/ha</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-box-title">Phosphorus (P)</div>
                            <div class="metric-box-val">{st.session_state.soil_data['phosphorus']} kg/ha</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-box-title">Potassium (K)</div>
                            <div class="metric-box-val">{st.session_state.soil_data['potassium']} kg/ha</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-box-title">Soil pH</div>
                            <div class="metric-box-val">{st.session_state.soil_data['ph']}</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-box-title">Moisture</div>
                            <div class="metric-box-val">{st.session_state.soil_data['moisture']}%</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Probability breakdown across all 7 classes
                st.markdown("<div style='margin-top: 14px; font-weight: 700; font-size: 0.88rem; color: #166534;'>📊 Probability Distribution Across All 7 Soil Classes:</div>", unsafe_allow_html=True)
                for cls_name, prob in list(res.get("all_probabilities", {}).items()):
                    col_p1, col_p2 = st.columns([3, 1])
                    col_p1.write(f"**{cls_name}**")
                    col_p2.write(f"{prob*100:.1f}%")
                    st.progress(float(prob))

                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("🚀 Auto-Recommend Crops for this Soil", key="auto_rec_btn", type="primary", use_container_width=True):
                    with st.spinner("Computing precision crop suitability for predicted soil..."):
                        manual_props = ManualSoilProperties(
                            nitrogen=st.session_state.soil_data["nitrogen"],
                            phosphorus=st.session_state.soil_data["phosphorus"],
                            potassium=st.session_state.soil_data["potassium"],
                            ph=st.session_state.soil_data["ph"],
                            soil_type=st.session_state.soil_data["soil_type"],
                            moisture=st.session_state.soil_data["moisture"],
                            zinc=st.session_state.soil_data["zinc"],
                            sulphur=st.session_state.soil_data["sulphur"],
                            electrical_conductivity=st.session_state.soil_data["electrical_conductivity"]
                        )
                        req = CropRecommendationRequest(
                            soil_properties=manual_props,
                            city=st.session_state.city,
                            temperature=weather.get("temperature"),
                            humidity=weather.get("humidity"),
                            rainfall=weather.get("rainfall")
                        )
                        recs_resp = crop_service.recommend_crops(req)
                        st.session_state.recommendations = recs_resp.get("recommendations", [])
                        if st.session_state.recommendations:
                            st.session_state.selected_crop = st.session_state.recommendations[0]["crop_name"]
                        st.success(f"✅ Crops recommended for {res.get('detected_soil_type')}! Navigate to Tab 2 to view details.")

    with col_v2:
        st.subheader("🧪 Manual Soil Chemistry Report")
        st.write("Input your soil testing laboratory values. These parameters are combined with live meteorological data for high-accuracy crop recommendations.")
        
        with st.form("soil_chemistry_form"):
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                n_val = st.number_input("Nitrogen (N kg/ha)", min_value=0.0, max_value=500.0, value=float(st.session_state.soil_data["nitrogen"]), step=1.0)
                ph_val = st.number_input("Soil pH (0-14)", min_value=3.0, max_value=11.0, value=float(st.session_state.soil_data["ph"]), step=0.1)
                zn_val = st.number_input("Zinc (Zn ppm)", min_value=0.0, max_value=20.0, value=float(st.session_state.soil_data["zinc"] or 1.2), step=0.1)
            with col_s2:
                p_val = st.number_input("Phosphorus (P kg/ha)", min_value=0.0, max_value=300.0, value=float(st.session_state.soil_data["phosphorus"]), step=1.0)
                soil_type_val = st.selectbox("Soil Type", ["Alluvial", "Arid", "Black", "Laterite", "Mountain", "Red", "Yellow", "Clayey", "Sandy"], index=["Alluvial", "Arid", "Black", "Laterite", "Mountain", "Red", "Yellow", "Clayey", "Sandy"].index(st.session_state.soil_data.get("soil_type", "Black")) if st.session_state.soil_data.get("soil_type", "Black") in ["Alluvial", "Arid", "Black", "Laterite", "Mountain", "Red", "Yellow", "Clayey", "Sandy"] else 2)
                s_val = st.number_input("Sulphur (S ppm)", min_value=0.0, max_value=100.0, value=float(st.session_state.soil_data["sulphur"] or 15.0), step=1.0)
            with col_s3:
                k_val = st.number_input("Potassium (K kg/ha)", min_value=0.0, max_value=600.0, value=float(st.session_state.soil_data["potassium"]), step=1.0)
                moist_val = st.number_input("Moisture (%)", min_value=0.0, max_value=100.0, value=float(st.session_state.soil_data["moisture"] or 45.0), step=1.0)
                ec_val = st.number_input("EC (dS/m)", min_value=0.0, max_value=10.0, value=float(st.session_state.soil_data["electrical_conductivity"] or 0.8), step=0.1)
            
            save_soil = st.form_submit_button("💾 Update Soil Chemistry Profile", type="primary", use_container_width=True)
            if save_soil:
                st.session_state.soil_data.update({
                    "nitrogen": n_val,
                    "phosphorus": p_val,
                    "potassium": k_val,
                    "ph": ph_val,
                    "soil_type": soil_type_val,
                    "moisture": moist_val,
                    "zinc": zn_val,
                    "sulphur": s_val,
                    "electrical_conductivity": ec_val
                })
                st.session_state.manual_soil_saved = True
                st.success("✅ Soil chemistry parameters saved successfully!")
                st.rerun()

        # Display current active manual soil report
        st.markdown(f"""
        <div class="agri-card" style="margin-top: 10px; background: #ffffff;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <b style="color: #1e293b; font-size: 0.95rem;">📋 Current Field Soil Status:</b>
                <span class="badge-pill badge-green">{st.session_state.soil_data['soil_type']} Soil Active</span>
            </div>
            <div class="metric-grid">
                <div class="metric-box">
                    <div class="metric-box-title">N-P-K (kg/ha)</div>
                    <div class="metric-box-val" style="font-size: 0.95rem;">{st.session_state.soil_data['nitrogen']:.0f} - {st.session_state.soil_data['phosphorus']:.0f} - {st.session_state.soil_data['potassium']:.0f}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-box-title">pH Reaction</div>
                    <div class="metric-box-val">{st.session_state.soil_data['ph']:.1f}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-box-title">Moisture</div>
                    <div class="metric-box-val">{st.session_state.soil_data['moisture']:.0f}%</div>
                </div>
                <div class="metric-box">
                    <div class="metric-box-title">EC (dS/m)</div>
                    <div class="metric-box-val">{st.session_state.soil_data['electrical_conductivity']:.1f}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# =============================================================
# TAB 2: Adaptive Crop Recommendation (ML Engine)
# =============================================================
with tab2:
    st.subheader("🌾 Step 2: Adaptive Precision Crop Recommendation")
    st.write("Generates optimal crop suitability rankings using the trained Ensemble ML classifier (30 crops), combining your soil parameters with real-time Open-Meteo satellite weather.")
    
    # Check if Step 1 was executed
    has_step1_vision = (st.session_state.cnn_result is not None and st.session_state.cnn_result.get("is_valid_soil", False))
    has_step1_manual = st.session_state.get("manual_soil_saved", False)
    
    # Soil Selector Box (Allows user to select/confirm soil in Step 2 directly)
    soil_options = ["Black", "Red", "Alluvial", "Laterite", "Arid", "Mountain", "Yellow", "Clayey", "Sandy"]
    current_soil_type = st.session_state.soil_data.get("soil_type", "Black")
    current_idx = soil_options.index(current_soil_type) if current_soil_type in soil_options else 0

    st.markdown("""<div class="agri-card" style="background: #f8fafc; border: 1.5px solid #94a3b8; margin-bottom: 1rem;">""", unsafe_allow_html=True)
    
    col_t2_1, col_t2_2 = st.columns([2, 1])
    with col_t2_1:
        if not has_step1_vision and not has_step1_manual:
            st.info("ℹ️ **No Step 1 Soil Test Run Yet**: Select your farm's preferred soil type below to auto-load regional baseline nutrients, or customize parameters in the expander.")
        else:
            st.success(f"✅ **Active Soil Type**: {current_soil_type} Soil (Configured from Step 1)")
        
        selected_soil_in_tab2 = st.selectbox(
            "🌱 Which soil type is in your farm / field?",
            soil_options,
            index=current_idx,
            key="tab2_preferred_soil"
        )
        if selected_soil_in_tab2 != current_soil_type:
            st.session_state.soil_data["soil_type"] = selected_soil_in_tab2
            soil_baselines = {
                "Black": {"ph": 7.8, "nitrogen": 90.0, "phosphorus": 42.0, "potassium": 48.0, "moisture": 45.0, "zinc": 1.2, "sulphur": 15.0, "electrical_conductivity": 0.8},
                "Alluvial": {"ph": 7.2, "nitrogen": 110.0, "phosphorus": 52.0, "potassium": 55.0, "moisture": 40.0, "zinc": 1.5, "sulphur": 18.0, "electrical_conductivity": 0.6},
                "Red": {"ph": 6.2, "nitrogen": 75.0, "phosphorus": 35.0, "potassium": 50.0, "moisture": 30.0, "zinc": 0.9, "sulphur": 12.0, "electrical_conductivity": 0.4},
                "Laterite": {"ph": 5.2, "nitrogen": 55.0, "phosphorus": 22.0, "potassium": 35.0, "moisture": 28.0, "zinc": 0.6, "sulphur": 8.0, "electrical_conductivity": 0.3},
                "Arid": {"ph": 8.4, "nitrogen": 40.0, "phosphorus": 18.0, "potassium": 65.0, "moisture": 15.0, "zinc": 0.5, "sulphur": 25.0, "electrical_conductivity": 1.8},
                "Mountain": {"ph": 5.6, "nitrogen": 85.0, "phosphorus": 30.0, "potassium": 45.0, "moisture": 50.0, "zinc": 1.1, "sulphur": 14.0, "electrical_conductivity": 0.4},
                "Yellow": {"ph": 6.0, "nitrogen": 65.0, "phosphorus": 28.0, "potassium": 40.0, "moisture": 35.0, "zinc": 0.8, "sulphur": 10.0, "electrical_conductivity": 0.5},
                "Clayey": {"ph": 7.4, "nitrogen": 95.0, "phosphorus": 40.0, "potassium": 45.0, "moisture": 50.0, "zinc": 1.0, "sulphur": 14.0, "electrical_conductivity": 0.7},
                "Sandy": {"ph": 6.5, "nitrogen": 50.0, "phosphorus": 25.0, "potassium": 30.0, "moisture": 20.0, "zinc": 0.6, "sulphur": 10.0, "electrical_conductivity": 0.4}
            }
            if selected_soil_in_tab2 in soil_baselines:
                st.session_state.soil_data.update(soil_baselines[selected_soil_in_tab2])
            st.rerun()

    with col_t2_2:
        st.markdown(f"""
        <div style="padding: 10px; background: #ffffff; border-radius: 8px; border: 1px solid #e2e8f0; font-size: 0.82rem;">
            <b>Active Nutrients for Model:</b><br>
            • N: <b>{st.session_state.soil_data['nitrogen']:.0f}</b> kg/ha<br>
            • P: <b>{st.session_state.soil_data['phosphorus']:.0f}</b> kg/ha<br>
            • K: <b>{st.session_state.soil_data['potassium']:.0f}</b> kg/ha<br>
            • pH: <b>{st.session_state.soil_data['ph']:.1f}</b> | Soil: <b>{st.session_state.soil_data['soil_type']}</b>
        </div>
        """, unsafe_allow_html=True)
    
    with st.expander("⚙️ Fine-Tune Soil Chemistry & Environmental Overrides (Optional)"):
        col_o1, col_o2, col_o3 = st.columns(3)
        with col_o1:
            t2_n = st.number_input("Nitrogen (kg/ha)", value=float(st.session_state.soil_data["nitrogen"]), step=1.0, key="t2_n")
            t2_ph = st.number_input("pH Level", value=float(st.session_state.soil_data["ph"]), step=0.1, key="t2_ph")
        with col_o2:
            t2_p = st.number_input("Phosphorus (kg/ha)", value=float(st.session_state.soil_data["phosphorus"]), step=1.0, key="t2_p")
            t2_temp = st.number_input("Override Temp (°C)", value=float(weather.get("temperature", 28.0)), step=0.5, key="t2_temp")
        with col_o3:
            t2_k = st.number_input("Potassium (kg/ha)", value=float(st.session_state.soil_data["potassium"]), step=1.0, key="t2_k")
            t2_hum = st.number_input("Override Humidity (%)", value=float(weather.get("humidity", 65.0)), step=1.0, key="t2_hum")
        
        if st.button("Apply Overrides", key="apply_overrides_btn"):
            st.session_state.soil_data["nitrogen"] = t2_n
            st.session_state.soil_data["phosphorus"] = t2_p
            st.session_state.soil_data["potassium"] = t2_k
            st.session_state.soil_data["ph"] = t2_ph
            st.success("Overrides applied to active model!")
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
    
    if st.button("🚀 Generate Crop Recommendations", type="primary", use_container_width=True):
        with st.spinner(f"Evaluating 30-crop ML model for {st.session_state.city} ({st.session_state.soil_data['soil_type']} Soil)..."):
            manual_props = ManualSoilProperties(
                nitrogen=st.session_state.soil_data["nitrogen"],
                phosphorus=st.session_state.soil_data["phosphorus"],
                potassium=st.session_state.soil_data["potassium"],
                ph=st.session_state.soil_data["ph"],
                soil_type=st.session_state.soil_data["soil_type"],
                moisture=st.session_state.soil_data["moisture"],
                zinc=st.session_state.soil_data["zinc"],
                sulphur=st.session_state.soil_data["sulphur"],
                electrical_conductivity=st.session_state.soil_data["electrical_conductivity"]
            )
            req = CropRecommendationRequest(
                soil_properties=manual_props,
                city=st.session_state.city,
                temperature=weather.get("temperature"),
                humidity=weather.get("humidity"),
                rainfall=weather.get("rainfall")
            )
            recs_resp = crop_service.recommend_crops(req)
            st.session_state.recommendations = recs_resp.get("recommendations", [])
            if st.session_state.recommendations:
                st.session_state.selected_crop = st.session_state.recommendations[0]["crop_name"]

    if st.session_state.recommendations:
        st.markdown(f"### Top Recommended Crops for **{st.session_state.city}** ({st.session_state.soil_data['soil_type']} Soil):")
        
        cols = st.columns(min(3, len(st.session_state.recommendations)))
        for idx, crop_item in enumerate(st.session_state.recommendations[:3]):
            with cols[idx]:
                is_top = (idx == 0)
                card_style = "agri-card-green" if is_top else "agri-card"
                badge_style = "badge-green" if is_top else "badge-blue"
                
                st.markdown(f"""
                <div class="{card_style}">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <h4 style="margin: 0; color: #1e293b;">{idx+1}. {crop_item['crop_name']}</h4>
                        <span class="badge-pill {badge_style}">{(crop_item.get('suitability_score', crop_item.get('confidence', 0)*100)):.1f}% Match</span>
                    </div>
                    <p style="font-size: 0.82rem; color: #475569; margin: 4px 0;"><b>📅 Season:</b> {crop_item.get('recommended_season', 'Kharif / Rabi')}</p>
                    <p style="font-size: 0.82rem; color: #475569; margin: 4px 0;"><b>⏳ Duration:</b> {crop_item.get('growth_duration_days', 120)} Days</p>
                    <p style="font-size: 0.82rem; color: #475569; margin: 4px 0;"><b>💧 Water Need:</b> {crop_item.get('water_requirement', 'Medium')}</p>
                    <p style="font-size: 0.82rem; color: #475569; margin: 4px 0;"><b>🧪 Recommended Fertilizers:</b> {', '.join(crop_item.get('optimal_fertilizers', [])[:2])}</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button(f"👉 Select {crop_item['crop_name']} for Planning", key=f"select_{crop_item['crop_name']}", use_container_width=True):
                    st.session_state.selected_crop = crop_item["crop_name"]
                    st.success(f"Selected {crop_item['crop_name']} as active crop! Proceed to Tab 3 / Tab 4.")


# =============================================================
# TAB 3: Quantitative Fertilizer Deficit Calculator
# =============================================================
with tab3:
    st.subheader(f"🧪 Quantitative Fertilizer Deficit Plan for **{st.session_state.selected_crop}**")
    st.write("Calculates exact commercial fertilizer quantities (50-kg bags of Urea, DAP, MOP, SSP) based on soil chemistry test values and crop nutrient uptake targets.")
    
    if st.button("📊 Calculate Fertilizer Deficits", type="primary", use_container_width=True):
        with st.spinner("Executing quantitative nutrient balance formulas..."):
            manual_props = ManualSoilProperties(
                nitrogen=st.session_state.soil_data["nitrogen"],
                phosphorus=st.session_state.soil_data["phosphorus"],
                potassium=st.session_state.soil_data["potassium"],
                ph=st.session_state.soil_data["ph"],
                soil_type=st.session_state.soil_data["soil_type"],
                moisture=st.session_state.soil_data["moisture"],
                zinc=st.session_state.soil_data["zinc"],
                sulphur=st.session_state.soil_data["sulphur"],
                electrical_conductivity=st.session_state.soil_data["electrical_conductivity"]
            )
            fert_req = FertilizerRecommendationRequest(
                crop_name=st.session_state.selected_crop,
                soil_properties=manual_props
            )
            st.session_state.fertilizer_data = fertilizer_service.recommend_fertilizers(fert_req)

    if st.session_state.fertilizer_data:
        fdata = st.session_state.fertilizer_data
        
        col_f1, col_f2 = st.columns([1, 1], gap="large")
        with col_f1:
            st.markdown(f"""
            <div class="agri-card-green">
                <h4 style="margin: 0 0 10px 0; color: #14532d;">🎯 Primary Recommendation: {fdata.get('primary_fertilizer')}</h4>
                <p style="font-size: 0.88rem; color: #166534; margin: 0 0 10px 0;">{fdata.get('application_guidelines')}</p>
                <div class="metric-grid">
                    <div class="metric-box">
                        <div class="metric-box-title">N Deficit</div>
                        <div class="metric-box-val">{fdata.get('dosage_kg_per_ha', {}).get('nitrogen_deficit_kg_ha', 0)} kg/ha</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-box-title">P Deficit</div>
                        <div class="metric-box-val">{fdata.get('dosage_kg_per_ha', {}).get('phosphorus_deficit_kg_ha', 0)} kg/ha</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-box-title">K Deficit</div>
                        <div class="metric-box-val">{fdata.get('dosage_kg_per_ha', {}).get('potassium_deficit_kg_ha', 0)} kg/ha</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### 📦 Commercial 50-kg Bag Requirements:")
            bags = fdata.get("commercial_bags_recommended", {})
            col_b1, col_b2, col_b3, col_b4 = st.columns(4)
            col_b1.metric("Urea Bags", f"{bags.get('urea_50kg_bags', 0)} bags")
            col_b2.metric("DAP Bags", f"{bags.get('dap_50kg_bags', 0)} bags")
            col_b3.metric("MOP Bags", f"{bags.get('mop_50kg_bags', 0)} bags")
            col_b4.metric("SSP Bags", f"{bags.get('ssp_50kg_bags', 0)} bags")
            
        with col_f2:
            st.markdown("#### 🌿 Organic & Micronutrient Amendments:")
            st.info(f"**Organic Recommendation:** {fdata.get('organic_alternatives', 'Apply Farm Yard Manure (FYM) @ 10 tonnes/ha')}")
            
            micro = fdata.get("micronutrient_advice", {})
            if micro:
                st.write("**Micronutrient Alerts:**")
                for k, v in micro.items():
                    st.warning(f"• **{k.title()}:** {v}")

# =============================================================
# TAB 4: Dynamic Crop Lifecycle Timeline (Sowing to Harvest)
# =============================================================
with tab4:
    st.subheader(f"📅 Stage-Wise Lifecycle Timeline for **{st.session_state.selected_crop}**")
    st.write("Generates calendar-anchored dates, growth stage activities, irrigation intervals, and critical notes starting from your Sowing Date.")
    
    col_t1, col_t2 = st.columns([1, 2])
    with col_t1:
        sowing_input = st.date_input("Select Sowing Date:", value=st.session_state.sowing_date)
        if sowing_input != st.session_state.sowing_date:
            st.session_state.sowing_date = sowing_input
            
        if st.button("🗓️ Generate Dynamic Calendar", type="primary", use_container_width=True):
            with st.spinner("Computing agronomic calendar date ranges..."):
                t_resp = timeline_service.generate_timeline(
                    crop_name=st.session_state.selected_crop,
                    sowing_date=st.session_state.sowing_date.strftime("%Y-%m-%d"),
                    soil_type=st.session_state.soil_data["soil_type"]
                )
                st.session_state.timeline_data = t_resp
                
    if st.session_state.timeline_data:
        stages = st.session_state.timeline_data.get("stages", [])
        st.markdown(f"### Total Growth Duration: **{st.session_state.timeline_data.get('total_duration_days', 120)} Days** ({st.session_state.timeline_data.get('sowing_date')} to {st.session_state.timeline_data.get('estimated_harvest_date')})")
        
        for stg in stages:
            with st.expander(f"📍 **Stage {stg['stage_id']}: {stg['stage_name']}** (Day {stg['start_day']} - {stg['end_day']} | {stg.get('start_date', '')} to {stg.get('end_date', '')})", expanded=(stg['stage_id'] <= 2)):
                st.markdown(f"**⚡ Critical Growth Focus:** {stg.get('critical_notes')}")
                
                col_act, col_fert = st.columns([1, 1])
                with col_act:
                    st.write("**Key Field Activities:**")
                    for act in stg.get("activities", []):
                        st.write(f"• {act}")
                    st.write(f"**💧 Irrigation:** {stg.get('irrigation_schedule')}")
                
                with col_fert:
                    st.write(f"**🧪 Fertilizer Advice:** {stg.get('fertilizer_advice')}")
                    st.write(f"**🐛 Pest & Disease Watch:** {stg.get('pest_disease_watch')}")

# =============================================================
# TAB 5: Proactive Weather & Stage Alerts
# =============================================================
with tab5:
    st.subheader(f"🔔 Proactive Weather & Hazard Intelligence ({st.session_state.city})")
    st.write("Streams real-time meteorological danger thresholds and provides preventative agronomic checklists.")
    
    with st.spinner("Evaluating weather risks & stage-specific agronomic advisories..."):
        sowing_str = st.session_state.sowing_date.strftime("%Y-%m-%d") if hasattr(st.session_state.sowing_date, "strftime") else str(st.session_state.sowing_date)
        notifs = notification_service.get_notifications(
            crop_name=st.session_state.selected_crop,
            sowing_date=sowing_str,
            city=st.session_state.city
        )
        
    for n in notifs:
        sev = n.get("severity", "info")
        if sev in ["danger", "critical"]:
            st.error(f"🚨 **{n.get('title')}**\n\n{n.get('message')}\n\n**Action:** {n.get('action_required')}")
        elif sev in ["warning", "medium"]:
            st.warning(f"⚠️ **{n.get('title')}**\n\n{n.get('message')}\n\n**Action:** {n.get('action_required')}")
        else:
            st.info(f"ℹ️ **{n.get('title')}**\n\n{n.get('message')}\n\n**Action:** {n.get('action_required')}")

# =============================================================
# TAB 6: AI Farm Knowledge Assistant (RAG Chatbot)
# =============================================================
with tab6:
    st.subheader("🤖 Agricultural AI Knowledge Assistant (RAG Grounded)")
    st.write("Ask any farming question. The conversational AI assistant answers naturally and retrieves scientific citations from 67+ authoritative ICAR & AgricultureGuruji Markdown guides and vector databases.")
    
    # Display Chat History
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                st.caption(f"📖 Sources: {', '.join(msg['sources'])}")

    # Chat Input
    if user_prompt := st.chat_input("Ask about fertilizer doses, pest control, weather precautions..."):
        st.session_state.chat_messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Searching agricultural knowledge base..."):
                weather_ctx = f"{weather.get('weather_condition', 'Clear')}, {weather.get('temperature', 28)}°C, Humidity {weather.get('humidity', 65)}%"
                ans_obj = rag_assistant_service.answer_query(
                    query=user_prompt,
                    crop_context=st.session_state.selected_crop,
                    soil_context=f"{st.session_state.soil_data['soil_type']} Soil (pH {st.session_state.soil_data['ph']})",
                    growth_stage_context="Vegetative / Active Growth",
                    weather_context=weather_ctx
                )
                st.markdown(ans_obj["answer"])
                if ans_obj.get("grounded_sources"):
                    st.caption(f"📖 Grounded Sources: {', '.join(ans_obj['grounded_sources'])}")
                    
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": ans_obj["answer"],
                    "sources": ans_obj.get("grounded_sources", [])
                })
