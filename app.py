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
from backend.app.translations import get_text, TRANSLATIONS

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
        font-size: 2.1rem;
        font-weight: 800;
        color: #14532d;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #4b5563;
        margin-bottom: 1.25rem;
    }
    
    /* Card Components */
    .agri-card {
        background: #ffffff;
        padding: 1.2rem 1.4rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 5px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }
    
    .agri-card-green {
        background: #f0fdf4;
        border: 1.5px solid #86efac;
        padding: 1.2rem;
        border-radius: 12px;
        margin-bottom: 1rem;
    }
    
    .agri-card-alert {
        background: #fef2f2;
        border: 1.5px solid #fca5a5;
        padding: 1rem 1.25rem;
        border-radius: 12px;
        margin-bottom: 1rem;
    }

    .agri-card-warning {
        background: #fffbeb;
        border: 1.5px solid #fcd34d;
        padding: 1rem 1.25rem;
        border-radius: 12px;
        margin-bottom: 1rem;
    }

    .agri-card-info {
        background: #f0f9ff;
        border: 1.5px solid #bae6fd;
        padding: 1rem 1.25rem;
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
        grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
        gap: 0.65rem;
        margin-top: 0.5rem;
    }
    .metric-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.65rem;
        text-align: center;
    }
    .metric-box-title { font-size: 0.72rem; color: #64748b; font-weight: 600; text-transform: uppercase; }
    .metric-box-val { font-size: 1.1rem; font-weight: 700; color: #1e293b; margin-top: 2px; }

    /* Timeline Step Card */
    .timeline-card {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-left: 5px solid #16a34a;
        padding: 1rem 1.25rem;
        border-radius: 8px;
        margin-bottom: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 2. Session State Initialization & Auto-Geolocation
# -------------------------------------------------------------
if "lang" not in st.session_state:
    st.session_state.lang = "en"

if "city" not in st.session_state:
    # Auto-detect real client location on initial launch
    detected = weather_service.detect_client_location()
    st.session_state.city = detected.get("city_raw") or detected.get("city", "Hyderabad")
    st.session_state.detected_loc_info = detected

if "soil_data" not in st.session_state:
    st.session_state.soil_data = {
        "nitrogen": 90.0,
        "phosphorus": 42.0,
        "potassium": 48.0,
        "ph": 7.2,
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
            "content": "👋 Hello! I am your AI Agronomist. Ask me anything in English, Hindi, Telugu, Tamil, Marathi, or Kannada about crop planning, fertilizer doses, pest management, or weather precautions."
        }
    ]

lang = st.session_state.lang

# Regional agronomic chemical baselines by soil taxonomy
SOIL_BASELINES = {
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

# -------------------------------------------------------------
# 3. Sidebar: Language Switcher, Geolocation & Weather Feed
# -------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/plant-under-rain.png", width=60)
    st.title(get_text("sidebar_title", lang))
    
    # Language Selector
    lang_map = {
        "en": "English",
        "hi": "हिन्दी (Hindi)",
        "te": "తెలుగు (Telugu)",
        "ta": "தமிழ் (Tamil)",
        "mr": "मराठी (Marathi)",
        "kn": "ಕನ್ನಡ (Kannada)"
    }
    selected_lang = st.selectbox(
        get_text("sidebar_lang_label", lang),
        options=list(lang_map.keys()),
        format_func=lambda code: lang_map[code],
        index=list(lang_map.keys()).index(st.session_state.lang)
    )
    if selected_lang != st.session_state.lang:
        st.session_state.lang = selected_lang
        st.rerun()

    st.markdown("---")
    st.subheader(get_text("sidebar_loc_header", lang))
    
    city_input = st.text_input(get_text("sidebar_city_label", lang), value=st.session_state.city)
    if city_input != st.session_state.city:
        st.session_state.city = city_input
        st.rerun()

    if st.button(get_text("sidebar_autodetect_btn", lang), use_container_width=True):
        with st.spinner("Detecting client GPS/IP coordinates..."):
            detected = weather_service.detect_client_location()
            st.session_state.city = detected.get("city_raw") or detected.get("city", "Hyderabad")
            st.session_state.detected_loc_info = detected
            st.success(f"📍 Detected: {st.session_state.city}")
            st.rerun()

    # Fetch Real-Time Satellite Weather Feed
    with st.spinner("Fetching meteorological telemetry..."):
        weather = weather_service.get_weather(st.session_state.city)

    st.markdown(f"""
    <div class="agri-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <b style="font-size: 0.95rem; color: #1e293b;">{weather.get('city', st.session_state.city)}</b>
            <span class="badge-pill badge-green">{get_text('live_satellite', lang)}</span>
        </div>
        <div style="font-size: 0.82rem; color: #15803d; font-weight: 600; margin-bottom: 8px;">
            🌤️ {weather.get('weather_condition', 'Clear Sky')} ({get_text('feels_like', lang)} {weather.get('feels_like', weather.get('temperature', 28.0))}°C)
        </div>
        <div class="metric-grid">
            <div class="metric-box">
                <div class="metric-box-title">{get_text('temp', lang)}</div>
                <div class="metric-box-val">{weather.get('temperature', 28.0)}°C</div>
            </div>
            <div class="metric-box">
                <div class="metric-box-title">{get_text('humidity', lang)}</div>
                <div class="metric-box-val">{weather.get('humidity', 65.0)}%</div>
            </div>
            <div class="metric-box">
                <div class="metric-box-title">{get_text('rain_24h', lang)}</div>
                <div class="metric-box-val">{weather.get('rainfall', 0.0)} mm</div>
            </div>
            <div class="metric-box">
                <div class="metric-box-title">{get_text('pressure', lang)}</div>
                <div class="metric-box-val">{weather.get('pressure', 1012.0)} hPa</div>
            </div>
            <div class="metric-box">
                <div class="metric-box-title">{get_text('wind', lang)}</div>
                <div class="metric-box-val">{weather.get('wind_speed', 5.0)} km/h</div>
            </div>
            <div class="metric-box">
                <div class="metric-box-title">{get_text('rain_chance', lang)}</div>
                <div class="metric-box-val">{weather.get('rain_probability', 0)}%</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("🤖 **Knowledge Base:** 67 Guides (343 Chunks)")
    st.caption("🧠 **ML Model:** XGBoost + LightGBM (30 Crops)")
    st.caption("👁️ **Vision Model:** 133 Features + OOD Domain Guard")

# -------------------------------------------------------------
# 4. Top Header & Proactive Weather/Pest Alert Banner
# -------------------------------------------------------------
st.markdown(f'<div class="main-header">{get_text("app_title", lang)}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-header">{get_text("app_subtitle", lang)}</div>', unsafe_allow_html=True)

# Fetch Proactive Hazards & Stage Alerts (Displayed across all tabs)
sowing_str = st.session_state.sowing_date.strftime("%Y-%m-%d") if hasattr(st.session_state.sowing_date, "strftime") else str(st.session_state.sowing_date)
active_alerts = notification_service.get_notifications(
    crop_name=st.session_state.selected_crop,
    sowing_date=sowing_str,
    city=st.session_state.city
)

critical_or_warn_alerts = [a for a in active_alerts if a.get("severity") in ["critical", "danger", "warning", "medium"]]

if critical_or_warn_alerts:
    top_alert = critical_or_warn_alerts[0]
    alert_cls = "agri-card-alert" if top_alert.get("severity") in ["critical", "danger"] else "agri-card-warning"
    icon = "🚨" if top_alert.get("severity") in ["critical", "danger"] else "⚠️"
    
    st.markdown(f"""
    <div class="{alert_cls}">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
            <b style="font-size: 0.95rem; color: #1e293b;">{icon} {top_alert.get('title')}</b>
            <span class="badge-pill badge-red" style="font-size: 0.75rem;">Active Hazard</span>
        </div>
        <p style="font-size: 0.85rem; color: #334155; margin: 2px 0 4px 0;">{top_alert.get('message')}</p>
        <p style="font-size: 0.82rem; color: #0f172a; margin: 0; font-weight: 600;"><b>💡 {get_text('action_required', lang)}</b> {top_alert.get('action_required')}</p>
    </div>
    """, unsafe_allow_html=True)

    if len(active_alerts) > 1:
        with st.expander(f"📋 View All {len(active_alerts)} Active Farm Hazard & Agro-Advisory Alerts for {st.session_state.city}"):
            for alt in active_alerts:
                sev = alt.get("severity", "info")
                if sev in ["critical", "danger"]:
                    st.error(f"🚨 **{alt.get('title')}**\n\n{alt.get('message')}\n\n**Action:** {alt.get('action_required')}")
                elif sev in ["warning", "medium"]:
                    st.warning(f"⚠️ **{alt.get('title')}**\n\n{alt.get('message')}\n\n**Action:** {alt.get('action_required')}")
                else:
                    st.info(f"ℹ️ **{alt.get('title')}**\n\n{alt.get('message')}\n\n**Action:** {alt.get('action_required')}")

# -------------------------------------------------------------
# 5. Main 5 Unified Tabs
# -------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    get_text("tab_soil", lang),
    get_text("tab_crop", lang),
    get_text("tab_fert", lang),
    get_text("tab_timeline", lang),
    get_text("tab_rag", lang)
])

# =============================================================
# TAB 1: Soil Vision & Soil Chemistry Lab Report
# =============================================================
with tab1:
    col_v1, col_v2 = st.columns([1, 1], gap="large")
    
    # Left Column: Soil Photo Vision & OOD Guard
    with col_v1:
        st.subheader(get_text("tab1_upload_title", lang))
        st.write(get_text("tab1_upload_desc", lang))
        
        uploaded_file = st.file_uploader(get_text("tab1_upload_label", lang), type=["jpg", "jpeg", "png", "webp"])
        
        if uploaded_file is not None:
            image_bytes = uploaded_file.read()
            image = Image.open(io.BytesIO(image_bytes))
            st.image(image, caption="Uploaded Soil Sample", use_container_width=True)
            
            if st.button(get_text("tab1_analyze_btn", lang), type="primary", use_container_width=True):
                with st.spinner("Extracting 133 visual features & verifying domain distribution..."):
                    result = soil_service.analyze_soil_image(image_bytes)
                    st.session_state.cnn_result = result
                    
                    if result.get("is_valid_soil") and result.get("detected_soil_type"):
                        clean_type = result["detected_soil_type"].replace(" Soil", "").strip()
                        st.session_state.soil_data["soil_type"] = clean_type
                        
                        # Auto-sync baseline chemical nutrients globally
                        if clean_type in SOIL_BASELINES:
                            st.session_state.soil_data.update(SOIL_BASELINES[clean_type])
                        
                        st.success(get_text("tab1_sync_msg", lang))
                        st.rerun()

        # Display Vision Result
        if st.session_state.cnn_result:
            res = st.session_state.cnn_result
            if not res.get("is_valid_soil", True):
                st.markdown(f"""
                <div class="agri-card-alert">
                    <b style="color: #991b1b; font-size: 0.95rem;">⚠️ Non-Soil Image Rejected (Out of Distribution)</b>
                    <p style="color: #7f1d1d; font-size: 0.85rem; margin: 4px 0;"><b>Reason:</b> {res.get('rejection_reason', 'Image does not match natural soil texture/color distribution.')}</p>
                    <p style="color: #991b1b; font-size: 0.78rem; margin: 0;">💡 Please upload a genuine photo of agricultural soil or enter lab test values on the right.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="agri-card-green">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <span style="font-size: 1.1rem; font-weight: 800; color: #14532d;">{get_text('tab1_pred_badge', lang)} {res.get('detected_soil_type')}</span>
                        <span class="badge-pill badge-green">{res.get('confidence', 0.0)*100:.1f}% {get_text('tab1_conf', lang)}</span>
                    </div>
                    <div style="font-size: 0.8rem; color: #166534;">✓ Soil type & nutrient profile auto-synced across Tabs 2, 3, 4</div>
                </div>
                """, unsafe_allow_html=True)
                
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

                if st.button(get_text("tab1_auto_rec_btn", lang), key="tab1_auto_rec_btn", type="primary", use_container_width=True):
                    with st.spinner("Evaluating crop recommendations for predicted soil..."):
                        manual_props = ManualSoilProperties(**st.session_state.soil_data)
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
                        st.success("✅ Recommendations ready! Navigate to Tab 2 to view details.")

    # Right Column: Manual Soil Chemistry Report
    with col_v2:
        st.subheader(get_text("tab1_manual_title", lang))
        st.write(get_text("tab1_manual_desc", lang))
        
        soil_type_list = ["Black", "Alluvial", "Red", "Laterite", "Arid", "Mountain", "Yellow", "Clayey", "Sandy"]
        cur_type = st.session_state.soil_data.get("soil_type", "Black")
        cur_type_idx = soil_type_list.index(cur_type) if cur_type in soil_type_list else 0
        
        with st.form("soil_chemistry_form"):
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                n_val = st.number_input(get_text("tab1_nitrogen", lang), min_value=0.0, max_value=500.0, value=float(st.session_state.soil_data["nitrogen"]), step=1.0)
                ph_val = st.number_input(get_text("tab1_ph", lang), min_value=3.0, max_value=11.0, value=float(st.session_state.soil_data["ph"]), step=0.1)
                zn_val = st.number_input(get_text("tab1_zinc", lang), min_value=0.0, max_value=20.0, value=float(st.session_state.soil_data["zinc"] or 1.2), step=0.1)
            with col_s2:
                p_val = st.number_input(get_text("tab1_phosphorus", lang), min_value=0.0, max_value=300.0, value=float(st.session_state.soil_data["phosphorus"]), step=1.0)
                soil_type_val = st.selectbox(get_text("tab1_soil_type", lang), soil_type_list, index=cur_type_idx)
                s_val = st.number_input(get_text("tab1_sulphur", lang), min_value=0.0, max_value=100.0, value=float(st.session_state.soil_data["sulphur"] or 15.0), step=1.0)
            with col_s3:
                k_val = st.number_input(get_text("tab1_potassium", lang), min_value=0.0, max_value=600.0, value=float(st.session_state.soil_data["potassium"]), step=1.0)
                moist_val = st.number_input(get_text("tab1_moisture", lang), min_value=0.0, max_value=100.0, value=float(st.session_state.soil_data["moisture"] or 45.0), step=1.0)
                ec_val = st.number_input(get_text("tab1_ec", lang), min_value=0.0, max_value=10.0, value=float(st.session_state.soil_data["electrical_conductivity"] or 0.8), step=0.1)
            
            save_soil = st.form_submit_button(get_text("tab1_save_btn", lang), type="primary", use_container_width=True)
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
                st.success("✅ Soil chemistry parameters saved and synchronized across all tabs!")
                st.rerun()

    # Unified Field Soil Status Card (Across entire app)
    st.markdown(f"""
    <div class="agri-card" style="margin-top: 8px; background: #f8fafc; border: 1.5px solid #cbd5e1;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
            <b style="color: #1e293b; font-size: 0.95rem;">{get_text('tab1_active_summary', lang)}</b>
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
                <div class="metric-box-title">Zinc (ppm)</div>
                <div class="metric-box-val">{st.session_state.soil_data['zinc']:.1f}</div>
            </div>
            <div class="metric-box">
                <div class="metric-box-title">Sulphur (ppm)</div>
                <div class="metric-box-val">{st.session_state.soil_data['sulphur']:.0f}</div>
            </div>
            <div class="metric-box">
                <div class="metric-box-title">EC (dS/m)</div>
                <div class="metric-box-val">{st.session_state.soil_data['electrical_conductivity']:.1f}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================
# TAB 2: Adaptive Crop Recommendation (30 Crops ML Model)
# =============================================================
with tab2:
    st.subheader(get_text("tab2_title", lang))
    st.write(get_text("tab2_desc", lang))
    
    soil_options = ["Black", "Alluvial", "Red", "Laterite", "Arid", "Mountain", "Yellow", "Clayey", "Sandy"]
    cur_soil = st.session_state.soil_data.get("soil_type", "Black")
    cur_idx = soil_options.index(cur_soil) if cur_soil in soil_options else 0

    st.markdown("""<div class="agri-card" style="background: #f8fafc; border: 1.5px solid #cbd5e1; margin-bottom: 1rem;">""", unsafe_allow_html=True)
    col_t2_1, col_t2_2 = st.columns([2, 1])
    
    with col_t2_1:
        selected_soil_in_tab2 = st.selectbox(
            get_text("tab2_soil_prompt", lang),
            soil_options,
            index=cur_idx,
            key="tab2_preferred_soil_select"
        )
        if selected_soil_in_tab2 != cur_soil:
            st.session_state.soil_data["soil_type"] = selected_soil_in_tab2
            if selected_soil_in_tab2 in SOIL_BASELINES:
                st.session_state.soil_data.update(SOIL_BASELINES[selected_soil_in_tab2])
            st.rerun()

    with col_t2_2:
        st.markdown(f"""
        <div style="padding: 8px 12px; background: #ffffff; border-radius: 8px; border: 1px solid #e2e8f0; font-size: 0.82rem;">
            <b>Active Nutrients for Model:</b><br>
            • N: <b>{st.session_state.soil_data['nitrogen']:.0f}</b> | P: <b>{st.session_state.soil_data['phosphorus']:.0f}</b> | K: <b>{st.session_state.soil_data['potassium']:.0f}</b> kg/ha<br>
            • pH: <b>{st.session_state.soil_data['ph']:.1f}</b> | Soil: <b>{st.session_state.soil_data['soil_type']}</b>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    if st.button(get_text("tab2_btn", lang), type="primary", use_container_width=True):
        with st.spinner(f"Evaluating 30-crop ML ensemble for {st.session_state.city} ({st.session_state.soil_data['soil_type']} Soil)..."):
            manual_props = ManualSoilProperties(**st.session_state.soil_data)
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
        st.markdown(f"### {get_text('tab2_top_heading', lang)} **{st.session_state.city}** ({st.session_state.soil_data['soil_type']} Soil):")
        
        cols = st.columns(min(3, len(st.session_state.recommendations)))
        for idx, crop_item in enumerate(st.session_state.recommendations[:3]):
            with cols[idx]:
                is_selected = (crop_item["crop_name"] == st.session_state.selected_crop)
                card_style = "agri-card-green" if is_selected else "agri-card"
                badge_style = "badge-green" if is_selected else "badge-blue"
                
                st.markdown(f"""
                <div class="{card_style}">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <h4 style="margin: 0; color: #1e293b;">{idx+1}. {crop_item['crop_name']}</h4>
                        <span class="badge-pill {badge_style}">{(crop_item.get('suitability_score', crop_item.get('confidence', 0)*100)):.1f}% {get_text('tab2_match', lang)}</span>
                    </div>
                    <p style="font-size: 0.82rem; color: #475569; margin: 4px 0;"><b>📅 {get_text('tab2_season', lang)}:</b> {crop_item.get('recommended_season', 'Kharif / Rabi')}</p>
                    <p style="font-size: 0.82rem; color: #475569; margin: 4px 0;"><b>⏳ {get_text('tab2_duration', lang)}:</b> {crop_item.get('growth_duration_days', 120)} Days</p>
                    <p style="font-size: 0.82rem; color: #475569; margin: 4px 0;"><b>💧 {get_text('tab2_water', lang)}:</b> {crop_item.get('water_requirement', 'Medium')}</p>
                    <p style="font-size: 0.82rem; color: #475569; margin: 4px 0;"><b>🧪 {get_text('tab2_fert', lang)}:</b> {', '.join(crop_item.get('optimal_fertilizers', [])[:2])}</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button(f"👉 {get_text('tab2_select_btn', lang)} ({crop_item['crop_name']})", key=f"select_btn_{crop_item['crop_name']}", use_container_width=True):
                    st.session_state.selected_crop = crop_item["crop_name"]
                    st.success(f"Selected {crop_item['crop_name']} as active crop! Synced to Fertilizer & Lifecycle Tabs.")
                    st.rerun()


# =============================================================
# TAB 3: Quantitative Fertilizer Deficit Calculator
# =============================================================
with tab3:
    st.subheader(f"{get_text('tab3_title', lang)} **{st.session_state.selected_crop}**")
    st.write(get_text("tab3_desc", lang))
    
    # Active Crop Switcher
    all_crops = [
        "Rice", "Wheat", "Barley", "Cotton", "Maize", "Sugarcane", "Groundnut", "Millets", "Sorghum",
        "Pomegranate", "Chickpea", "Kidneybeans", "Pigeonpeas", "Mothbeans", "Mungbean", "Blackgram",
        "Lentil", "Coffee", "Jute", "Coconut", "Apple", "Orange", "Papaya", "Banana", "Mango",
        "Grapes", "Watermelon", "Muskmelon", "Tomato", "Potato", "Mustard", "Soybean"
    ]
    cur_crop_idx = all_crops.index(st.session_state.selected_crop) if st.session_state.selected_crop in all_crops else 0
    
    col_fc1, col_fc2 = st.columns([2, 1])
    with col_fc1:
        chosen_crop = st.selectbox(
            get_text("tab3_crop_select", lang),
            all_crops,
            index=cur_crop_idx,
            key="tab3_crop_selector"
        )
        if chosen_crop != st.session_state.selected_crop:
            st.session_state.selected_crop = chosen_crop
            st.rerun()
            
    with col_fc2:
        st.markdown(f"""
        <div style="padding: 10px; background: #f0fdf4; border-radius: 8px; border: 1px solid #86efac; font-size: 0.82rem; margin-top: 15px;">
            <b>Active Soil Baseline:</b><br>
            • {st.session_state.soil_data['soil_type']} Soil | pH {st.session_state.soil_data['ph']:.1f}<br>
            • N-P-K: <b>{st.session_state.soil_data['nitrogen']:.0f}-{st.session_state.soil_data['phosphorus']:.0f}-{st.session_state.soil_data['potassium']:.0f}</b> kg/ha
        </div>
        """, unsafe_allow_html=True)
    
    if st.button(get_text("tab3_calc_btn", lang), type="primary", use_container_width=True):
        with st.spinner("Computing precision crop nutrient deficits & 50-kg commercial bags..."):
            manual_props = ManualSoilProperties(**st.session_state.soil_data)
            fert_req = FertilizerRecommendationRequest(
                crop_name=st.session_state.selected_crop,
                soil_properties=manual_props
            )
            st.session_state.fertilizer_data = fertilizer_service.recommend_fertilizers(fert_req)

    # Automatically compute if not present
    if not st.session_state.fertilizer_data:
        manual_props = ManualSoilProperties(**st.session_state.soil_data)
        fert_req = FertilizerRecommendationRequest(
            crop_name=st.session_state.selected_crop,
            soil_properties=manual_props
        )
        st.session_state.fertilizer_data = fertilizer_service.recommend_fertilizers(fert_req)

    if st.session_state.fertilizer_data:
        fdata = st.session_state.fertilizer_data
        target_target = fertilizer_service.crop_npk_targets.get(st.session_state.selected_crop, {"N": 120, "P": 60, "K": 60})
        
        # Display Target vs Current Supply Comparison
        st.markdown(f"#### {get_text('tab3_target_card', lang)}")
        col_t1, col_t2, col_t3 = st.columns(3)
        
        with col_t1:
            n_target = target_target["N"]
            n_supply = st.session_state.soil_data["nitrogen"]
            n_def = fdata.get("dosage_kg_per_ha", {}).get("nitrogen_deficit_kg_ha", 0)
            st.metric("Nitrogen (N)", f"{n_target} kg/ha Target", f"-{n_def} kg/ha Deficit" if n_def > 0 else "Sufficient", delta_color="inverse" if n_def > 0 else "normal")
            
        with col_t2:
            p_target = target_target["P"]
            p_supply = st.session_state.soil_data["phosphorus"]
            p_def = fdata.get("dosage_kg_per_ha", {}).get("phosphorus_deficit_kg_ha", 0)
            st.metric("Phosphorus (P)", f"{p_target} kg/ha Target", f"-{p_def} kg/ha Deficit" if p_def > 0 else "Sufficient", delta_color="inverse" if p_def > 0 else "normal")
            
        with col_t3:
            k_target = target_target["K"]
            k_supply = st.session_state.soil_data["potassium"]
            k_def = fdata.get("dosage_kg_per_ha", {}).get("potassium_deficit_kg_ha", 0)
            st.metric("Potassium (K)", f"{k_target} kg/ha Target", f"-{k_def} kg/ha Deficit" if k_def > 0 else "Sufficient", delta_color="inverse" if k_def > 0 else "normal")

        # Commercial 50-kg Bags
        st.markdown(f"#### {get_text('tab3_bags_title', lang)}")
        bags = fdata.get("commercial_bags_recommended", {})
        col_b1, col_b2, col_b3, col_b4 = st.columns(4)
        col_b1.metric("Urea (46% N)", f"{bags.get('urea_50kg_bags', 0)} bags")
        col_b2.metric("DAP (18% N, 46% P)", f"{bags.get('dap_50kg_bags', 0)} bags")
        col_b3.metric("MOP (60% K)", f"{bags.get('mop_50kg_bags', 0)} bags")
        col_b4.metric("SSP (16% P, 11% S)", f"{bags.get('ssp_50kg_bags', 0)} bags")

        col_fg1, col_fg2 = st.columns([1, 1], gap="large")
        with col_fg1:
            st.markdown(f"#### {get_text('tab3_split_title', lang)}")
            st.info(f"**Application Guidelines:**\n\n{fdata.get('application_guidelines')}")
            
            micro = fdata.get("micronutrient_advice", {})
            if micro:
                st.write("**🧪 Micronutrient & Soil Amendment Fixes:**")
                for k, v in micro.items():
                    st.warning(f"• **{k}:** {v}")

        with col_fg2:
            st.markdown(f"#### {get_text('tab3_organic_title', lang)}")
            st.success(f"{fdata.get('organic_alternatives')}")
            st.markdown(f"""
            <div class="agri-card" style="background: #f8fafc; font-size: 0.84rem;">
                <b>🌿 Bio-Fertilizer Inoculation Tips:</b><br>
                • <b>Rhizobium / Azospirillum:</b> Fixes 20-40 kg atmospheric Nitrogen/ha.<br>
                • <b>PSB (Phosphate Solubilizing Bacteria):</b> Converts locked soil phosphorus into available form.<br>
                • <b>Trichoderma viride:</b> Protects roots against damping-off, wilt, and root rot pathogens.
            </div>
            """, unsafe_allow_html=True)


# =============================================================
# TAB 4: Condensed & Easily Understandable Lifecycle Calendar
# =============================================================
with tab4:
    st.subheader(f"{get_text('tab4_title', lang)} ({st.session_state.selected_crop})")
    st.write(get_text("tab4_desc", lang))
    
    col_t1, col_t2 = st.columns([1, 2])
    with col_t1:
        sowing_input = st.date_input(get_text("tab4_sowing_label", lang), value=st.session_state.sowing_date)
        if sowing_input != st.session_state.sowing_date:
            st.session_state.sowing_date = sowing_input
            st.session_state.timeline_data = None
            st.rerun()
            
    with col_t2:
        if not st.session_state.timeline_data:
            st.session_state.timeline_data = timeline_service.generate_timeline(
                crop_name=st.session_state.selected_crop,
                sowing_date=st.session_state.sowing_date.strftime("%Y-%m-%d"),
                soil_type=st.session_state.soil_data["soil_type"]
            )
            
        tdata = st.session_state.timeline_data
        curr_day = tdata.get("current_day", 0)
        tot_days = tdata.get("total_duration_days", 120)
        progress_pct = min(1.0, max(0.0, curr_day / tot_days)) if tot_days > 0 else 0.0
        
        st.write(f"**Growth Progress:** Day {curr_day} of {tot_days} ({tdata.get('sowing_date')} ➔ {tdata.get('estimated_harvest_date')})")
        st.progress(progress_pct)

    if st.session_state.timeline_data:
        stages = st.session_state.timeline_data.get("stages", [])
        st.markdown("---")
        
        for stg in stages:
            is_active = (stg.get("status") == "current")
            card_border = "#16a34a" if is_active else "#94a3b8"
            status_badge = '<span class="badge-pill badge-green">Active Phase</span>' if is_active else '<span class="badge-pill badge-blue">Scheduled</span>'
            
            st.markdown(f"""
            <div class="timeline-card" style="border-left: 5px solid {card_border};">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <b style="font-size: 1.05rem; color: #1e293b;">📍 Stage {stg['stage_id']}: {stg['stage_name']}</b>
                    <div>
                        <span style="font-size: 0.82rem; color: #64748b; margin-right: 8px;">Day {stg['start_day']}–{stg['end_day']} ({stg.get('start_date')} to {stg.get('end_date')})</span>
                        {status_badge}
                    </div>
                </div>
                <div style="font-size: 0.85rem; color: #15803d; font-weight: 600; margin-bottom: 6px;">⚡ Critical Focus: {stg.get('critical_notes')}</div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; font-size: 0.84rem; color: #334155; margin-top: 6px;">
                    <div>
                        <b>🛠️ Key Activities:</b>
                        <ul style="margin: 2px 0 6px 16px; padding: 0;">
                            {''.join([f'<li>{act}</li>' for act in stg.get('activities', [])[:2]])}
                        </ul>
                        <b>💧 Irrigation:</b> {stg.get('irrigation_schedule')}
                    </div>
                    <div>
                        <b>🧪 Fertilizer Split:</b> {stg.get('fertilizer_advice')}<br>
                        <b>🐛 Pest Watch:</b> {stg.get('pest_disease_watch')}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)


# =============================================================
# TAB 5: Conversational AI Farm Assistant (RAG Grounded)
# =============================================================
with tab5:
    st.subheader(get_text("tab5_chat_title", lang))
    st.write(get_text("tab5_chat_desc", lang))
    
    # Display Chat History
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                st.caption(f"{get_text('tab5_chat_sources', lang)} {', '.join(msg['sources'])}")

    # Chat Input
    if user_prompt := st.chat_input(get_text("tab5_chat_placeholder", lang)):
        st.session_state.chat_messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Searching agricultural knowledge base & synthesizing guidance..."):
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
                    st.caption(f"{get_text('tab5_chat_sources', lang)} {', '.join(ans_obj['grounded_sources'])}")
                    
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": ans_obj["answer"],
                    "sources": ans_obj.get("grounded_sources", [])
                })
