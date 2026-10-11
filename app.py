"""
app.py — Main Interactive Streamlit Application for PulseVision.
Contactless Heart Rate Monitoring & Analytics System using Remote Photoplethysmography (rPPG).
"""

from datetime import datetime
import os
import time
import cv2
import numpy as np
import pandas as pd
import streamlit as st

# Configure Streamlit page layout and metadata
st.set_page_config(
    page_title="PulseVision — Contactless Heart Rate Analytics",
    page_icon="💓",
    layout="wide",
    initial_sidebar_state="expanded"
)

from src.analytics import (
    create_filtered_pulse_chart,
    create_frequency_spectrum_chart,
    create_heart_rate_trend_chart,
    create_raw_color_chart,
    create_session_comparison_chart,
    create_sqi_trend_chart,
)
from src.camera import CameraManager
from src.config import (
    ALG_CHROM,
    ALG_GREEN,
    ALG_POS,
    ALG_ADAPTIVE,
    APP_NAME,
    APP_SUBTITLE,
    APP_VERSION,
    ASSETS_DIR,
    AVAILABLE_ALGORITHMS,
    BPM_MAX,
    BPM_MIN,
    DEFAULT_ALGORITHM,
    DEFAULT_CAMERA_INDEX,
    DEFAULT_USER_NAME,
    DEFAULT_USER_AGE,
    DEFAULT_USER_GENDER,
    DEFAULT_ACTIVITY_STATE,
    GENDER_OPTIONS,
    ACTIVITY_OPTIONS,
    CARDIAC_ZONES,
    get_max_heart_rate,
    get_heart_rate_zone,
)
from src.database import DatabaseManager
from src.face_tracking import FaceTracker
from src.pulse_processing import compare_all_algorithms
from src.report_generator import ReportGenerator
from src.roi_extraction import ROIExtractor
from src.session_manager import SessionManager, SessionState


# ---------------------------------------------------------
# Load Custom CSS Styling
# ---------------------------------------------------------
def load_custom_css():
    css_path = ASSETS_DIR / "styles.css"
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_custom_css()


# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "session_manager" not in st.session_state:
    st.session_state.session_manager = SessionManager()

if "camera_manager" not in st.session_state:
    st.session_state.camera_manager = None

if "face_tracker" not in st.session_state:
    st.session_state.face_tracker = None

if "db_manager" not in st.session_state:
    st.session_state.db_manager = DatabaseManager()

if "selected_cam_idx" not in st.session_state:
    st.session_state.selected_cam_idx = DEFAULT_CAMERA_INDEX

if "selected_algorithm" not in st.session_state:
    st.session_state.selected_algorithm = DEFAULT_ALGORITHM

if "selected_duration" not in st.session_state:
    st.session_state.selected_duration = 60.0

if "auto_save" not in st.session_state:
    st.session_state.auto_save = True

if "monitoring_running" not in st.session_state:
    st.session_state.monitoring_running = False

# User Demographic & Health Profile in session state
if "user_name" not in st.session_state:
    st.session_state.user_name = DEFAULT_USER_NAME

if "user_age" not in st.session_state:
    st.session_state.user_age = DEFAULT_USER_AGE

if "user_gender" not in st.session_state:
    st.session_state.user_gender = DEFAULT_USER_GENDER

if "activity_state" not in st.session_state:
    st.session_state.activity_state = DEFAULT_ACTIVITY_STATE

if "current_page" not in st.session_state:
    st.session_state.current_page = "🏠 Dashboard"


# ---------------------------------------------------------
# Sidebar Navigation & Demographic Controls
# ---------------------------------------------------------
st.sidebar.markdown(f"""
    <div style="padding: 10px 0 16px 0;">
        <span style="font-size: 1.65rem; font-weight: 800; color: #00F2FE; letter-spacing: -0.5px;">💓 {APP_NAME}</span><br/>
        <span style="font-size: 0.8rem; color: #94A3B8;">{APP_SUBTITLE}</span>
    </div>
""", unsafe_allow_html=True)

nav_options = [
    "🏠 Dashboard",
    "💓 Live Monitor",
    "📊 Analytics",
    "📜 Session History",
    "📄 Reports",
    "⚙️ Settings & Privacy"
]

page_selection = st.sidebar.radio(
    "Navigation",
    nav_options,
    index=nav_options.index(st.session_state.current_page) if st.session_state.current_page in nav_options else 0,
    key="nav_radio"
)
st.session_state.current_page = page_selection

st.sidebar.markdown("---")

# User Demographic Quick-Profile Widget in Sidebar
max_hr_val = get_max_heart_rate(st.session_state.user_age)
st.sidebar.markdown(f"""
    <div class="metric-card" style="padding: 0.9rem; margin-bottom: 0.8rem;">
        <div class="metric-label" style="font-size: 0.75rem;">👤 Participant Profile</div>
        <div style="font-size: 0.95rem; font-weight: 700; color: #F8FAFC;">{st.session_state.user_name}</div>
        <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 2px;">
            Age: <b style="color: #00F2FE;">{st.session_state.user_age} yrs</b> • {st.session_state.user_gender}
        </div>
        <div style="font-size: 0.78rem; color: #CBD5E1; margin-top: 4px;">
            Context: {st.session_state.activity_state}
        </div>
        <div style="margin-top: 6px; padding-top: 6px; border-top: 1px solid rgba(255,255,255,0.08); font-size: 0.78rem; color: #94A3B8;">
            Predicted Max HR: <b style="color: #EF4444;">{max_hr_val} BPM</b>
        </div>
    </div>
""", unsafe_allow_html=True)

with st.sidebar.expander("✏️ Quick Edit Profile", expanded=False):
    sb_name = st.text_input("Name / ID", value=st.session_state.user_name, key="sb_name_input")
    sb_age = st.number_input("Age (Years)", min_value=5, max_value=120, value=int(st.session_state.user_age), step=1, key="sb_age_input")
    sb_gender = st.selectbox("Biological Gender", GENDER_OPTIONS, index=GENDER_OPTIONS.index(st.session_state.user_gender) if st.session_state.user_gender in GENDER_OPTIONS else 0, key="sb_gender_input")
    sb_act = st.selectbox("Activity State", ACTIVITY_OPTIONS, index=ACTIVITY_OPTIONS.index(st.session_state.activity_state) if st.session_state.activity_state in ACTIVITY_OPTIONS else 0, key="sb_act_input")
    if st.button("Apply Profile Changes", key="sb_apply_btn", use_container_width=True):
        st.session_state.user_name = sb_name.strip() or DEFAULT_USER_NAME
        st.session_state.user_age = int(sb_age)
        st.session_state.user_gender = sb_gender
        st.session_state.activity_state = sb_act
        st.rerun()

st.sidebar.markdown("---")
# Quick Status indicator in sidebar
sm = st.session_state.session_manager
status_color = "#10B981" if sm.state == SessionState.MONITORING else ("#F59E0B" if sm.state == SessionState.PAUSED else "#94A3B8")
st.sidebar.markdown(f"""
    <div style="font-size: 0.82rem; color: #94A3B8;">
        <b>Engine Status:</b> <span style="color: {status_color}; font-weight: 700;">{sm.state}</span><br/>
        <b>Algorithm:</b> <span style="color: #00F2FE;">{st.session_state.selected_algorithm}</span><br/>
        <b>Version:</b> v{APP_VERSION}
    </div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.caption("⚠️ **Notice:** Experimental wellness and computer-vision tool. Not a medical diagnostic device.")


# ---------------------------------------------------------
# HELPER: Render Cardiac Training Zone Widget
# ---------------------------------------------------------
def render_zone_widget(bpm_val: float, user_age: int) -> str:
    """Generates HTML string for the interactive cardiovascular zone meter."""
    zone_data = get_heart_rate_zone(bpm_val, user_age)
    zone_name = zone_data["zone_name"]
    zone_pct = zone_data["percent_max"]
    zone_col = zone_data["color"]
    zone_desc = zone_data["description"]
    max_hr = get_max_heart_rate(user_age)

    is_z1 = "active" if "Zone 1" in zone_name else ""
    is_z2 = "active" if "Zone 2" in zone_name else ""
    is_z3 = "active" if "Zone 3" in zone_name else ""
    is_z4 = "active" if "Zone 4" in zone_name else ""
    is_z5 = "active" if "Zone 5" in zone_name else ""

    pct_str = f"{zone_pct:.0f}%" if zone_pct > 0 else "--%"

    return f"""
        <div class="zone-container">
            <div class="zone-label-row">
                <span><b>Cardio Zone:</b> <span style="color: {zone_col}; font-weight: 700;">{zone_name}</span></span>
                <span style="color: #94A3B8; font-size: 0.78rem;">{pct_str} of Max ({max_hr} BPM)</span>
            </div>
            <div class="zone-meter">
                <div class="zone-segment zone-seg-1 {is_z1}" title="Zone 1: Warm Up (50-60%)"></div>
                <div class="zone-segment zone-seg-2 {is_z2}" title="Zone 2: Fat Burn (60-70%)"></div>
                <div class="zone-segment zone-seg-3 {is_z3}" title="Zone 3: Aerobic (70-80%)"></div>
                <div class="zone-segment zone-seg-4 {is_z4}" title="Zone 4: Anaerobic (80-90%)"></div>
                <div class="zone-segment zone-seg-5 {is_z5}" title="Zone 5: Maximum (90-100%)"></div>
            </div>
            <div style="font-size: 0.78rem; color: #CBD5E1; margin-top: 5px; line-height: 1.3;">
                {zone_desc}
            </div>
        </div>
    """


# ---------------------------------------------------------
# PAGE 1: DASHBOARD (Welcome, Profile Onboarding & Setup)
# ---------------------------------------------------------
if page_selection == "🏠 Dashboard":
    st.markdown('<div class="brand-title">💓 PulseVision</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-subtitle">Contactless Remote Photoplethysmography (rPPG) & Cardiovascular Analytics</div>', unsafe_allow_html=True)

    col_hero, col_side = st.columns([1.6, 1.4])

    with col_hero:
        st.markdown("""
        ### What is Contactless Heart Rate Estimation?
        With every heartbeat, oxygenated blood surges through your arterial network. The capillary micro-vessels 
        beneath your facial skin expand and contract cyclically, creating **microscopic, sub-perceptual changes in light absorption**.
        
        **PulseVision** extracts these cardiac optical pulses directly through an ordinary webcam:
        1. **MediaPipe Face Mesh**: Sub-millimeter landmark tracking isolating optimal forehead and cheek dermis ROIs.
        2. **Optical Decomposition**: State-of-the-art **ADAPTIVE Ensemble**, **POS** (Plane-Orthogonal), and **CHROM** color algorithms.
        3. **DSP Filtration**: Zero-phase Butterworth filtering targeting physiological human cardiac bounds (42–210 BPM).
        4. **Cardiovascular Analytics**: Real-time spectral density peak finding, SNR confidence scoring, and age-adapted heart rate training zones.
        """)

        # User Demographic Setup Card
        st.markdown("""
        <div class="metric-card" style="border-color: rgba(0, 242, 254, 0.4);">
            <div class="metric-label" style="color: #00F2FE;">👤 Step 1: Participant Health Profile & Calibration</div>
            <div style="font-size: 0.85rem; color: #94A3B8; margin-bottom: 12px;">
                Your age and physical activity context personalize your maximum heart rate and target cardiovascular zones.
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.container():
            f_col1, f_col2 = st.columns(2)
            with f_col1:
                input_name = st.text_input("Participant Name / Subject ID", value=st.session_state.user_name, max_chars=40)
                input_age = st.number_input("Age (Years)", min_value=5, max_value=120, value=int(st.session_state.user_age), step=1, help="Used for age-predicted maximum heart rate: 220 - Age")
            with f_col2:
                input_gender = st.selectbox("Biological Gender", GENDER_OPTIONS, index=GENDER_OPTIONS.index(st.session_state.user_gender) if st.session_state.user_gender in GENDER_OPTIONS else 0)
                input_activity = st.selectbox("Current Activity Context", ACTIVITY_OPTIONS, index=ACTIVITY_OPTIONS.index(st.session_state.activity_state) if st.session_state.activity_state in ACTIVITY_OPTIONS else 0)

            st.markdown("##### ⏱️ Measurement Parameters")
            p_col1, p_col2 = st.columns(2)
            with p_col1:
                sel_dur_dash = st.selectbox("Measurement Duration", [30, 60, 120, "Continuous"], index=1, key="dash_dur")
                dur_val_dash = 0.0 if sel_dur_dash == "Continuous" else float(sel_dur_dash)
            with p_col2:
                sel_alg_dash = st.selectbox("rPPG Extraction Algorithm", AVAILABLE_ALGORITHMS, index=AVAILABLE_ALGORITHMS.index(st.session_state.selected_algorithm), key="dash_alg")

            start_btn = st.button("🚀 Save Profile & Launch Live Monitor", type="primary", use_container_width=True)
            if start_btn:
                # Save profile details
                st.session_state.user_name = input_name.strip() or DEFAULT_USER_NAME
                st.session_state.user_age = int(input_age)
                st.session_state.user_gender = input_gender
                st.session_state.activity_state = input_activity
                st.session_state.selected_duration = dur_val_dash
                st.session_state.selected_algorithm = sel_alg_dash

                # Start session manager
                st.session_state.session_manager.start_session(
                    user_label=st.session_state.user_name,
                    user_age=st.session_state.user_age,
                    user_gender=st.session_state.user_gender,
                    activity_state=st.session_state.activity_state,
                    duration_sec=dur_val_dash,
                    algorithm=sel_alg_dash
                )
                st.session_state.monitoring_running = True
                st.session_state.current_page = "💓 Live Monitor"
                st.rerun()

    with col_side:
        # Physiological Target Heart Rate Zones Preview Card
        curr_age = int(input_age if 'input_age' in locals() else st.session_state.user_age)
        calc_max_hr = get_max_heart_rate(curr_age)

        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: #00F2FE; margin-top: 0; display: flex; align-items: center; gap: 8px;">
                🎯 Personalized Cardiac Reference
            </h4>
            <div style="font-size: 0.85rem; color: #CBD5E1; margin-bottom: 12px;">
                Calculated based on <b>Age {curr_age}</b> (Fox Formula: HR<sub>max</sub> = 220 - Age):
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; background: rgba(0,0,0,0.25); padding: 10px 14px; border-radius: 10px; margin-bottom: 14px;">
                <span style="font-size: 0.9rem; color: #94A3B8; font-weight: 600;">Max Heart Rate:</span>
                <span style="font-size: 1.4rem; color: #EF4444; font-weight: 800; font-family: 'JetBrains Mono', monospace;">{calc_max_hr} BPM</span>
            </div>
            <div style="font-size: 0.8rem; line-height: 1.7; color: #CBD5E1;">
                <div style="display: flex; justify-content: space-between;"><span style="color: #94A3B8;">🛋️ Resting Baseline:</span> <b>60 – 100 BPM</b></div>
                <div style="display: flex; justify-content: space-between;"><span style="color: #48CAE4;">🔵 Zone 1 (Warm Up 50-60%):</span> <b>{int(calc_max_hr * 0.50)} – {int(calc_max_hr * 0.60)} BPM</b></div>
                <div style="display: flex; justify-content: space-between;"><span style="color: #10B981;">🟢 Zone 2 (Fat Burn 60-70%):</span> <b>{int(calc_max_hr * 0.60)} – {int(calc_max_hr * 0.70)} BPM</b></div>
                <div style="display: flex; justify-content: space-between;"><span style="color: #F59E0B;">🟡 Zone 3 (Aerobic 70-80%):</span> <b>{int(calc_max_hr * 0.70)} – {int(calc_max_hr * 0.80)} BPM</b></div>
                <div style="display: flex; justify-content: space-between;"><span style="color: #F97316;">🟠 Zone 4 (Anaerobic 80-90%):</span> <b>{int(calc_max_hr * 0.80)} – {int(calc_max_hr * 0.90)} BPM</b></div>
                <div style="display: flex; justify-content: space-between;"><span style="color: #EF4444;">🔴 Zone 5 (Peak 90-100%):</span> <b>{int(calc_max_hr * 0.90)} – {calc_max_hr} BPM</b></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="metric-card">
            <h4 style="color: #10B981; margin-top: 0;">💡 Acquisition Best Practices</h4>
            <ul style="color: #CBD5E1; font-size: 0.85rem; line-height: 1.6; margin-bottom: 0; padding-left: 1.2rem;">
                <li><b>Direct Lighting:</b> Sit facing a window or soft desk lamp. Minimize backlighting and flickering bulbs.</li>
                <li><b>Face Stability:</b> Rest comfortably. Avoid head shaking, speaking, or deep chewing motions during the 30-60s window.</li>
                <li><b>Camera Distance:</b> Keep your face ~50–90 cm from the webcam so the forehead and cheeks fill the ROI boundaries.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="metric-card">
            <h4 style="color: #00F2FE; margin-top: 0;">🔒 100% In-Memory Privacy</h4>
            <p style="color: #94A3B8; font-size: 0.82rem; line-height: 1.5; margin-bottom: 0;">
                Frames are processed strictly in volatile RAM. No facial imagery or video streams are ever stored on disk or transmitted across the internet.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer-box">
        <b>Regulatory & Non-Diagnostic Medical Disclaimer:</b><br/>
        PulseVision is strictly an experimental educational computer-vision research and wellness tool. It is <b>not a medical diagnostic device</b> 
        and does not diagnose, treat, monitor, cure, or prevent any illness or cardiovascular condition. Do not use its output for clinical evaluations.
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------
# PAGE 2: LIVE MONITOR
# ---------------------------------------------------------
elif page_selection == "💓 Live Monitor":
    sm = st.session_state.session_manager

    # Top Header & User Demographic Banner
    st.markdown(f"""
        <div style="display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; margin-bottom: 0.5rem;">
            <div>
                <span class="brand-title" style="font-size: 2.0rem;">💓 Live Heart Rate Monitor</span>
            </div>
            <div style="display: flex; gap: 8px; align-items: center; flex-wrap: wrap;">
                <span class="user-pill">👤 <b>{st.session_state.user_name}</b> ({st.session_state.user_age}y, {st.session_state.user_gender})</span>
                <span class="user-pill" style="border-color: rgba(16, 185, 129, 0.4); color: #10B981;">{st.session_state.activity_state}</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Top Control Bar
    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4, ctrl_col5 = st.columns([1, 1, 1, 1, 2])

    with ctrl_col1:
        if sm.state in [SessionState.IDLE, SessionState.COMPLETED]:
            if st.button("▶️ Start", type="primary", use_container_width=True):
                sm.start_session(
                    user_label=st.session_state.user_name,
                    user_age=st.session_state.user_age,
                    user_gender=st.session_state.user_gender,
                    activity_state=st.session_state.activity_state,
                    duration_sec=st.session_state.selected_duration,
                    algorithm=st.session_state.selected_algorithm
                )
                st.session_state.monitoring_running = True
                st.rerun()
        else:
            if sm.state == SessionState.MONITORING:
                if st.button("⏸️ Pause", use_container_width=True):
                    sm.pause_session()
                    st.rerun()
            elif sm.state == SessionState.PAUSED:
                if st.button("▶️ Resume", type="primary", use_container_width=True):
                    sm.resume_session()
                    st.rerun()

    with ctrl_col2:
        stop_disabled = (sm.state in [SessionState.IDLE, SessionState.COMPLETED])
        if st.button("⏹️ Stop", disabled=stop_disabled, use_container_width=True):
            sm.stop_session()
            st.session_state.monitoring_running = False
            if st.session_state.auto_save:
                stats = sm.get_session_statistics()
                st.session_state.db_manager.save_session(
                    stats, sm.history_bpm_records, sm.latest_pulse_waveform
                )
            st.rerun()

    with ctrl_col3:
        if st.button("🔄 Restart", use_container_width=True):
            sm.restart_session()
            st.session_state.monitoring_running = True
            st.rerun()

    with ctrl_col4:
        if st.button("🗑️ Clear", use_container_width=True):
            sm.clear_session()
            st.session_state.monitoring_running = False
            st.rerun()

    with ctrl_col5:
        # Progress / Timer
        stats = sm.get_session_statistics()
        target_dur = stats.get("target_duration_sec", 60.0)
        curr_dur = stats.get("duration_sec", 0.0)
        if target_dur > 0:
            prog_val = min(1.0, max(0.0, curr_dur / target_dur))
            st.progress(prog_val, text=f"Progress: {curr_dur:.1f}s / {target_dur:.0f}s ({sm.state})")
        else:
            st.markdown(f"**Elapsed Time:** {curr_dur:.1f}s (Continuous)")

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # Main Live Layout: Video & Primary Metrics
    view_col1, view_col2 = st.columns([1.35, 1.05])

    with view_col1:
        st.markdown("##### 📹 Live Camera & Dermal ROI Tracking")
        video_placeholder = st.empty()

    with view_col2:
        st.markdown("##### 💓 Real-Time Cardiac Telemetry")
        metrics_placeholder = st.empty()

    # Waveform and Spectrum Placeholders below
    st.markdown("---")
    wave_col, spec_col = st.columns(2)
    with wave_col:
        st.markdown("##### 📈 Real-Time Optical Pulse Waveform")
        wave_placeholder = st.empty()
    with spec_col:
        st.markdown("##### 📊 Real-Time Power Spectral Density (PSD)")
        spec_placeholder = st.empty()

    # Active Live Capture Loop
    if st.session_state.monitoring_running and sm.state == SessionState.MONITORING:
        # Ensure camera and tracker initialized
        if st.session_state.camera_manager is None:
            st.session_state.camera_manager = CameraManager(camera_index=st.session_state.selected_cam_idx)
            st.session_state.camera_manager.start()

        if st.session_state.face_tracker is None:
            st.session_state.face_tracker = FaceTracker()

        cam = st.session_state.camera_manager
        tracker = st.session_state.face_tracker

        # Run loop for a set burst of frames, then let Streamlit breathe
        for _ in range(25):
            if sm.state != SessionState.MONITORING:
                break

            ok, frame, ts, is_new = cam.read_frame()
            if not ok or frame is None:
                time.sleep(0.02)
                continue

            # Mirror horizontally for natural mirror interaction
            frame = cv2.flip(frame, 1)

            # 1. Detect face and track motion
            face_res = tracker.process_frame(frame)

            # 2. Extract skin color from forehead & cheeks
            rgb_mean, skin_ratio = ROIExtractor.extract_color_means(frame, face_res)

            # 3. Feed sample into session manager
            # Only ingest fresh camera frames to keep precise 30Hz physical sampling rate
            if is_new or face_res.detected:
                sm.add_sample(
                    current_time=ts,
                    raw_rgb=rgb_mean,
                    face_detected=face_res.detected,
                    motion_velocity=face_res.motion_velocity,
                    skin_ratio=skin_ratio
                )

            # 4. Draw HUD overlays on frame
            annotated_frame = ROIExtractor.draw_roi_overlays(frame, face_res)
            rgb_display = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(rgb_display, channels="RGB", use_container_width=True)

            # 5. Render Primary Metric Card and Statistics
            est = sm.latest_estimate
            sqi = sm.latest_quality
            sess_stats = sm.get_session_statistics()

            # Responsive BPM formatting — never stuck on "--" once calculated
            if est.bpm is not None and est.bpm > 0:
                if est.is_valid:
                    bpm_display = f"{est.bpm:.0f}"
                    pulse_icon = '<span class="pulse-heart">❤️</span>'
                    status_badge = f'<span class="badge-status badge-excellent">ACTIVE MONITORING</span>'
                else:
                    bpm_display = f"{est.bpm:.0f}"
                    pulse_icon = '<span class="pulse-heart" style="color: #48CAE4;">🫀</span>'
                    status_badge = f'<span class="badge-status badge-warning">REFINING ({est.quality_score:.0f}%)</span>'
            else:
                bpm_display = "--"
                pulse_icon = '🤍'
                buf_count = len(sm.buf_timestamps)
                status_badge = f'<span class="badge-status badge-warning">ACQUIRING ({buf_count}/45 FRAMES)</span>'

            # SQI Badge class
            sqi_badge_class = "badge-excellent" if sqi.status_category == "EXCELLENT" else (
                "badge-good" if sqi.status_category == "GOOD" else (
                    "badge-warning" if sqi.status_category in ["FAIR", "COLLECTING_BUFFER"] else "badge-alert"
                )
            )

            # Cardiovascular Zone HTML
            zone_html = render_zone_widget(est.bpm, st.session_state.user_age)

            metrics_placeholder.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Estimated Heart Rate</div>
                    <div class="metric-value-huge">
                        {pulse_icon} {bpm_display} <span class="metric-unit">BPM</span>
                    </div>
                    <div style="margin-top: 10px; display: flex; gap: 8px; flex-wrap: wrap;">
                        {status_badge}
                        <span class="badge-status {sqi_badge_class}">SQI: {sqi.status_category} ({sqi.sqi:.0f}%)</span>
                    </div>
                    <div style="margin-top: 8px; font-size: 0.84rem; color: #CBD5E1;">
                        {sqi.guidance if sqi.guidance else est.message}
                    </div>
                    {zone_html}
                </div>

                <div class="metric-card">
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
                        <div>
                            <div class="metric-label">Session Avg</div>
                            <div class="metric-value">{sess_stats['avg_bpm'] or '--'} <span style="font-size: 0.9rem; color: #94A3B8;">BPM</span></div>
                        </div>
                        <div>
                            <div class="metric-label">Valid Readings</div>
                            <div class="metric-value">{sess_stats['valid_count']}</div>
                        </div>
                        <div>
                            <div class="metric-label">Min - Max</div>
                            <div class="metric-value" style="font-size: 1.3rem;">{sess_stats['min_bpm'] or '--'} - {sess_stats['max_bpm'] or '--'}</div>
                        </div>
                        <div>
                            <div class="metric-label">Signal Usable</div>
                            <div class="metric-value" style="font-size: 1.3rem;">{sess_stats['usable_pct']}%</div>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # 6. Render Live Pulse Waveform
            if len(sm.latest_pulse_waveform) > 10:
                fig_wave = create_filtered_pulse_chart(sm.latest_pulse_waveform, fs=cam.get_fps())
                wave_placeholder.plotly_chart(fig_wave, use_container_width=True, key=f"live_wave_{time.time()}")

            # 7. Render Frequency Spectrum
            if len(est.frequencies) > 0 and len(est.power_spectrum) > 0:
                fig_spec = create_frequency_spectrum_chart(est.frequencies, est.power_spectrum, peak_bpm=est.bpm)
                spec_placeholder.plotly_chart(fig_spec, use_container_width=True, key=f"live_spec_{time.time()}")

            time.sleep(0.015)

        # Trigger automatic rerun to keep live monitor flowing
        if sm.state == SessionState.MONITORING:
            st.rerun()

    else:
        # Static display when not running
        video_placeholder.info("Camera is currently idle. Click **▶️ Start** above to begin acquisition.")
        sess_stats = sm.get_session_statistics()
        est = sm.latest_estimate
        
        if est.bpm is not None and est.bpm > 0:
            bpm_display = f"{est.bpm:.0f}"
            pulse_icon = '❤️' if est.is_valid else '🫀'
        else:
            bpm_display = "--"
            pulse_icon = '🤍'

        zone_html = render_zone_widget(est.bpm, st.session_state.user_age)

        metrics_placeholder.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Heart Rate</div>
                <div class="metric-value-huge">{pulse_icon} {bpm_display} <span class="metric-unit">BPM</span></div>
                <div style="margin-top: 10px;">
                    <span class="badge-status badge-warning">STATUS: {sm.state}</span>
                </div>
                {zone_html}
            </div>
            <div class="metric-card">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
                    <div><div class="metric-label">Session Avg</div><div class="metric-value">{sess_stats['avg_bpm'] or '--'}</div></div>
                    <div><div class="metric-label">Valid Points</div><div class="metric-value">{sess_stats['valid_count']}</div></div>
                    <div><div class="metric-label">Observed Range</div><div class="metric-value" style="font-size: 1.2rem;">{sess_stats['min_bpm'] or '--'} - {sess_stats['max_bpm'] or '--'}</div></div>
                    <div><div class="metric-label">Signal Usable</div><div class="metric-value" style="font-size: 1.2rem;">{sess_stats['usable_pct']}%</div></div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        if len(sm.latest_pulse_waveform) > 10:
            fig_wave = create_filtered_pulse_chart(sm.latest_pulse_waveform)
            wave_placeholder.plotly_chart(fig_wave, use_container_width=True)
        else:
            wave_placeholder.caption("Pulse waveform will appear here once monitoring begins.")

        if len(sm.latest_estimate.frequencies) > 0:
            fig_spec = create_frequency_spectrum_chart(sm.latest_estimate.frequencies, sm.latest_estimate.power_spectrum, peak_bpm=est.bpm)
            spec_placeholder.plotly_chart(fig_spec, use_container_width=True)
        else:
            spec_placeholder.caption("Power spectrum will appear here once monitoring begins.")


# ---------------------------------------------------------
# PAGE 3: ANALYTICS
# ---------------------------------------------------------
elif page_selection == "📊 Analytics":
    st.markdown('<div class="brand-title">Detailed Signal Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-subtitle">In-Depth Signal Processing, Waveforms, Demographics, and Algorithm Comparisons</div>', unsafe_allow_html=True)

    sm = st.session_state.session_manager
    stats = sm.get_session_statistics()

    # Participant Profile Card
    p_age = stats.get("user_age", st.session_state.user_age)
    p_gen = stats.get("user_gender", st.session_state.user_gender)
    p_act = stats.get("activity_state", st.session_state.activity_state)
    p_max = stats.get("max_hr", get_max_heart_rate(p_age))
    p_zone = stats.get("target_zone", "Resting")

    st.markdown(f"""
    <div class="metric-card" style="margin-bottom: 1.2rem;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div>
                <span style="font-size: 0.8rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 1px;">Participant Profile:</span>
                <span style="font-size: 1.15rem; font-weight: 700; color: #00F2FE; margin-left: 8px;">{stats.get('user_label', st.session_state.user_name)}</span>
            </div>
            <div style="display: flex; gap: 8px; flex-wrap: wrap; margin-top: 4px;">
                <span class="user-pill">Age: <b>{p_age} yrs</b></span>
                <span class="user-pill">{p_gen}</span>
                <span class="user-pill" style="border-color: rgba(16, 185, 129, 0.4); color: #10B981;">{p_act}</span>
                <span class="user-pill" style="border-color: rgba(239, 68, 68, 0.4); color: #EF4444;">HR<sub>max</sub>: <b>{p_max} BPM</b></span>
                <span class="badge-status badge-good">Dominant Zone: <b>{p_zone}</b></span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Executive Summary Cards
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Average BPM</div><div class="metric-value" style="color: #00F2FE;">{stats['avg_bpm'] or '--'}</div></div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Min BPM</div><div class="metric-value">{stats['min_bpm'] or '--'}</div></div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Max BPM</div><div class="metric-value">{stats['max_bpm'] or '--'}</div></div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Duration</div><div class="metric-value">{stats['duration_sec']:.1f}s</div></div>""", unsafe_allow_html=True)
    with m5:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Usable Signal</div><div class="metric-value" style="color: #10B981;">{stats['usable_pct']}%</div></div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Tabs for analytics views
    t1, t2, t3, t4 = st.tabs(["📈 Heart Rate & Waveforms", "🌈 Raw RGB Channels", "🔍 Frequency & Quality", "⚙️ Algorithm Comparison"])

    with t1:
        st.markdown("#### Heart Rate Trend & Optical Pulse")
        fig_trend = create_heart_rate_trend_chart(sm.history_bpm_records, session_avg_bpm=stats.get("avg_bpm"))
        st.plotly_chart(fig_trend, use_container_width=True)

        if len(sm.latest_pulse_waveform) > 0:
            fig_wave = create_filtered_pulse_chart(sm.latest_pulse_waveform)
            st.plotly_chart(fig_wave, use_container_width=True)

    with t2:
        st.markdown("#### Extracted Facial Skin Color Time-Series")
        ch_filter = st.radio("Channel View", ["All", "Red", "Green", "Blue"], horizontal=True)
        fig_raw = create_raw_color_chart(
            sm.history_elapsed_sec, sm.history_raw_r, sm.history_raw_g, sm.history_raw_b, channel_selection=ch_filter
        )
        st.plotly_chart(fig_raw, use_container_width=True)

    with t3:
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown("#### Power Spectral Density")
            est = sm.latest_estimate
            fig_psd = create_frequency_spectrum_chart(est.frequencies, est.power_spectrum, peak_bpm=est.bpm)
            st.plotly_chart(fig_psd, use_container_width=True)

        with col_s2:
            st.markdown("#### Signal Quality Index (SQI) Trend")
            fig_sqi = create_sqi_trend_chart(sm.history_sqi_records)
            st.plotly_chart(fig_sqi, use_container_width=True)

    with t4:
        st.markdown("#### Direct Algorithm Comparison (Adaptive vs Green vs CHROM vs POS)")
        st.caption("Processes the current session's recorded RGB color buffer through all extraction algorithms simultaneously.")

        if len(sm.history_raw_g) >= 45:
            r_arr = np.array(sm.history_raw_r)
            g_arr = np.array(sm.history_raw_g)
            b_arr = np.array(sm.history_raw_b)
            total_t = max(1.0, sm.history_elapsed_sec[-1] - sm.history_elapsed_sec[0])
            fs_est = len(r_arr) / total_t

            comp_results = compare_all_algorithms(r_arr, g_arr, b_arr, fs_est)
            c1, c2, c3 = st.columns(3)
            with c1:
                st.plotly_chart(create_filtered_pulse_chart(comp_results[ALG_GREEN], fs=fs_est, title_suffix="[GREEN]"), use_container_width=True)
            with c2:
                st.plotly_chart(create_filtered_pulse_chart(comp_results[ALG_CHROM], fs=fs_est, title_suffix="[CHROM]"), use_container_width=True)
            with c3:
                st.plotly_chart(create_filtered_pulse_chart(comp_results[ALG_POS], fs=fs_est, title_suffix="[POS]"), use_container_width=True)
        else:
            st.info("Insufficient data recorded yet to run comparison. Complete at least 5 seconds of monitoring first.")


# ---------------------------------------------------------
# PAGE 4: SESSION HISTORY
# ---------------------------------------------------------
elif page_selection == "📜 Session History":
    st.markdown('<div class="brand-title">Session History</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-subtitle">Persistent SQLite Session Records & Comparative Demographic Analysis</div>', unsafe_allow_html=True)

    db = st.session_state.db_manager

    # Top search & management controls
    h_col1, h_col2, h_col3 = st.columns([2, 1, 1])
    with h_col1:
        search_query = st.text_input("🔍 Search Sessions by Participant, Label, or ID", placeholder="e.g. Guest User, Morning, PV-...")
    with h_col2:
        st.write("")
        st.write("")
        if st.button("💾 Save Active Session", use_container_width=True):
            sm = st.session_state.session_manager
            stats = sm.get_session_statistics()
            if stats.get("valid_count", 0) > 0 or stats.get("duration_sec", 0) > 2:
                db.save_session(stats, sm.history_bpm_records, sm.latest_pulse_waveform)
                st.success("Session saved successfully!")
                st.rerun()
            else:
                st.warning("No active session data to save.")
    with h_col3:
        st.write("")
        st.write("")
        if st.button("🗑️ Clear All History", type="secondary", use_container_width=True):
            db.clear_all_sessions()
            st.success("All session records deleted.")
            st.rerun()

    # Query session records
    sessions = db.list_sessions(search=search_query)

    if not sessions:
        st.info("No saved sessions found. Complete a monitoring session and save it to review history here.")
    else:
        # Build comprehensive dataframe
        df_records = []
        for s in sessions:
            df_records.append({
                "Session ID": s.get("session_id"),
                "Participant": s.get("user_label"),
                "Age": s.get("user_age"),
                "Gender": s.get("user_gender"),
                "Activity State": s.get("activity_state"),
                "Target Zone": s.get("target_zone"),
                "Start Time": str(s.get("start_time"))[:19],
                "Duration (s)": s.get("duration_sec"),
                "Avg BPM": s.get("avg_bpm"),
                "Min - Max BPM": f"{s.get('min_bpm') or '--'} - {s.get('max_bpm') or '--'}",
                "Usable (%)": s.get("usable_pct"),
            })
        df_display = pd.DataFrame(df_records)
        st.dataframe(df_display, use_container_width=True, hide_index=True)

        st.markdown("---")
        # Session Detail & Comparison
        st.markdown("#### 🔍 Session Exploration & Multi-Session Comparison")
        session_ids = [s["session_id"] for s in sessions]
        selected_ids = st.multiselect("Select Sessions to Compare", session_ids, default=[session_ids[0]] if session_ids else [])

        if selected_ids:
            full_sessions = [db.get_session(sid) for sid in selected_ids if db.get_session(sid) is not None]
            if len(selected_ids) == 1:
                sess = full_sessions[0]
                s_age = sess.get("user_age", 25)
                s_gen = sess.get("user_gender", "N/A")
                s_act = sess.get("activity_state", "N/A")
                s_zone = sess.get("target_zone", "N/A")
                s_max = sess.get("max_hr", get_max_heart_rate(s_age))

                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size: 1.15rem; font-weight: 700; color: #00F2FE;">
                        {sess.get('user_label')} <span style="font-size: 0.85rem; color: #94A3B8;">(`{sess.get('session_id')}`)</span>
                    </div>
                    <div style="margin-top: 6px; display: flex; gap: 8px; flex-wrap: wrap;">
                        <span class="user-pill">Age: <b>{s_age} yrs</b></span>
                        <span class="user-pill">{s_gen}</span>
                        <span class="user-pill">{s_act}</span>
                        <span class="user-pill" style="color: #EF4444;">HR<sub>max</sub>: <b>{s_max} BPM</b></span>
                        <span class="badge-status badge-good">Target Zone: <b>{s_zone}</b></span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                fig_single = create_heart_rate_trend_chart(sess.get("bpm_series", []), session_avg_bpm=sess.get("avg_bpm"))
                st.plotly_chart(fig_single, use_container_width=True)

                if st.button("❌ Delete This Session"):
                    db.delete_session(sess["session_id"])
                    st.success("Session deleted.")
                    st.rerun()
            else:
                st.markdown(f"##### Comparative BPM Overlays ({len(selected_ids)} Sessions)")
                fig_comp = create_session_comparison_chart(full_sessions)
                st.plotly_chart(fig_comp, use_container_width=True)


# ---------------------------------------------------------
# PAGE 5: REPORTS
# ---------------------------------------------------------
elif page_selection == "📄 Reports":
    st.markdown('<div class="brand-title">Downloadable Analytics Reports</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-subtitle">Export Standardized CSV Data and Publication-Grade PDF Reports</div>', unsafe_allow_html=True)

    sm = st.session_state.session_manager
    db = st.session_state.db_manager
    sessions = db.list_sessions()

    source_choice = st.radio("Report Source", ["Active Session", "Saved Historical Session"], horizontal=True)

    if source_choice == "Active Session":
        stats = sm.get_session_statistics()
        bpm_records = sm.history_bpm_records
        sqi_records = sm.history_sqi_records
        waveform = sm.latest_pulse_waveform
    else:
        if not sessions:
            st.warning("No saved sessions available. Switch to Active Session or complete a measurement.")
            stats = sm.get_session_statistics()
            bpm_records = []
            sqi_records = []
            waveform = None
        else:
            sel_sid = st.selectbox("Select Saved Session", [f"{s['user_label']} ({s['session_id']})" for s in sessions])
            sid = sel_sid.split("(")[-1].replace(")", "").strip()
            loaded = db.get_session(sid)
            stats = loaded if loaded else sm.get_session_statistics()
            bpm_records = loaded.get("bpm_series", []) if loaded else []
            sqi_records = []
            waveform = np.array(loaded.get("pulse_waveform", [])) if loaded else None

    # Participant Details Pill Banner
    st.markdown(f"""
    <div class="metric-card" style="padding: 0.9rem; margin-top: 0.5rem; margin-bottom: 1.2rem;">
        <span style="font-size: 0.85rem; color: #94A3B8;">Selected Subject:</span>
        <b style="color: #00F2FE; margin-left: 6px;">{stats.get('user_label', 'Subject')}</b>
        <span style="color: #94A3B8; margin-left: 10px;">• Age: <b>{stats.get('user_age', st.session_state.user_age)}</b></span>
        <span style="color: #94A3B8; margin-left: 10px;">• Gender: <b>{stats.get('user_gender', st.session_state.user_gender)}</b></span>
        <span style="color: #94A3B8; margin-left: 10px;">• Activity: <b>{stats.get('activity_state', st.session_state.activity_state)}</b></span>
        <span style="color: #94A3B8; margin-left: 10px;">• Max HR: <b style="color: #EF4444;">{stats.get('max_hr', 195)} BPM</b></span>
    </div>
    """, unsafe_allow_html=True)

    col_csv, col_pdf = st.columns(2)

    with col_csv:
        st.markdown("### 📊 CSV Data Export")
        st.markdown("Export raw and processed time-series for third-party analysis in Python, R, or Excel.")
        detailed_csv, summary_csv = ReportGenerator.generate_csv_data(stats, bpm_records, sqi_records)

        st.download_button(
            label="📥 Download Detailed Time-Series CSV",
            data=detailed_csv,
            file_name=f"pulsevision_points_{stats.get('session_id', 'export')}.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.download_button(
            label="📥 Download Session Summary CSV",
            data=summary_csv,
            file_name=f"pulsevision_summary_{stats.get('session_id', 'export')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col_pdf:
        st.markdown("### 📄 Formal PDF Report")
        st.markdown("Generate a clinical-style summary document with demographics, executive metrics, charts, and disclaimers.")

        if st.button("📑 Generate PDF Report", type="primary", use_container_width=True):
            with st.spinner("Compiling ReportLab PDF document and rendering charts..."):
                pdf_bytes = ReportGenerator.generate_pdf_report(stats, bpm_records, waveform)
                st.download_button(
                    label="⬇️ Download Generated PDF",
                    data=pdf_bytes,
                    file_name=f"pulsevision_report_{stats.get('session_id', 'export')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )


# ---------------------------------------------------------
# PAGE 6: SETTINGS & PRIVACY
# ---------------------------------------------------------
elif page_selection == "⚙️ Settings & Privacy":
    st.markdown('<div class="brand-title">Settings & Privacy Policy</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-subtitle">Configure Camera Devices, DSP Parameters, Demographics, and Review Data Privacy</div>', unsafe_allow_html=True)

    col_set1, col_set2 = st.columns(2)

    with col_set1:
        st.markdown("### 🛠️ Hardware & DSP Settings")
        
        # Camera selector
        avail_cams = CameraManager.list_available_cameras(3)
        cam_choice = st.selectbox("Webcam Device Index", avail_cams, index=0)
        if cam_choice != st.session_state.selected_cam_idx:
            st.session_state.selected_cam_idx = cam_choice
            if st.session_state.camera_manager:
                st.session_state.camera_manager.stop()
                st.session_state.camera_manager = None
            st.success(f"Selected Camera Index {cam_choice}")

        # Algorithm preset
        alg_choice = st.selectbox("Primary rPPG Algorithm", AVAILABLE_ALGORITHMS,
                                  index=AVAILABLE_ALGORITHMS.index(st.session_state.selected_algorithm))
        st.session_state.selected_algorithm = alg_choice

        # Measurement duration
        dur_choice = st.selectbox("Default Measurement Duration", [30, 60, 120, "Continuous"], index=1)
        st.session_state.selected_duration = 0.0 if dur_choice == "Continuous" else float(dur_choice)

        # Auto-save
        st.session_state.auto_save = st.checkbox("Automatically save completed sessions to SQLite", value=st.session_state.auto_save)

    with col_set2:
        st.markdown("### 🔒 Privacy & Safety Notice")
        st.markdown("""
        <div class="metric-card">
            <h4 style="color: #00F2FE; margin-top: 0;">1. Local In-Memory Processing</h4>
            <p style="color: #CBD5E1; font-size: 0.85rem; line-height: 1.5;">
                All camera frames are ingested directly into memory and discarded immediately after extracting spatial RGB means. 
                Video is never saved to disk, nor is it ever sent over the network.
            </p>

            <h4 style="color: #00F2FE; margin-top: 15px;">2. Anonymity by Default</h4>
            <p style="color: #CBD5E1; font-size: 0.85rem; line-height: 1.5;">
                PulseVision does not perform biometric facial recognition. No facial templates or identifiers are created.
            </p>

            <h4 style="color: #00F2FE; margin-top: 15px;">3. Data Retention Control</h4>
            <p style="color: #CBD5E1; font-size: 0.85rem; line-height: 1.5;">
                Only numeric time-series values (BPM estimates, timestamps, SQI, user demographic tags) are stored in the local SQLite database. 
                You can delete any session or wipe the entire database at any time.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer-box">
        <b>Final Medical Disclaimer:</b><br/>
        Measurements produced by PulseVision are estimates derived from optical signals and are subject to noise from ambient light, 
        camera compression, and subject movement. This tool is not intended for diagnostic or clinical use. Always consult a healthcare professional.
    </div>
    """, unsafe_allow_html=True)
