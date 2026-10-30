"""
app.py — Main Streamlit application for the ALPR system.

Author: Karam Singh Bana (Team Member 2) — Streamlit UI development.

Run with:
    streamlit run app.py

Pages:
    Home        — Project overview, team, technologies
    Recognition — Upload → detect → OCR → display results
    History     — Session-level recognition history
    About       — Methodology, architecture, limitations
"""

from __future__ import annotations

import io
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import streamlit as st
from PIL import Image

import config
from src.annotation import annotate_no_detection
from src.detector import ConfigurationError, PlateDetector
from src.ocr_engine import EasyOCREngine
from src.pipeline import ALPRPipeline, PipelineResult
from src.report import (
    append_history,
    history_to_csv_bytes,
    history_to_dataframe,
    report_to_csv_bytes,
    report_to_json_bytes,
)
from utils.file_utils import validate_upload

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG if config.DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page configuration — MUST be the very first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* --- Google Font: Inter --- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Sidebar */
    [data-testid="stSidebar"] { background: linear-gradient(160deg, #0f172a 0%, #1e293b 100%); }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }

    /* Main background */
    .main { background-color: #f8fafc; }

    /* Hero header */
    .hero-header {
        background: linear-gradient(135deg, #1e293b 0%, #334155 50%, #0f766e 100%);
        color: white;
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .hero-header h1 { font-size: 2.2rem; font-weight: 700; margin: 0; }
    .hero-header p  { font-size: 1.05rem; color: #94a3b8; margin-top: 0.5rem; }

    /* Result card */
    .result-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }

    /* Status badges */
    .badge-success { background:#dcfce7; color:#166534; padding:3px 10px; border-radius:20px; font-size:0.82rem; font-weight:600; }
    .badge-warning { background:#fef9c3; color:#854d0e; padding:3px 10px; border-radius:20px; font-size:0.82rem; font-weight:600; }
    .badge-error   { background:#fee2e2; color:#991b1b; padding:3px 10px; border-radius:20px; font-size:0.82rem; font-weight:600; }
    .badge-info    { background:#e0f2fe; color:#075985; padding:3px 10px; border-radius:20px; font-size:0.82rem; font-weight:600; }

    /* Team card */
    .team-card {
        background: white; border-radius: 12px; padding: 1.2rem;
        border-left: 4px solid #0f766e; margin-bottom: 0.8rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .team-card h4 { margin: 0; font-size: 1rem; color: #1e293b; }
    .team-card .role { font-size: 0.82rem; color: #64748b; margin-top: 2px; }

    /* Metric tile */
    .metric-tile {
        background: white; border-radius: 10px; padding: 1rem;
        text-align: center; border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-tile .value { font-size: 1.6rem; font-weight: 700; color: #0f766e; }
    .metric-tile .label { font-size: 0.82rem; color: #64748b; margin-top: 2px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached model initialization
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading detector model…")
def get_detector() -> PlateDetector:
    """Return a shared PlateDetector instance (loaded once per session)."""
    detector = PlateDetector()
    try:
        detector.load_model()
    except ConfigurationError as exc:
        logger.warning("Detector not loaded: %s", exc)
        # Return the unloaded detector; the pipeline will surface the error.
    return detector


@st.cache_resource(show_spinner="Initializing OCR engine…")
def get_ocr_engine() -> EasyOCREngine:
    """Return a shared EasyOCREngine instance (loaded once per session)."""
    engine = EasyOCREngine(
        languages=config.EASYOCR_LANGUAGES,
        use_gpu=config.DEVICE != "cpu",
    )
    try:
        engine.initialize()
    except RuntimeError as exc:
        logger.warning("OCR engine not loaded: %s", exc)
    return engine


# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------


def init_session_state() -> None:
    defaults = {
        "recognition_history": [],
        "last_result": None,
        "current_filename": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session_state()


# ---------------------------------------------------------------------------
# Helper: convert BGR numpy → PIL → bytes for Streamlit display / download
# ---------------------------------------------------------------------------


def bgr_to_display_bytes(bgr: np.ndarray, fmt: str = "JPEG") -> bytes:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    buf = io.BytesIO()
    pil.save(buf, format=fmt)
    return buf.getvalue()


def display_bgr_image(bgr: np.ndarray, caption: str = "") -> None:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    st.image(rgb, caption=caption, width='stretch')


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------


def render_sidebar() -> str:
    with st.sidebar:
        st.markdown("## 🚗 ALPR System")
        st.markdown("---")
        page = st.radio(
            "Navigation",
            ["🏠 Home", "🔍 Recognition", "📋 History", "ℹ️ About"],
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.markdown("**Configuration**")
        st.caption(f"Model: `{Path(config.MODEL_PATH).name}`")
        st.caption(f"Device: `{config.DEVICE}`")
        st.caption(f"OCR Engine: `{config.OCR_ENGINE}`")
        st.markdown("---")
        st.markdown("**Project Team**")
        st.caption("👤 Abhayvir Singh — Lead / Detection")
        st.caption("👤 Himank Devnani — OCR")
        st.caption("👤 Karam Singh Bana — UI / Backend")
        st.markdown("---")
        st.caption("📅 Built: 22 October 2026")
    return page


# ---------------------------------------------------------------------------
# PAGE: HOME
# ---------------------------------------------------------------------------


def page_home() -> None:
    st.markdown(
        """
        <div class="hero-header">
            <h1>🚗 Vehicle Number Plate Recognition System</h1>
            <p>AI-powered vehicle number plate detection and character recognition</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Objective
    st.markdown("### 🎯 Project Objective")
    st.info(
        "Automatically detect and read vehicle number plates from uploaded images "
        "using deep-learning-based object detection (YOLOv8) and optical character "
        "recognition (OCR), and display the extracted registration number with "
        "annotated output images."
    )

    # Workflow
    st.markdown("### ⚙️ System Workflow")
    cols = st.columns(4)
    steps = [
        ("📤", "1. Upload", "Upload a vehicle image (JPG/PNG, up to 10 MB)"),
        ("🔍", "2. Detect", "YOLOv8 detects the number plate region"),
        ("✂️", "3. Crop & OCR", "Plate is cropped and read by EasyOCR"),
        ("📊", "4. Results", "Annotated image, text, and confidence shown"),
    ]
    for col, (icon, title, desc) in zip(cols, steps):
        with col:
            st.markdown(
                f"""<div class="metric-tile">
                <div style="font-size:2rem">{icon}</div>
                <div class="value" style="font-size:1rem;margin-top:0.3rem">{title}</div>
                <div class="label">{desc}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # Technologies
    st.markdown("### 🛠️ Technology Stack")
    tech_col1, tech_col2 = st.columns(2)
    with tech_col1:
        st.markdown(
            """
            | Component | Technology |
            |-----------|-----------|
            | Object Detection | YOLOv8 (Ultralytics) |
            | OCR Engine | EasyOCR |
            | Web Interface | Streamlit |
            | Image Processing | OpenCV |
            """
        )
    with tech_col2:
        st.markdown(
            """
            | Component | Technology |
            |-----------|-----------|
            | Image I/O | Pillow (PIL) |
            | Numerics | NumPy |
            | Data Handling | Pandas |
            | Testing | Pytest |
            """
        )

    st.markdown("---")

    # Team section
    st.markdown("### 👥 Project Team")
    team = [
        {
            "name": "Abhayvir Singh",
            "role": "Team Lead — AI/ML & ALPR",
            "contributions": [
                "ALPR environment setup",
                "Image preprocessing pipeline",
                "YOLOv8 model integration",
                "Plate detection & region extraction",
                "Detection pipeline testing",
            ],
        },
        {
            "name": "Himank Devnani",
            "role": "Team Member — OCR & Computer Vision",
            "contributions": [
                "EasyOCR engine integration",
                "Alphanumeric character recognition",
                "OCR output normalization",
                "Registration number extraction",
                "OCR testing & error handling",
            ],
        },
        {
            "name": "Karam Singh Bana",
            "role": "Team Member — Python, Streamlit UI & Backend",
            "contributions": [
                "Streamlit UI development",
                "Image upload & validation",
                "Recognition result display",
                "Annotated output & confidence display",
                "System testing & demonstration interface",
            ],
        },
    ]
    for member in team:
        st.markdown(
            f"""<div class="team-card">
            <h4>👤 {member['name']}</h4>
            <div class="role">{member['role']}</div>
            </div>""",
            unsafe_allow_html=True,
        )
        with st.expander(f"Contributions — {member['name']}"):
            for c in member["contributions"]:
                st.markdown(f"- {c}")


# ---------------------------------------------------------------------------
# PAGE: RECOGNITION
# ---------------------------------------------------------------------------


def page_recognition() -> None:
    st.markdown("## 🔍 Number Plate Recognition")
    st.markdown(
        "Upload a vehicle image to detect and recognize the number plate. "
        "Accepted formats: **JPG, JPEG, PNG** — maximum **10 MB**."
    )

    # --- Upload ---
    uploaded_file = st.file_uploader(
        "Upload vehicle image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded_file is None:
        st.info("👆 Upload a vehicle image to get started.")
        return

    image_bytes = uploaded_file.read()
    filename = uploaded_file.name

    # Validate upload
    upload_error = validate_upload(
        filename,
        image_bytes,
        config.ALLOWED_EXTENSIONS,
        config.MAX_UPLOAD_BYTES,
    )
    if upload_error:
        st.error(f"❌ Upload rejected: {upload_error}")
        return

    # Show uploaded image
    col_orig, col_info = st.columns([2, 1])
    with col_orig:
        st.markdown("**Uploaded Image**")
        st.image(image_bytes, width='stretch')
    with col_info:
        size_kb = len(image_bytes) / 1024
        try:
            pil_img = Image.open(io.BytesIO(image_bytes))
            w, h = pil_img.size
        except Exception:
            w = h = "?"
        st.markdown("**Image Info**")
        st.markdown(
            f"""
            <div class="result-card">
            📄 <b>Filename:</b> {filename}<br>
            📐 <b>Dimensions:</b> {w} × {h} px<br>
            💾 <b>Size:</b> {size_kb:.1f} KB<br>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Confidence slider
        st.markdown("**Detection Settings**")
        conf_threshold = st.slider(
            "Min. detection confidence",
            min_value=0.10,
            max_value=0.95,
            value=config.DETECTION_CONFIDENCE,
            step=0.05,
            help="Lower = more detections (may include false positives); Higher = fewer but more reliable.",
        )

    st.markdown("---")

    # --- Run button ---
    if st.button("🚀 Run Recognition", type="primary", use_container_width=True):
        _run_recognition(image_bytes, filename, conf_threshold)


def _run_recognition(
    image_bytes: bytes,
    filename: str,
    conf_threshold: float,
) -> None:
    """Execute the pipeline and display results."""

    # Clear previous result
    st.session_state["last_result"] = None
    st.session_state["current_filename"] = filename

    # Load models (cached)
    detector = get_detector()
    detector.conf_threshold = conf_threshold   # honour slider
    ocr_engine = get_ocr_engine()

    pipeline = ALPRPipeline(detector=detector, ocr_engine=ocr_engine)

    with st.spinner("🔄 Running ALPR pipeline… this may take a few seconds."):
        result: PipelineResult = pipeline.process_bytes(image_bytes, filename)

    st.session_state["last_result"] = result

    # Append to history
    if result.report:
        st.session_state["recognition_history"] = append_history(
            st.session_state["recognition_history"],
            result.report,
        )

    _display_result(result, image_bytes)


def _display_result(result: PipelineResult, original_bytes: bytes) -> None:
    """Render the pipeline result in the UI."""

    st.markdown("---")
    st.markdown("### 📊 Recognition Results")

    # Status banner
    if result.success:
        st.success(f"✅ {result.user_message}")
    elif result.plate_count == 0:
        st.warning(f"⚠️ {result.user_message}")
    else:
        st.error(f"❌ {result.user_message}")

    # Metrics row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Plates Detected", result.plate_count)
    with m2:
        st.metric(
            "Recognized Text",
            result.recognized_text if result.recognized_text else "—",
        )
    with m3:
        det_conf = result.detector_confidence
        st.metric(
            "Detector Confidence",
            f"{det_conf:.1%}" if det_conf is not None else "N/A",
        )
    with m4:
        ocr_conf = result.ocr_confidence
        st.metric(
            "OCR Confidence",
            f"{ocr_conf:.1%}" if ocr_conf is not None else "N/A",
        )

    # Image tabs
    st.markdown("### 🖼️ Visual Output")
    tabs = st.tabs(["Original", "Plate Crop", "Annotated Output"])

    with tabs[0]:
        st.image(original_bytes, caption="Uploaded Image", width='stretch')

    with tabs[1]:
        if result.plate_crop is not None:
            display_bgr_image(result.plate_crop, "Detected Plate Crop")
        else:
            st.info("No plate crop available.")

    with tabs[2]:
        if result.annotated_image is not None:
            display_bgr_image(result.annotated_image, "Annotated Output")
        else:
            st.info("No annotated image available.")

    # Downloads
    st.markdown("### 💾 Downloads")
    dl_col1, dl_col2, dl_col3 = st.columns(3)

    with dl_col1:
        if result.annotated_image is not None:
            annotated_bytes = bgr_to_display_bytes(result.annotated_image, "JPEG")
            st.download_button(
                "📥 Annotated Image",
                data=annotated_bytes,
                file_name=f"annotated_{result.filename}",
                mime="image/jpeg",
                width='stretch',
            )

    with dl_col2:
        if result.report:
            st.download_button(
                "📥 Report (JSON)",
                data=report_to_json_bytes(result.report),
                file_name=f"report_{result.filename}.json",
                mime="application/json",
                width='stretch',
            )

    with dl_col3:
        if result.report:
            st.download_button(
                "📥 Report (CSV)",
                data=report_to_csv_bytes(result.report),
                file_name=f"report_{result.filename}.csv",
                mime="text/csv",
                width='stretch',
            )

    # Validation detail
    if result.validation_result:
        with st.expander("🔎 Validation Details"):
            val = result.validation_result
            st.write(f"**Text evaluated:** `{val.text}`")
            st.write(f"**Valid:** {'✅ Yes' if val.is_valid else '❌ No'}")
            st.write(f"**Message:** {val.message}")
            if val.matched_pattern:
                st.write(f"**Pattern matched:** `{val.matched_pattern}`")

    # Debug / diagnostic expander
    if config.DEBUG or result.error_message:
        with st.expander("🛠️ Diagnostic Info"):
            st.write(f"**Processing time:** {result.processing_time_sec:.3f} s")
            if result.error_message:
                st.error(f"Internal error: {result.error_message}")
            if result.ocr_result:
                st.write(f"**OCR raw text:** `{result.ocr_result.raw_text}`")
                st.write(f"**OCR variant used:** {result.ocr_result.variant_index}")


# ---------------------------------------------------------------------------
# PAGE: HISTORY
# ---------------------------------------------------------------------------


def page_history() -> None:
    st.markdown("## 📋 Recognition History")
    st.caption("History is maintained for this browser session only and resets on page refresh.")

    history = st.session_state.get("recognition_history", [])

    if not history:
        st.info("No recognition history yet. Process an image on the Recognition page first.")
        return

    st.markdown(f"**{len(history)} record(s) in this session.**")

    # Display as DataFrame
    df = history_to_dataframe(history)
    st.dataframe(df, use_container_width=True)

    # Controls
    col_export, col_clear = st.columns([2, 1])
    with col_export:
        csv_bytes = history_to_csv_bytes(history)
        st.download_button(
            "📥 Export History (CSV)",
            data=csv_bytes,
            file_name=f"alpr_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col_clear:
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state["recognition_history"] = []
            st.success("History cleared.")
            st.rerun()


# ---------------------------------------------------------------------------
# PAGE: ABOUT
# ---------------------------------------------------------------------------


def page_about() -> None:
    st.markdown("## ℹ️ About This Project")

    with st.expander("📌 Problem Statement", expanded=True):
        st.markdown(
            """
            Manual reading of vehicle number plates is error-prone and impractical
            at scale. Automated License Plate Recognition (ALPR) systems address this
            by combining computer vision and machine learning to detect and read plates
            reliably across varying lighting, angles, and vehicle speeds.

            This project implements a self-contained ALPR pipeline suitable for
            demonstration and educational purposes.
            """
        )

    with st.expander("🎯 Objectives"):
        st.markdown(
            """
            1. Detect licence-plate regions in uploaded vehicle images using a
               trained YOLOv8 model.
            2. Extract and enhance the plate crop for optical character recognition.
            3. Recognize alphanumeric registration text using EasyOCR.
            4. Display annotated images, confidence scores, and downloadable reports.
            5. Provide a clean Streamlit interface suitable for college demonstration.
            """
        )

    with st.expander("🗺️ Methodology"):
        st.markdown(
            """
            **Detection (YOLOv8)**
            - A YOLOv8 model fine-tuned on licence-plate imagery predicts bounding
              boxes around number plate regions.
            - Confidence threshold is configurable; detections below the threshold
              are discarded.

            **Preprocessing**
            - The detected plate crop is resized to a uniform height.
            - Multiple preprocessing variants are generated: colour, CLAHE-enhanced
              grayscale, Otsu binary, adaptive threshold, and sharpened CLAHE.

            **OCR (EasyOCR)**
            - All variants are passed through EasyOCR.
            - The variant that yields the highest average per-word confidence is
              selected as the final result.

            **Validation**
            - The normalized text is checked against common Indian registration
              patterns as a heuristic.  Validation failure does not discard the
              OCR output; it is displayed as an advisory note.
            """
        )

    with st.expander("🏗️ System Architecture"):
        st.markdown(
            """
            ```
            Image Upload
                ↓
            File Validation (type, size, readability)
                ↓
            Preprocessing (resize, colour space)
                ↓
            YOLOv8 Plate Detection
                ↓
            Bounding Box Clipping & Sorting
                ↓
            Plate Crop Extraction (with padding)
                ↓
            OCR Variant Generation (5 variants)
                ↓
            EasyOCR Recognition (best variant selected)
                ↓
            Text Normalization & Format Validation
                ↓
            Image Annotation (bounding box + label)
                ↓
            Result Display + Downloads
            ```
            """
        )

    with st.expander("⚠️ Known Limitations"):
        st.markdown(
            """
            - **Model weights**: A licence-plate-specific YOLOv8 model must be
              supplied. The system cannot detect plates without valid weights.
            - **OCR accuracy**: Heavily occluded, blurry, or non-standard plates
              may not be recognized correctly.
            - **Indian plates**: Validation patterns are tailored to common Indian
              formats; international formats may not pass strict validation.
            - **Multi-plate images**: Only the highest-confidence plate is shown
              in detail (all detections are annotated on the image).
            - **CPU inference**: Detection and OCR are slower on CPU-only machines
              (~2–10 s per image depending on hardware).
            - **No real-time video**: This application processes still images only.
            """
        )

    with st.expander("🔭 Future Scope"):
        st.markdown(
            """
            - Real-time video stream processing via webcam.
            - Support for multiple simultaneous plates with individual reports.
            - Integration with vehicle registration databases (with appropriate
              legal authorization).
            - Mobile-optimized deployment.
            - State-aware Indian registration validation with state-code lookup.
            - On-device model optimization using ONNX or TensorRT.
            """
        )

    with st.expander("👥 Team Member Responsibilities"):
        responsibilities = {
            "Abhayvir Singh (Team Lead)": [
                "ALPR environment setup and configuration",
                "Image preprocessing utilities",
                "YOLOv8 model integration",
                "Vehicle number plate detection",
                "Plate region extraction and cropping",
                "Detection pipeline testing",
                "Integration and coordination",
            ],
            "Himank Devnani": [
                "OCR engine integration (EasyOCR)",
                "Detected plate image processing",
                "Alphanumeric character recognition",
                "Registration number extraction",
                "OCR output normalization and formatting",
                "OCR testing and error handling",
            ],
            "Karam Singh Bana": [
                "Streamlit UI development",
                "Vehicle image upload functionality",
                "Recognition result display",
                "Annotated output image display",
                "Confidence score display",
                "System testing and demonstration interface",
            ],
        }
        for member, tasks in responsibilities.items():
            st.markdown(f"**{member}**")
            for task in tasks:
                st.markdown(f"  - {task}")

    with st.expander("📜 License & Attribution"):
        st.markdown(
            """
            - **Ultralytics YOLOv8**: AGPL-3.0 license (for research use).
            - **EasyOCR**: Apache 2.0 license.
            - **OpenCV**: Apache 2.0 license.
            - **Streamlit**: Apache 2.0 license.

            Licence-plate model weights: see `models/README.md` for source and
            license information.

            This project is developed for educational purposes as part of a
            college engineering curriculum.
            """
        )


# ---------------------------------------------------------------------------
# MAIN ROUTER
# ---------------------------------------------------------------------------


def main() -> None:
    page = render_sidebar()

    if "Home" in page:
        page_home()
    elif "Recognition" in page:
        page_recognition()
    elif "History" in page:
        page_history()
    elif "About" in page:
        page_about()


if __name__ == "__main__":
    main()
