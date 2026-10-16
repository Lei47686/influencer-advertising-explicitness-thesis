import json
import uuid
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from study1_agent import Study1Agent
from study1_code import get_batch_required_columns

st.set_page_config(page_title="Study 1 AES Workbench", layout="wide", initial_sidebar_state="collapsed")

# Batch CSV validation requirements.
BATCH_REQUIRED_COLUMNS = get_batch_required_columns()

# Single-run metadata CSV validation requirements.
SINGLE_METADATA_REQUIRED_COLUMNS = [
    "video_id",
    "likes",
    "comments",
    "shares",
    "views",
    "post_date",
    "creator_id",
]
TEMP_BASE_DIR = Path(tempfile.gettempdir()) / "aes_v3_streamlit_uploads"
DEFAULT_RESULTS_BASE_DIR = Path.cwd() / "aes_streamlit_results"

CUSTOM_CSS = """
<style>
:root { --text: #f4f7fb; --muted: rgba(255,255,255,0.86); }
.stApp { background: linear-gradient(180deg, #07090f 0%, #0b1018 42%, #0e1521 100%); color: var(--text); }
.stApp::before {
    content: ""; position: fixed; inset: 0; pointer-events: none; z-index: 0;
    background-image:
      linear-gradient(rgba(5, 8, 16, 0.62), rgba(5, 8, 16, 0.68)),
      url("https://images.pexels.com/photos/11565603/pexels-photo-11565603.jpeg?cs=srgb&dl=pexels-ilya-kovalchuk-164429912-11565603.jpg&fm=jpg");
    background-size: cover; background-position: center center; background-repeat: no-repeat; background-attachment: fixed; opacity: 0.82;
}
.block-container { position: relative; z-index: 1; max-width: 1320px; padding-top: 1.8rem; padding-bottom: 2rem; }
section[data-testid="stSidebar"] { display: none !important; }
header[data-testid="stHeader"] { background: transparent !important; height: 0 !important; }
button[kind="header"], [data-testid="stToolbar"] { display: none !important; }

/* global text visibility */
html, body, p, div, span, label, small, li, h1, h2, h3, h4 {
    color: var(--text) !important;
}
label, .stMarkdown p, .stCaptionContainer, [data-testid="stMarkdownContainer"] p {
    color: var(--muted) !important;
    -webkit-text-fill-color: var(--muted) !important;
}
[data-testid="stTextInputRootElement"] label,
[data-testid="stNumberInput"] label,
[data-testid="stRadio"] label,
[data-testid="stFileUploader"] label {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    font-weight: 600 !important;
}

/* buttons */
.stButton > button, button[data-testid="baseButton-secondary"], button[data-testid="baseButton-primary"] {
    border-radius: 14px !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    background: linear-gradient(135deg, rgba(255,47,146,0.20), rgba(0,209,255,0.16)) !important;
    color: #ffffff !important;
    font-weight: 650 !important;
}

/* inputs */
div[data-baseweb="input"], div[data-baseweb="base-input"], div[data-baseweb="textarea"], div[data-baseweb="select"] > div,
.stTextInput > div > div, .stNumberInput > div > div, .stTextArea > div > div, .stSelectbox > div > div {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 14px !important;
    color: #ffffff !important;
}
input, textarea {
    background: transparent !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}

/* uploader glass override: fully cover Streamlit internal layers */
/* Key fix area: uploader outer container, dropzone, button, instructions, uploaded file chip, plus/remove controls */
div[data-testid="stFileUploader"] {
    background: transparent !important;
}
div[data-testid="stFileUploader"] > div,
div[data-testid="stFileUploader"] section,
div[data-testid="stFileUploader"] section > div,
div[data-testid="stFileUploaderDropzone"],
div[data-testid="stFileUploaderDropzone"] > div,
div[data-testid="stFileUploaderDropzone"] > div > div {
    background: rgba(12, 16, 24, 0.68) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 18px !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.03), 0 10px 30px rgba(0,0,0,0.16) !important;
    backdrop-filter: blur(8px) !important;
}
div[data-testid="stFileUploader"] section [role="button"],
div[data-testid="stFileUploader"] section [role="button"]:hover,
div[data-testid="stFileUploader"] section [role="button"]:focus,
div[data-testid="stFileUploader"] button,
div[data-testid="stFileUploader"] button:hover,
div[data-testid="stFileUploader"] button:focus,
div[data-testid="stFileUploader"] button:active,
div[data-testid="stFileUploaderDropzone"] [role="button"],
div[data-testid="stFileUploaderDropzone"] [role="button"]:hover,
div[data-testid="stFileUploaderDropzone"] [role="button"]:focus,
div[data-testid="stFileUploaderDropzone"] button,
div[data-testid="stFileUploaderDropzone"] button:hover,
div[data-testid="stFileUploaderDropzone"] button:focus,
div[data-testid="stFileUploaderDropzone"] button:active {
    background: rgba(20, 24, 34, 0.94) !important;
    background-image: none !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: 16px !important;
    box-shadow: none !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}
div[data-testid="stFileUploader"] button > div,
div[data-testid="stFileUploader"] button > div > div,
div[data-testid="stFileUploader"] button > div > div > div,
div[data-testid="stFileUploader"] section [role="button"] > div,
div[data-testid="stFileUploader"] section [role="button"] > div > div,
div[data-testid="stFileUploader"] section [role="button"] > div > div > div,
div[data-testid="stFileUploaderDropzone"] button > div,
div[data-testid="stFileUploaderDropzone"] button > div > div,
div[data-testid="stFileUploaderDropzone"] button > div > div > div,
div[data-testid="stFileUploaderDropzone"] [role="button"] > div,
div[data-testid="stFileUploaderDropzone"] [role="button"] > div > div,
div[data-testid="stFileUploaderDropzone"] [role="button"] > div > div > div {
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
}
div[data-testid="stFileUploader"] button [data-testid],
div[data-testid="stFileUploaderDropzone"] button [data-testid],
div[data-testid="stFileUploader"] section [role="button"] [data-testid],
div[data-testid="stFileUploaderDropzone"] [role="button"] [data-testid] {
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
}
div[data-testid="stFileUploader"] button span,
div[data-testid="stFileUploader"] button p,
div[data-testid="stFileUploader"] button small,
div[data-testid="stFileUploaderDropzone"] button span,
div[data-testid="stFileUploaderDropzone"] button p,
div[data-testid="stFileUploaderDropzone"] button small,
div[data-testid="stFileUploader"] section [role="button"] span,
div[data-testid="stFileUploader"] section [role="button"] p,
div[data-testid="stFileUploaderDropzone"] [role="button"] span,
div[data-testid="stFileUploaderDropzone"] [role="button"] p {
    background: transparent !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}
div[data-testid="stFileUploader"] svg,
div[data-testid="stFileUploader"] svg *,
div[data-testid="stFileUploaderDropzone"] svg,
div[data-testid="stFileUploaderDropzone"] svg * {
    background: transparent !important;
    box-shadow: none !important;
    color: #ffffff !important;
    fill: #ffffff !important;
}
div[data-testid="stFileUploaderDropzoneInstructions"],
div[data-testid="stFileUploaderDropzoneInstructions"] > div,
div[data-testid="stFileUploaderDropzoneInstructions"] > div > div {
    background: rgba(20, 24, 34, 0.94) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 14px !important;
    box-shadow: none !important;
}
div[data-testid="stFileUploaderDropzoneInstructions"] *,
div[data-testid="stFileUploader"] small,
div[data-testid="stFileUploader"] span {
    background: transparent !important;
    color: rgba(255,255,255,0.84) !important;
    -webkit-text-fill-color: rgba(255,255,255,0.84) !important;
}
div[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"],
div[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] > div,
div[data-testid="stFileUploader"] [data-testid="stFileUploaderFileData"],
div[data-testid="stFileUploader"] [data-testid="stFileUploaderFileData"] > div,
div[data-testid="stFileUploader"] [data-testid="stFileUploaderFileData"] > div > div {
    background: rgba(20, 24, 34, 0.94) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 14px !important;
    box-shadow: none !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}
div[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] span,
div[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] p,
div[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] small,
div[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] label {
    background: transparent !important;
    color: rgba(255,255,255,0.90) !important;
    -webkit-text-fill-color: rgba(255,255,255,0.90) !important;
}
div[data-testid="stFileUploader"] button[kind="secondary"],
div[data-testid="stFileUploader"] button[kind="secondary"]:hover,
div[data-testid="stFileUploader"] button[kind="secondary"]:focus,
div[data-testid="stFileUploader"] [aria-label="+"],
div[data-testid="stFileUploader"] [aria-label="+"]:hover,
div[data-testid="stFileUploader"] [aria-label="+"]:focus,
div[data-testid="stFileUploader"] [aria-label*="Remove"],
div[data-testid="stFileUploader"] [aria-label*="Remove"]:hover,
div[data-testid="stFileUploader"] [aria-label*="Remove"]:focus,
div[data-testid="stFileUploader"] [title*="Remove"],
div[data-testid="stFileUploader"] [title*="Remove"]:hover,
div[data-testid="stFileUploader"] [title*="Remove"]:focus {
    background: rgba(20, 24, 34, 0.94) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    box-shadow: none !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}

/* radio text */
div[role="radiogroup"] label, div[role="radiogroup"] p, div[role="radiogroup"] span {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}

/* alerts */
div[data-testid="stAlert"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 16px !important;
    color: #f4f7fb !important;
}

/* metrics: fix dim labels and values */
div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 18px !important;
    padding: 0.8rem 0.9rem !important;
    min-height:132px !important;
}
div[data-testid="metric-container"] * {
    color: #f4f7fb !important;
    -webkit-text-fill-color: #f4f7fb !important;
}
div[data-testid="metric-container"] [data-testid="stMetricLabel"] {
    color: rgba(255,255,255,0.88) !important;
    font-weight: 600 !important;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 1.7rem !important;
}

/* dataframe / json / code / expander internals */
div[data-testid="stDataFrame"] {
    border-radius: 16px !important;
    overflow: hidden !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
}

div[data-testid="stJson"],
div[data-testid="stJson"] *,
div[data-testid="stCodeBlock"],
div[data-testid="stCodeBlock"] *,
pre, code {
    background: rgba(12,16,24,0.96) !important;
    color: #f4f7fb !important;
    -webkit-text-fill-color: #f4f7fb !important;
    border-color: rgba(255,255,255,0.08) !important;
}
div[data-testid="stJson"],
div[data-testid="stCodeBlock"],
pre {
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 16px !important;
    overflow: hidden !important;
}

div[data-testid="stExpander"] details {
    background: transparent !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 18px !important;
    overflow: hidden !important;
}
div[data-testid="stExpander"] summary {
    background: rgba(255,255,255,0.04) !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    border-radius: 16px !important;
}
div[data-testid="stExpander"] summary * {
    background: transparent !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    box-shadow: none !important;
}
div[data-testid="stExpander"] div[role="button"],
div[data-testid="stExpander"] div[role="button"] * {
    background: transparent !important;
    box-shadow: none !important;
}
div[data-testid="stExpander"] svg {
    fill: #ffffff !important;
    color: #ffffff !important;
    background: transparent !important;
}

/* input right-side buttons / copy / controls */
button[title],
[data-testid="stTextInput"] button,
[data-testid="stNumberInput"] button,
[data-testid="stTextInput"] [role="button"],
[data-testid="stNumberInput"] [role="button"] {
    background: rgba(24,28,38,0.96) !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
}
button[title] svg,
[data-testid="stTextInput"] button svg,
[data-testid="stNumberInput"] button svg,
[data-testid="stTextInput"] [role="button"] svg,
[data-testid="stNumberInput"] [role="button"] svg {
    color: #ffffff !important;
    fill: #ffffff !important;
}

/* banner */
.app-banner {
    background: linear-gradient(135deg, rgba(8,18,20,0.88) 0%, rgba(10,28,24,0.84) 48%, rgba(8,20,30,0.86) 100%);
    border: 1px solid rgba(120,255,190,0.14);
    border-radius: 24px;
    padding: 1.25rem 1.45rem;
    margin-bottom: 1.2rem;
}

/* ===== cleanup: number input unified dark glass style ===== */
[data-testid="stNumberInput"] {
    background: transparent !important;
}
[data-testid="stNumberInput"] > div,
[data-testid="stNumberInput"] > div > div,
[data-testid="stNumberInput"] div[data-baseweb="input"],
[data-testid="stNumberInput"] div[data-baseweb="base-input"] {
    background: rgba(255,255,255,0.07) !important;
    border-radius: 16px !important;
    box-shadow: none !important;
}
[data-testid="stNumberInput"] input {
    background: transparent !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}
[data-testid="stNumberInput"] button,
[data-testid="stNumberInput"] button:hover,
[data-testid="stNumberInput"] button:focus,
[data-testid="stNumberInput"] [role="button"],
[data-testid="stNumberInput"] [role="button"]:hover,
[data-testid="stNumberInput"] [role="button"]:focus {
    background: rgba(20,24,36,0.92) !important;
    border: 0 !important;
    border-left: 1px solid rgba(255,255,255,0.10) !important;
    box-shadow: none !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}
[data-testid="stNumberInput"] button svg,
[data-testid="stNumberInput"] [role="button"] svg,
[data-testid="stNumberInput"] button svg *,
[data-testid="stNumberInput"] [role="button"] svg * {
    fill: #ffffff !important;
    color: #ffffff !important;
    background: transparent !important;
}

</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.title("Study 1 AES Workbench")
st.markdown('<div class="app-banner"><b>Study 1 AES Analysis Workbench</b><br/>Main-page workflow for single-run debugging and batch processing. No sidebar dependency.</div>', unsafe_allow_html=True)

def init_session_state():
    defaults = {
        "uploaded_video_meta": None,
        "uploaded_video_path": "",
        "uploaded_info_meta": None,
        "uploaded_info_path": "",
        "uploaded_image_meta": None,
        "uploaded_image_path": "",
        "last_saved_paths": None,
        "last_result": None,
        "last_batch_response": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
init_session_state()

# Single-run metadata CSV validation.
def validate_single_metadata_csv_columns(df: pd.DataFrame):
    return [c for c in SINGLE_METADATA_REQUIRED_COLUMNS if c not in df.columns]


# Batch CSV validation.
def validate_batch_csv_columns(df: pd.DataFrame):
    return [c for c in BATCH_REQUIRED_COLUMNS if c not in df.columns]

def safe_str(x):
    return "" if pd.isna(x) else str(x)

def build_metadata_from_row(row: pd.Series) -> dict:
    return {
        "video_id": safe_str(row.get("video_id", "")),
        "likes": int(row.get("likes", 0) or 0),
        "comments": int(row.get("comments", 0) or 0),
        "shares": int(row.get("shares", 0) or 0),
        "views": int(row.get("views", 0) or 0),
        "post_date": safe_str(row.get("post_date", "")),
        "creator_id": safe_str(row.get("creator_id", "")),
    }

def upload_signature(uploaded_file):
    if uploaded_file is None:
        return None
    return (uploaded_file.name, uploaded_file.size, getattr(uploaded_file, "type", ""))

def save_uploaded_file(uploaded_file, category: str, temp_dir: Path) -> str:
    temp_dir.mkdir(parents=True, exist_ok=True)
    unique_name = f"{category}_{uuid.uuid4().hex}_{uploaded_file.name}"
    output_path = temp_dir / unique_name
    with open(output_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(output_path)

def persist_uploaded_file(uploaded_file, category: str):
    path_key = f"uploaded_{category}_path"
    meta_key = f"uploaded_{category}_meta"
    if uploaded_file is None:
        return st.session_state.get(path_key, "")
    current_sig = upload_signature(uploaded_file)
    if st.session_state.get(meta_key) == current_sig and st.session_state.get(path_key):
        return st.session_state[path_key]
    saved_path = save_uploaded_file(uploaded_file, category=category, temp_dir=TEMP_BASE_DIR)
    st.session_state[path_key] = saved_path
    st.session_state[meta_key] = current_sig
    return saved_path

def render_saved_paths(saved):
    if not saved:
        return
    st.success("Results were saved automatically.")
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Output directory**")
        st.code(saved["output_dir"])
    with c2:
        st.write("**Main files**")
        st.write(saved["result_json_path"])
        st.write(saved["flat_record_csv_path"])

def classify_aes_band(aes_score: float) -> str:
    if aes_score >= 0.55:
        return "High explicitness"
    if aes_score <= 0.40:
        return "Low explicitness"
    return "Medium explicitness"

def render_simple_result(result: dict, saved_paths: dict | None):
    aes_score = float(result.get("aes_score", 0.0) or 0.0)
    aes_reason = result.get("aes_reason", "")
    status = result.get("status", "")
    video_mode = result.get("video_mode", "")
    transcript_quality = result.get("transcript_quality", "")
    controller_reason = result.get("controller_reason", "")
    engagement_rate = result.get("engagement_rate", 0.0)

    render_saved_paths(saved_paths)
    st.subheader("Analysis Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AES Score", f"{aes_score:.4f}")
    c2.metric("Explicitness level", classify_aes_band(aes_score))
    c3.metric("Video Mode", video_mode or "-")
    c4.metric("Transcript Quality", transcript_quality or "-")

    if status.endswith("FAILED") or status in {"NOT_APPLICABLE", "MISSING_PRODUCT_REFERENCE", "MISSING_PRODUCT_IMAGE"}:
        st.error(result.get("status_reason", "Run failed."))
    elif status == "PRECHECK_WARNING":
        st.warning(result.get("status_reason", "Warnings were detected, but the analysis continued."))
    else:
        st.success("Analysis completed.")

    st.markdown("### Core Interpretation")
    st.write(f"**AES interpretation:** {aes_reason or '-'}")
    st.write(f"**Controller interpretation:** {controller_reason or '-'}")
    st.markdown("### Key Metrics")
    cc1, cc2, cc3, cc4 = st.columns(4)
    cc1.metric("Visual Avg", f"{result.get('visual_similarity_score', 0.0):.4f}")
    cc2.metric("Visual Top10", f"{result.get('top_10_frame_mean', 0.0):.4f}")
    cc3.metric("Semantic Global", f"{result.get('global_semantic_similarity', 0.0):.4f}")
    cc4.metric("Semantic Top5", f"{result.get('top_5_sentence_mean', 0.0):.4f}")

    st.markdown("### Metadata Summary")
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Views", f"{int(result.get('views', 0) or 0)}")
    mc2.metric("Likes", f"{int(result.get('likes', 0) or 0)}")
    mc3.metric("Shares", f"{int(result.get('shares', 0) or 0)}")
    mc4.metric("Engagement Rate", f"{float(engagement_rate or 0.0):.6f}")

    with st.expander("View top relevant sentences", expanded=True):
        sentences = result.get("sentences", [])
        if sentences:
            df = pd.DataFrame(sentences)
            if "similarity" in df.columns:
                keep_cols = [c for c in ["start", "end", "text", "similarity", "window_visual_mean", "window_visual_peak"] if c in df.columns]
                df = df[keep_cols].sort_values(by="similarity", ascending=False).head(5)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No sentence-level results are available yet.")

    with st.expander("View top relevant frames", expanded=False):
        frame_items = result.get("frame_similarity_items", [])
        if frame_items:
            df = pd.DataFrame(frame_items)
            keep_cols = [c for c in ["timestamp", "similarity", "path"] if c in df.columns]
            df = df[keep_cols].sort_values(by="similarity", ascending=False).head(10)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No frame-level results are available yet.")

    with st.expander("View transcript summary", expanded=False):
        transcript = result.get("transcript", "")
        if transcript:
            st.text_area("Transcript", transcript, height=220)
            qc1, qc2, qc3 = st.columns(3)
            qc1.metric("Text units", int(result.get("transcript_text_unit_count", 0) or 0))
            qc2.metric("CJK characters", int(result.get("transcript_cjk_character_count", 0) or 0))
            qc3.metric(
                "Avg units / sentence",
                f"{float(result.get('avg_text_units_per_sentence', 0.0) or 0.0):.2f}",
            )
        else:
            st.info("No transcript is available yet.")

    with st.expander("View save and run details", expanded=False):
        st.write({
            "run_id": result.get("run_id", ""),
            "status": result.get("status", ""),
            "status_reason": result.get("status_reason", ""),
            "error_stage": result.get("error_stage", ""),
            "audio_error": result.get("audio_error", ""),
            "text_model_name": result.get("text_model_name", ""),
            "embedding_dimension": len(result.get("product_text_embedding", [])),
            "transcription_language": result.get("transcription_language", ""),
            "semantic_scoring_version": result.get("semantic_scoring_version", ""),
            "threshold_calibration_status": result.get("threshold_calibration_status", ""),
            "run_dir": result.get("run_dir", ""),
        })

def render_batch_response(batch_response: dict):
    summary = batch_response["batch_summary"]
    st.success(batch_response.get("agent_message", "Batch analysis completed."))
    c1, c2, c3 = st.columns(3)
    c1.metric("Total rows", summary["total_rows"])
    c2.metric("Success rows", summary["success_rows"])
    c3.metric("Failed rows", summary["failed_rows"])
    c4, c5 = st.columns(2)
    c4.metric("Success Rate", f"{float(summary.get('success_rate', 0.0) or 0.0):.2%}")
    c5.write("")
    st.write("**Batch output directory**")
    st.code(summary["batch_output_dir"])
    if batch_response.get("output_csv_path"):
        st.write("**Batch results CSV**")
        st.code(batch_response["output_csv_path"])
        if Path(batch_response["output_csv_path"]).exists():
            out_df = pd.read_csv(batch_response["output_csv_path"])
            st.dataframe(out_df.head(20), use_container_width=True)
    if batch_response.get("failed_rows_csv_path"):
        st.write("**Failed rows CSV**")
        st.code(batch_response["failed_rows_csv_path"])
    if batch_response.get("batch_summary_json_path"):
        st.write("**Batch summary JSON**")
        st.code(batch_response["batch_summary_json_path"])
    if batch_response.get("video_modeling_dataset_path"):
        st.write("**Video modeling dataset**")
        st.code(batch_response["video_modeling_dataset_path"])
        if Path(batch_response["video_modeling_dataset_path"]).exists():
            st.dataframe(pd.read_csv(batch_response["video_modeling_dataset_path"]).head(20), use_container_width=True)
    if batch_response.get("product_day_panel_path"):
        st.write("**Product-day panel**")
        st.code(batch_response["product_day_panel_path"])
        if Path(batch_response["product_day_panel_path"]).exists():
            st.dataframe(pd.read_csv(batch_response["product_day_panel_path"]).head(20), use_container_width=True)
    if batch_response.get("brand_day_panel_path"):
        st.write("**Brand-day panel**")
        st.code(batch_response["brand_day_panel_path"])
        if Path(batch_response["brand_day_panel_path"]).exists():
            st.dataframe(pd.read_csv(batch_response["brand_day_panel_path"]).head(20), use_container_width=True)
    if batch_response.get("panel_regression_summary_path"):
        st.write("**Panel regression summary**")
        st.code(batch_response["panel_regression_summary_path"])
    if batch_response.get("panel_regression_coefficients_path"):
        st.write("**Panel regression coefficients**")
        st.code(batch_response["panel_regression_coefficients_path"])

    # New regression display: show the key panel regression result directly on the page.
    regression_summary_path = batch_response.get("panel_regression_summary_path")
    regression_coefficients_path = batch_response.get("panel_regression_coefficients_path")
    regression_key_result = batch_response.get("panel_regression_key_result") or summary.get("panel_regression_key_result", {})

    if regression_summary_path or regression_coefficients_path:
        st.markdown("### Panel Regression Overview")

        regression_summary = None
        if regression_summary_path:
            summary_path_obj = Path(regression_summary_path)
            if summary_path_obj.exists():
                try:
                    regression_summary = json.loads(summary_path_obj.read_text(encoding="utf-8"))
                except Exception:
                    st.warning("Panel regression summary file could not be parsed.")
            else:
                st.warning("Panel regression summary file not found.")

        regression_coefficients_df = None
        if regression_coefficients_path:
            coef_path_obj = Path(regression_coefficients_path)
            if coef_path_obj.exists():
                try:
                    regression_coefficients_df = pd.read_csv(coef_path_obj)
                except Exception:
                    st.warning("Panel regression coefficients file could not be parsed.")
            else:
                st.warning("Panel regression coefficients file not found.")

        if regression_summary:
            st.write({
                "outcome": regression_summary.get("outcome", ""),
                "fe_level": regression_summary.get("fe_level", ""),
                "n_obs": regression_summary.get("n_obs", ""),
                "r_squared": regression_summary.get("r_squared", ""),
                "adj_r_squared": regression_summary.get("adj_r_squared", ""),
                "formula": regression_summary.get("formula", ""),
            })

        if not regression_key_result and regression_summary:
            regression_key_result = regression_summary.get("panel_regression_key_result", {}) or {}

        st.markdown("#### AES Coefficient")
        aes_row = None
        if isinstance(regression_coefficients_df, pd.DataFrame) and not regression_coefficients_df.empty:
            aes_match = regression_coefficients_df.loc[
                regression_coefficients_df["term"].astype(str) == "aes_score"
            ]
            if not aes_match.empty:
                aes_row = aes_match.iloc[0]

        if regression_key_result:
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("AES Beta", f"{float(regression_key_result.get('aes_coefficient')):.6f}" if regression_key_result.get("aes_coefficient") is not None else "-")
            k2.metric("Std Error", f"{float(regression_key_result.get('aes_std_error')):.6f}" if regression_key_result.get("aes_std_error") is not None else "-")
            k3.metric("t Value", f"{float(regression_key_result.get('aes_t_value')):.6f}" if regression_key_result.get("aes_t_value") is not None else "-")
            k4.metric("p Value", f"{float(regression_key_result.get('aes_p_value')):.6f}" if regression_key_result.get("aes_p_value") is not None else "-")
            r1, r2, r3 = st.columns(3)
            r1.metric("R²", f"{float(regression_key_result.get('r_squared')):.6f}" if regression_key_result.get("r_squared") is not None else "-")
            r2.metric("Adj. R²", f"{float(regression_key_result.get('adj_r_squared')):.6f}" if regression_key_result.get("adj_r_squared") is not None else "-")
            r3.metric("N Obs", f"{int(regression_key_result.get('n_obs'))}" if regression_key_result.get("n_obs") is not None else "-")
        elif aes_row is not None:
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("AES Beta", f"{float(aes_row['coefficient']):.6f}")
            k2.metric("Std Error", f"{float(aes_row['std_error']):.6f}")
            k3.metric("t Value", f"{float(aes_row['t_value']):.6f}")
            k4.metric("p Value", f"{float(aes_row['p_value']):.6f}")
            if regression_summary:
                r1, r2, r3 = st.columns(3)
                r1.metric("R²", f"{float(regression_summary.get('r_squared', 0.0)):.6f}")
                r2.metric("Adj. R²", f"{float(regression_summary.get('adj_r_squared', 0.0)):.6f}")
                r3.metric("N Obs", f"{int(regression_summary.get('n_obs', 0) or 0)}")
        else:
            st.info("AES coefficient not found in regression output.")

        if isinstance(regression_coefficients_df, pd.DataFrame):
            if regression_coefficients_df.empty:
                st.info("Panel regression coefficients file is empty.")
            else:
                with st.expander("View full regression coefficients", expanded=False):
                    st.dataframe(regression_coefficients_df, use_container_width=True)

    # New batch-level summary display for research diagnostics.
    st.markdown("### Batch Summary")
    st.write("**Video mode counts**")
    st.write(summary.get("video_mode_counts", {}))

    st.write("**AES band counts**")
    st.write(summary.get("aes_band_counts", {}))

    st.write("**Failure reason counts**")
    st.write(summary.get("failure_reason_counts", {}))

    with st.expander("View detailed batch summary", expanded=False):
        st.json(summary)

    top_failure_examples = summary.get("top_failure_examples", []) or []
    if top_failure_examples:
        with st.expander("View top failure examples", expanded=False):
            st.dataframe(pd.DataFrame(top_failure_examples), use_container_width=True)

results_base_dir_str = st.text_input("results_base_dir", value=str(DEFAULT_RESULTS_BASE_DIR))
results_base_dir = Path(results_base_dir_str).expanduser().resolve()
analysis_mode = st.radio("Analysis mode", ["Single run", "Batch run"], horizontal=True)
st.markdown("---")

if analysis_mode == "Single run":
    st.subheader("Step 1 · Upload analysis files")
    st.caption("Single-run mode uses one video, one product text file, and one product image.")
    c1, c2, c3 = st.columns(3)
    with c1:
        uploaded_video = st.file_uploader("Upload video file", type=["mp4", "mov", "avi", "mkv"], key="video_uploader_single")
    with c2:
        uploaded_product_info = st.file_uploader("Upload product_info / introduction", type=["txt"], key="info_uploader_single")
    with c3:
        uploaded_product_image = st.file_uploader("Upload product image", type=["jpg", "jpeg", "png", "webp"], key="image_uploader_single")
    persist_uploaded_file(uploaded_video, "video")
    persist_uploaded_file(uploaded_product_info, "info")
    persist_uploaded_file(uploaded_product_image, "image")

    with st.expander("View resolved input paths", expanded=False):
        st.write({
            "video_path": st.session_state.get("uploaded_video_path", ""),
            "product_info_path": st.session_state.get("uploaded_info_path", ""),
            "product_image_path": st.session_state.get("uploaded_image_path", ""),
        })

    st.markdown("---")
    st.subheader("Step 2 · Provide metadata")
    metadata_mode = st.radio("Metadata input mode", ["Manual entry", "Use CSV"], horizontal=True, key="single_metadata_mode")
    metadata = None
    if metadata_mode == "Manual entry":
        a, b = st.columns(2)
        with a:
            video_id = st.text_input("video_id", value="demo_001")
            creator_id = st.text_input("creator_id", value="creator_demo")
            post_date = st.text_input("post_date", value="2026-04-01")
        with b:
            likes = st.number_input("likes", min_value=0, value=1200, step=1)
            comments = st.number_input("comments", min_value=0, value=86, step=1)
            shares = st.number_input("shares", min_value=0, value=43, step=1)
            views = st.number_input("views", min_value=1, value=23000, step=1)
        metadata = {"video_id": video_id, "likes": likes, "comments": comments, "shares": shares, "views": views, "post_date": post_date, "creator_id": creator_id}
    else:
        uploaded_metadata_csv = st.file_uploader("Upload metadata CSV", type=["csv"], key="single_metadata_csv")
        if uploaded_metadata_csv is not None:
            csv_df = pd.read_csv(uploaded_metadata_csv)
            missing = validate_single_metadata_csv_columns(csv_df)
            if missing:
                st.error("The CSV is missing required fields:")
                for c in missing:
                    st.write(f"- {c}")
            else:
                st.success(f"CSV loaded: {len(csv_df)} rows")
                option_labels = [f"{idx} | {safe_str(row['video_id'])}" for idx, row in csv_df.iterrows()]
                selected_label = st.selectbox("Select metadata row", option_labels, key="single_csv_row_select")
                selected_idx = int(selected_label.split("|")[0].strip())
                metadata = build_metadata_from_row(csv_df.iloc[selected_idx])
                st.dataframe(pd.DataFrame([metadata]), use_container_width=True)
        else:
            st.info("Upload a metadata CSV here for single-run CSV mode.")

    st.markdown("---")
    st.subheader("Step 3 · Run analysis")
    if st.button("Run current analysis"):
        agent = Study1Agent(results_base_dir=results_base_dir)
        with st.spinner("Running analysis..."):
            response = agent.run_single(
                video_path=st.session_state.get("uploaded_video_path", ""),
                product_info_path=st.session_state.get("uploaded_info_path", ""),
                product_image_path=st.session_state.get("uploaded_image_path", ""),
                metadata=metadata,
            )
        if not response.get("ok", False):
            st.error(response.get("agent_message", "Run failed."))
        else:
            st.session_state["last_saved_paths"] = response["saved_paths"]
            st.session_state["last_result"] = response["result"]
            st.session_state["last_batch_response"] = None

    if st.session_state.get("last_result") is not None:
        st.markdown("---")
        render_simple_result(st.session_state["last_result"], st.session_state.get("last_saved_paths"))
    else:
        st.info("Complete the three steps above to run a single-sample AES analysis.")

else:
    st.subheader("Step 1 · Upload batch CSV")
    st.caption("Batch mode does not use the three single-sample upload boxes. Each row in the batch CSV should already contain file paths and metadata.")
    batch_csv_file = st.file_uploader("Upload batch CSV", type=["csv"], key="batch_csv_main")
    batch_df = None
    batch_missing = []
    if batch_csv_file is not None:
        batch_df = pd.read_csv(batch_csv_file)
        batch_missing = validate_batch_csv_columns(batch_df)
        if batch_missing:
            st.error("The batch CSV is missing required fields:")
            for c in batch_missing:
                st.write(f"- {c}")
        else:
            st.success(f"Batch CSV loaded: {len(batch_df)} rows")
            st.dataframe(batch_df.head(10), use_container_width=True)

    with st.expander("View required batch CSV fields", expanded=False):
        st.code("\n".join(BATCH_REQUIRED_COLUMNS), language="text")

    st.markdown("---")
    st.subheader("Step 2 · Review batch input")
    if batch_csv_file is None:
        st.info("Upload the batch CSV in Step 1 first.")
    elif batch_missing:
        st.warning("Please fix the missing columns before running batch analysis.")
    else:
        c1, c2 = st.columns(2)
        c1.metric("Batch rows", len(batch_df))
        c2.metric("Required columns", len(BATCH_REQUIRED_COLUMNS))
        st.success("Batch input structure looks valid.")

    st.markdown("---")
    st.subheader("Step 3 · Run batch analysis")
    if st.button("Run batch analysis"):
        if batch_csv_file is None:
            st.error("Please upload a batch CSV first.")
        elif batch_missing:
            st.error("Please fix the batch CSV columns first.")
        else:
            batch_csv_path = save_uploaded_file(batch_csv_file, category="batchcsv", temp_dir=TEMP_BASE_DIR)
            agent = Study1Agent(results_base_dir=results_base_dir)
            with st.spinner("Running batch analysis..."):
                batch_response = agent.run_batch(csv_path=batch_csv_path)
            if not batch_response.get("ok", False):
                st.error(batch_response.get("agent_message", "Batch run failed."))
            else:
                st.session_state["last_batch_response"] = batch_response
                st.session_state["last_result"] = None
                st.session_state["last_saved_paths"] = None

    if st.session_state.get("last_batch_response") is not None:
        st.markdown("---")
        render_batch_response(st.session_state["last_batch_response"])
    else:
        st.info("Upload a valid batch CSV and run the batch analysis.")
