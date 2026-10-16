
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from study2_agent import Study2Agent
from study2_code import (
    AI_NATURALNESS_KEY,
    BASE_OUTPUT_DIR,
    DEFAULT_ANALYSIS_MODEL,
    DEFAULT_MODEL,
    MULTIMODAL_ASSET_TYPES,
    NON_UNCANNY_NATURALNESS_KEY,
    VIDEO_INPUT_MODES,
    build_default_repair_patch,
    build_condition_prompt,
    default_hard_sell_controls,
    default_high_integration_controls,
    get_score_with_backward_compat,
    get_condition_controls,
    get_run_asset_dir,
    infer_regeneration_route,
    get_run_config_path,
    get_run_dir,
    get_run_table,
    load_analysis,
    load_run_config,
    normalize_multimodal_generation_options,
    resolve_multimodal_inputs,
    safe_read_json,
)

st.set_page_config(page_title="Study2 Workbench", layout="wide")
st.title("Study2 Workbench")
st.caption("Fixed candidate pool → automated video analysis → RAISE score suggestions → screening → final selection")


CUSTOM_CSS = """
<style>
:root {
    --card: rgba(15, 19, 29, 0.90);
    --text: #f4f7fb;
    --muted: #aeb8c9;
    --border: rgba(255,255,255,0.10);
}
@keyframes bgFloatA {
    0% { transform: translate3d(0, 0, 0) scale(1); }
    50% { transform: translate3d(40px, -30px, 0) scale(1.08); }
    100% { transform: translate3d(0, 0, 0) scale(1); }
}
@keyframes bgFloatB {
    0% { transform: translate3d(0, 0, 0) scale(1); }
    50% { transform: translate3d(-35px, 25px, 0) scale(1.12); }
    100% { transform: translate3d(0, 0, 0) scale(1); }
}
.stApp {
    background: linear-gradient(180deg, #07090f 0%, #0b1018 42%, #0e1521 100%);
    color: var(--text);
}
.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    background-image:
      linear-gradient(rgba(5, 8, 16, 0.62), rgba(5, 8, 16, 0.68)),
      url("https://images.pexels.com/photos/11565603/pexels-photo-11565603.jpeg?cs=srgb&dl=pexels-ilya-kovalchuk-164429912-11565603.jpg&fm=jpg");
    background-size: cover;
    background-position: center center;
    background-repeat: no-repeat;
    background-attachment: fixed;
    opacity: 0.82;
    transform: scale(1.02);
}
.stApp::after {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    background:
      radial-gradient(circle at 14% 18%, rgba(255,47,146,0.34), transparent 16%),
      radial-gradient(circle at 86% 14%, rgba(0,209,255,0.28), transparent 16%),
      radial-gradient(circle at 76% 78%, rgba(123,92,255,0.24), transparent 20%),
      linear-gradient(115deg, transparent 0 38%, rgba(255,47,146,0.08) 39%, transparent 43%),
      linear-gradient(295deg, transparent 0 52%, rgba(0,209,255,0.08) 53%, transparent 57%),
      repeating-linear-gradient(115deg, rgba(255,255,255,0.035) 0 2px, transparent 2px 88px),
      repeating-linear-gradient(25deg, rgba(255,255,255,0.022) 0 1px, transparent 1px 70px);
    mix-blend-mode: screen;
    opacity: 0.95;
}
.block-container, section[data-testid="stSidebar"] { position: relative; z-index: 1; }
.block-container { max-width: 1320px; padding-top: 2rem; padding-bottom: 2rem; }
section[data-testid="stSidebar"] { background: linear-gradient(180deg, rgba(11,14,21,0.98), rgba(9,12,19,0.98)); border-right: 1px solid var(--border); }
h1,h2,h3,label,p,span,div { color: var(--text); }
h1 { font-size: 2.35rem !important; font-weight: 800 !important; letter-spacing: -0.03em; }
h2 { font-size: 1.65rem !important; font-weight: 760 !important; }
h3 { font-size: 1.18rem !important; font-weight: 700 !important; }
.stTabs [data-baseweb="tab"] { background: rgba(255,255,255,0.04) !important; border: 1px solid rgba(255,255,255,0.08) !important; border-radius: 14px !important; }
.stTabs [aria-selected="true"] { background: linear-gradient(135deg, rgba(255,47,146,0.18), rgba(0,209,255,0.14)) !important; border-color: rgba(255,255,255,0.14) !important; }
div[data-testid="stForm"], div[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 20px !important;
    box-shadow: 0 14px 38px rgba(0,0,0,0.28) !important;
    backdrop-filter: blur(8px);
}
div[data-testid="stAlert"] { background: rgba(255,255,255,0.04) !important; border: 1px solid rgba(255,255,255,0.08) !important; border-radius: 16px !important; color: var(--text) !important; }
.stButton > button { border-radius: 14px !important; border: 1px solid rgba(255,255,255,0.10) !important; background: linear-gradient(135deg, rgba(255,47,146,0.20), rgba(0,209,255,0.16)) !important; color: #ffffff !important; font-weight: 650 !important; }
div[data-baseweb="input"],
div[data-baseweb="base-input"],
div[data-baseweb="textarea"],
div[data-baseweb="select"] > div,
.stTextInput > div > div,
.stNumberInput > div > div,
.stTextArea > div > div,
.stSelectbox > div > div {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 14px !important;
    color: #ffffff !important;
}
input, textarea { background: transparent !important; color: #ffffff !important; -webkit-text-fill-color: #ffffff !important; caret-color: #ffffff !important; }
input::placeholder, textarea::placeholder { color: rgba(255,255,255,0.50) !important; -webkit-text-fill-color: rgba(255,255,255,0.50) !important; }
[data-baseweb="select"] span, [data-baseweb="select"] div, [data-baseweb="select"] input { color: #ffffff !important; -webkit-text-fill-color: #ffffff !important; }
div[data-testid="stJson"],
div[data-testid="stJson"] * {
    background: rgba(255,255,255,0.05) !important;
    color: #f4f7fb !important;
    border-color: rgba(255,255,255,0.08) !important;
}
div[data-testid="stJson"] {
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 18px !important;
    overflow: hidden !important;
}
div[data-testid="stSidebar"] div[data-testid="stJson"],
div[data-testid="stSidebar"] div[data-testid="stJson"] * {
    background: rgba(255,255,255,0.04) !important;
    color: #f4f7fb !important;
}
pre, code {
    color: #ffd0e7 !important;
    background: rgba(255,255,255,0.05) !important;
}
pre {
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 16px !important;
}
div[data-testid="stDataFrame"] { border-radius: 16px !important; overflow: hidden !important; border: 1px solid rgba(255,255,255,0.10) !important; }
.app-banner { background: linear-gradient(135deg, rgba(8,18,20,0.88) 0%, rgba(10,28,24,0.84) 48%, rgba(8,20,30,0.86) 100%); border: 1px solid rgba(120,255,190,0.14); border-radius: 24px; padding: 1.25rem 1.45rem; box-shadow: 0 18px 38px rgba(0,0,0,0.30), inset 0 1px 0 rgba(255,255,255,0.04); backdrop-filter: blur(8px); margin-bottom: 1.2rem; }
.app-banner-title { color: white !important; font-size: 1.18rem; font-weight: 800; }
.app-banner-subtitle { color: rgba(255,255,255,0.95) !important; font-size: 0.98rem; }
.section-note { background: rgba(255,255,255,0.04) !important; border: 1px solid rgba(255,255,255,0.08) !important; border-radius: 16px !important; color: var(--muted) !important; }
header[data-testid="stHeader"] { background: transparent !important; height: 0 !important; }
button[kind="header"], [data-testid="stToolbar"] { display: none !important; }
div[data-baseweb="popover"], div[role="listbox"], ul[role="listbox"] {
    background: rgba(18,22,32,0.98) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: 14px !important;
}
div[role="option"], li[role="option"] { background: transparent !important; color: #ffffff !important; }
div[role="option"]:hover, li[role="option"]:hover { background: rgba(255,255,255,0.08) !important; }
.stTextArea textarea, textarea[data-testid="stTextArea"] { background: rgba(255,255,255,0.07) !important; color: #ffffff !important; }
[data-baseweb="layer"], [data-baseweb="modal"], [role="dialog"] { background: transparent !important; }
.stButton > button:disabled, button:disabled {
    background: rgba(255,255,255,0.08) !important;
    color: rgba(255,255,255,0.55) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    opacity: 1 !important;
}
[data-baseweb="popover"] *, [role="listbox"] *, ul[role="listbox"] * { background: transparent !important; color: #ffffff !important; }
[data-baseweb="popover"], [role="listbox"], ul[role="listbox"] {
    background: rgba(18,22,32,0.98) !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: 14px !important;
    box-shadow: 0 18px 38px rgba(0,0,0,0.35) !important;
}
body div[role="listbox"], body ul[role="listbox"], body div[data-baseweb="select"] + div {
    background: rgba(18,22,32,0.98) !important;
    color: #ffffff !important;
}
body div[role="option"], body li[role="option"] { background: rgba(18,22,32,0.98) !important; color: #ffffff !important; }
body div[role="option"][aria-selected="true"], body li[role="option"][aria-selected="true"] {
    background: rgba(255,255,255,0.10) !important;
    color: #ffffff !important;
}
button[data-testid="baseButton-secondary"],
button[data-testid="baseButton-primary"],
button[kind="secondary"],
button[kind="primary"],
form button,
.stForm button,
.stButton button {
    background: linear-gradient(135deg, rgba(255,47,146,0.20), rgba(0,209,255,0.16)) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    box-shadow: 0 8px 20px rgba(0,0,0,0.18) !important;
}
button[data-testid="baseButton-secondary"]:hover,
button[data-testid="baseButton-primary"]:hover,
button[kind="secondary"]:hover,
button[kind="primary"]:hover,
form button:hover,
.stForm button:hover,
.stButton button:hover {
    background: linear-gradient(135deg, rgba(255,47,146,0.30), rgba(0,209,255,0.22)) !important;
    color: #ffffff !important;
}
button[data-testid="baseButton-secondary"]:disabled,
button[data-testid="baseButton-primary"]:disabled,
button[kind="secondary"]:disabled,
button[kind="primary"]:disabled,
form button:disabled,
.stForm button:disabled,
.stButton button:disabled {
    background: rgba(255,255,255,0.08) !important;
    color: rgba(255,255,255,0.55) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    opacity: 1 !important;
    box-shadow: none !important;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.markdown(
    """
    <div class="app-banner">
        <div class="app-banner-title">Study 2 TikTok Stimulus Workbench</div>
        <div class="app-banner-subtitle">Generate candidate TikTok-style videos, inspect machine analysis, review RAISE score suggestions, and curate final stimulus selections.</div>
    </div>
    """,
    unsafe_allow_html=True,
)


RAISE_SCORE_KEYS = [
    "realism_score_1to7",
    "platform_fit_score_1to7",
    "integration_score_1to7",
    "commercial_salience_score_1to7",
    "character_consistency_score_1to7",
    AI_NATURALNESS_KEY,
    NON_UNCANNY_NATURALNESS_KEY,
]


study2_agent = Study2Agent(results_base_dir=BASE_OUTPUT_DIR)
PRODUCT_ENTRY_MODE_OPTIONS = ["incidental", "embedded_use", "direct_showcase", "pitch_opening"]
SPEAKING_MODE_OPTIONS = ["narrative", "direct_pitch", "mixed", "no_speech"]
PRODUCT_FOCUS_PATTERN_OPTIONS = ["background", "intermittent", "shared_focus", "dominant_focus"]
PERSUASIVE_INTENSITY_OPTIONS = ["low", "medium", "high"]
NARRATIVE_INTERRUPTION_OPTIONS = ["none", "mild", "strong"]
DIRECT_ADDRESS_OPTIONS = ["no", "partial", "yes"]
CALL_TO_ACTION_OPTIONS = ["none", "soft", "explicit"]
PRODUCT_CENTRALITY_OPTIONS = ["contextual", "supportive", "central", "dominant"]
REGENERATION_ROUTE_OPTIONS = ["structural_regenerate", "surface_regenerate"]


def _safe_df(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    normalized_rows = []
    for row in rows:
        normalized = dict(row)
        for key in RAISE_SCORE_KEYS:
            value = get_score_with_backward_compat(normalized, key, "")
            if str(value).strip() != "":
                normalized[key] = value
        normalized_rows.append(normalized)
    df = pd.DataFrame(normalized_rows)
    if "candidate_index" in df.columns:
        df["candidate_index"] = pd.to_numeric(df["candidate_index"], errors="coerce").fillna(0).astype(int)
    return df


@st.cache_data(show_spinner=False)
def load_runs_cached() -> pd.DataFrame:
    return _safe_df(study2_agent.list_runs())


@st.cache_data(show_spinner=False)
def load_run_table_cached(run_id: str) -> pd.DataFrame:
    return _safe_df(get_run_table(run_id))


@st.cache_data(show_spinner=False)
def load_analysis_cached(run_id: str, condition: str, candidate_index: int) -> Dict[str, Any]:
    return load_analysis(run_id, condition, candidate_index)


def clear_cache() -> None:
    load_runs_cached.clear()
    load_run_table_cached.clear()
    load_analysis_cached.clear()


def load_json_from_path(path_value: Any) -> Dict[str, Any]:
    path_text = str(path_value or "").strip()
    if not path_text:
        return {}
    path = Path(path_text)
    if not path.exists():
        return {}
    return safe_read_json(path)


def load_text_from_path(path_value: Any) -> str:
    path_text = str(path_value or "").strip()
    if not path_text:
        return ""
    path = Path(path_text)
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def save_uploaded_image_to_stem(uploaded_file: Any, directory: Path, stem: str) -> str:
    if uploaded_file is None:
        return ""
    ensure_exts = [".png", ".jpg", ".jpeg", ".webp"]
    directory.mkdir(parents=True, exist_ok=True)
    suffix = Path(str(getattr(uploaded_file, "name", "") or "")).suffix.lower()
    if suffix not in ensure_exts:
        suffix = ".png"
    for ext in ensure_exts:
        existing = directory / f"{stem}{ext}"
        if existing.exists():
            try:
                existing.unlink()
            except Exception:
                pass
    target = directory / f"{stem}{suffix}"
    target.write_bytes(uploaded_file.getbuffer())
    return str(target)


def select_enum(label: str, options: List[str], value: str, key: str) -> str:
    safe_value = value if value in options else options[0]
    return st.selectbox(label, options=options, index=options.index(safe_value), key=key)


def pick_row(df: pd.DataFrame, condition: str, candidate_index: int) -> Optional[pd.Series]:
    if df.empty:
        return None
    match = df[
        (df["condition"].astype(str) == str(condition))
        & (pd.to_numeric(df["candidate_index"], errors="coerce").fillna(0).astype(int) == int(candidate_index))
    ]
    if match.empty:
        return None
    return match.iloc[0]


def condition_label(condition: str) -> str:
    return "High Integration" if condition == "high_integration" else "Hard Sell"


def _score_default(row: pd.Series, analysis: Dict[str, Any], key: str, fallback: int = 4) -> int:
    analysis_result = analysis.get("analysis_result", {}) or {}
    analysis_scores = analysis_result.get("raise_scoring", {}) or analysis_result.get("scores", {}) or {}
    raw = get_score_with_backward_compat(row.to_dict() if hasattr(row, "to_dict") else {}, key, row.get(key, ""))
    if str(raw).strip() == "":
        raw = get_score_with_backward_compat(analysis_scores, key, fallback)
    try:
        return max(1, min(7, int(float(raw))))
    except Exception:
        return fallback


def _saved_score_value(row: pd.Series, key: str) -> Optional[int]:
    raw = get_score_with_backward_compat(row.to_dict() if hasattr(row, "to_dict") else {}, key, row.get(key, ""))
    if str(raw).strip() == "":
        return None
    try:
        return max(1, min(7, int(float(raw))))
    except Exception:
        return None


def _keep_default(row: pd.Series, analysis: Dict[str, Any]) -> str:
    current_keep = str(row.get("overall_keep_yes_no", "")).strip().lower()
    if current_keep in {"yes", "no"}:
        return current_keep
    analysis_keep = str((analysis.get("analysis_result", {}) or {}).get("keep_recommendation", "")).strip().lower()
    return analysis_keep if analysis_keep in {"yes", "no"} else ""


def _notes_default(row: pd.Series, analysis: Dict[str, Any]) -> str:
    current_notes = str(row.get("notes", "") or "")
    if current_notes.strip():
        return current_notes
    analysis_result = analysis.get("analysis_result", {}) or {}
    parts = []
    if analysis_result.get("summary"):
        parts.append(str(analysis_result.get("summary")))
    if analysis_result.get("main_issue"):
        parts.append(f"Main issue: {analysis_result.get('main_issue')}")
    return "\n\n".join(parts).strip()


def _working_keep_default(row: pd.Series, analysis: Dict[str, Any]) -> str:
    return _keep_default(row, analysis)


def _machine_analysis_snapshot(analysis: Dict[str, Any]) -> Dict[str, Any]:
    if not analysis or analysis.get("analysis_status") != "ANALYSIS_READY":
        return {}
    result = analysis.get("analysis_result", {}) or {}
    diagnostic_focus = result.get("diagnostic_focus", {}) or {}
    return {
        "summary": result.get("summary", ""),
        "main_issue": result.get("main_issue", ""),
        "revision_priority": result.get("revision_priority", ""),
        "keep_recommendation": result.get("keep_recommendation", ""),
        "strongest_dimensions": diagnostic_focus.get("strongest_dimensions", []),
        "weakest_dimensions": diagnostic_focus.get("weakest_dimensions", []),
        "condition_success_dimensions": diagnostic_focus.get("condition_success_dimensions", []),
        "validity_risks": diagnostic_focus.get("validity_risks", []),
    }


def _machine_score_df(analysis: Dict[str, Any]) -> pd.DataFrame:
    if not analysis or analysis.get("analysis_status") != "ANALYSIS_READY":
        return pd.DataFrame()
    result = analysis.get("analysis_result", {}) or {}
    scores = result.get("raise_scoring", {}) or result.get("scores", {}) or {}
    if not scores:
        return pd.DataFrame()
    return pd.DataFrame(
        [{"dimension": key, "machine_score": scores.get(key, "")} for key in RAISE_SCORE_KEYS]
    )


def _machine_score_value(analysis: Dict[str, Any], key: str, fallback: int = 4) -> int:
    analysis_result = analysis.get("analysis_result", {}) or {}
    analysis_scores = analysis_result.get("raise_scoring", {}) or analysis_result.get("scores", {}) or {}
    try:
        return max(1, min(7, int(float(analysis_scores.get(key, fallback)))))
    except Exception:
        return fallback


def _machine_score_map(analysis: Dict[str, Any]) -> Dict[str, int]:
    return {key: _machine_score_value(analysis, key) for key in RAISE_SCORE_KEYS}


def _manager_score_widget_keys(run_id: str, condition: str, candidate_index: int) -> Dict[str, str]:
    prefix = f"mgr_{run_id}_{condition}_{int(candidate_index)}"
    return {
        "analysis_signature": f"{prefix}_analysis_signature",
        "score_source": f"{prefix}_score_source",
        "realism_score_1to7": f"{prefix}_realism",
        "platform_fit_score_1to7": f"{prefix}_platform_fit",
        "integration_score_1to7": f"{prefix}_integration",
        "commercial_salience_score_1to7": f"{prefix}_commercial_salience",
        "character_consistency_score_1to7": f"{prefix}_character_consistency",
        AI_NATURALNESS_KEY: f"{prefix}_ai_naturalness",
        NON_UNCANNY_NATURALNESS_KEY: f"{prefix}_non_uncanny_naturalness",
        "working_keep": f"{prefix}_working_keep",
        "working_notes": f"{prefix}_working_notes",
        "revision_text": f"{prefix}_revision_text",
        "regeneration_route": f"{prefix}_regeneration_route",
        "video_input_mode": f"{prefix}_video_input_mode",
        "selected_asset_types": f"{prefix}_selected_asset_types",
        "selected_storyboard_panel_ids": f"{prefix}_selected_storyboard_panel_ids",
        "selected_first_frame_panel_id": f"{prefix}_selected_first_frame_panel_id",
        "selected_last_frame_panel_id": f"{prefix}_selected_last_frame_panel_id",
        "use_storyboard_panel_as_first_frame": f"{prefix}_use_storyboard_panel_as_first_frame",
        "use_storyboard_panel_as_last_frame": f"{prefix}_use_storyboard_panel_as_last_frame",
    }


def _sync_candidate_manager_state(
    run_id: str,
    condition: str,
    candidate_index: int,
    row: pd.Series,
    analysis: Dict[str, Any],
) -> Dict[str, Any]:
    widget_keys = _manager_score_widget_keys(run_id, condition, candidate_index)
    run_config = load_run_config(run_id)
    machine_scores = _machine_score_map(analysis)
    analysis_ready = analysis.get("analysis_status") == "ANALYSIS_READY"
    analysis_signature = str(analysis.get("analyzed_at_epoch", "") or "") if analysis_ready else ""
    previous_signature = str(st.session_state.get(widget_keys["analysis_signature"], "") or "")

    if analysis_ready and analysis_signature and analysis_signature != previous_signature:
        for score_key in RAISE_SCORE_KEYS:
            st.session_state[widget_keys[score_key]] = machine_scores.get(score_key, 4)
        st.session_state[widget_keys["analysis_signature"]] = analysis_signature
        st.session_state[widget_keys["score_source"]] = "machine"
    else:
        if widget_keys["score_source"] not in st.session_state:
            st.session_state[widget_keys["score_source"]] = ""
        for score_key in RAISE_SCORE_KEYS:
            if widget_keys[score_key] in st.session_state:
                continue
            if analysis_ready:
                st.session_state[widget_keys[score_key]] = machine_scores.get(score_key, 4)
                st.session_state[widget_keys["score_source"]] = "machine"
            else:
                saved_score = _saved_score_value(row, score_key)
                st.session_state[widget_keys[score_key]] = saved_score if saved_score is not None else 4
                if saved_score is not None and not st.session_state[widget_keys["score_source"]]:
                    st.session_state[widget_keys["score_source"]] = "saved_review"
        if analysis_ready and analysis_signature and not previous_signature:
            st.session_state[widget_keys["analysis_signature"]] = analysis_signature

    keep_default = _working_keep_default(row, analysis)
    if widget_keys["working_keep"] not in st.session_state:
        st.session_state[widget_keys["working_keep"]] = keep_default if keep_default in {"yes", "no"} else ""
    if widget_keys["working_notes"] not in st.session_state:
        st.session_state[widget_keys["working_notes"]] = _notes_default(row, analysis)
    if widget_keys["revision_text"] not in st.session_state:
        st.session_state[widget_keys["revision_text"]] = _revision_text_default(row, analysis)
    inferred_route = infer_regeneration_route((analysis.get("analysis_result", {}) or {})) if analysis_ready else "surface_regenerate"
    if analysis_ready and analysis_signature and analysis_signature != previous_signature:
        st.session_state[widget_keys["regeneration_route"]] = inferred_route
    elif widget_keys["regeneration_route"] not in st.session_state:
        st.session_state[widget_keys["regeneration_route"]] = inferred_route

    multimodal_defaults = normalize_multimodal_generation_options(
        {"generation_mode": row.get("video_input_mode", "")},
        run_config,
    )
    st.session_state.setdefault(widget_keys["video_input_mode"], multimodal_defaults["generation_mode"])
    st.session_state.setdefault(widget_keys["selected_asset_types"], multimodal_defaults["selected_asset_types"])
    st.session_state.setdefault(widget_keys["selected_storyboard_panel_ids"], [])
    st.session_state.setdefault(widget_keys["selected_first_frame_panel_id"], "")
    st.session_state.setdefault(widget_keys["selected_last_frame_panel_id"], "")
    st.session_state.setdefault(widget_keys["use_storyboard_panel_as_first_frame"], multimodal_defaults["use_storyboard_panel_as_first_frame"])
    st.session_state.setdefault(widget_keys["use_storyboard_panel_as_last_frame"], multimodal_defaults["use_storyboard_panel_as_last_frame"])

    return {
        "widget_keys": widget_keys,
        "analysis_ready": analysis_ready,
        "machine_scores": machine_scores,
        "score_source": str(st.session_state.get(widget_keys["score_source"], "") or ""),
        "inferred_route": inferred_route,
    }


def _current_editable_score_targets(widget_keys: Dict[str, str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for score_key in RAISE_SCORE_KEYS:
        try:
            out[score_key] = max(1, min(7, int(float(st.session_state.get(widget_keys[score_key], 4)))))
        except Exception:
            out[score_key] = 4
    return out


def _revision_text_default(row: pd.Series, analysis: Dict[str, Any]) -> str:
    existing_revision_text = str(row.get("regeneration_instruction_text", "") or "").strip()
    if existing_revision_text:
        return existing_revision_text
    if analysis.get("analysis_status") != "ANALYSIS_READY":
        return ""
    repair_patch = build_default_repair_patch(analysis.get("analysis_result", {}) or {})
    if repair_patch:
        return repair_patch
    suggestions = analysis.get("analysis_result", {}).get("prompt_revision_suggestions", []) or []
    if not suggestions:
        return ""
    return "\n".join(f"- {x}" for x in suggestions)


def short_error_summary(error_text: Any, blueprint_status: str = "", status: str = "") -> str:
    text = str(error_text or "").strip()
    if not text:
        return ""
    upper_text = text.upper()
    if "503" in upper_text or "UNAVAILABLE" in upper_text:
        if str(blueprint_status).upper() == "BLUEPRINT_FAILED":
            return "Blueprint failed because the model was temporarily unavailable (503)."
        if str(status).upper() == "VIDEO_FAILED":
            return "Video generation failed because the model was temporarily unavailable (503)."
        return "Model temporarily unavailable (503)."
    if str(blueprint_status).upper() == "BLUEPRINT_FAILED":
        attempts = len([line for line in text.split("|") if line.strip()]) or "multiple"
        return f"Blueprint failed after {attempts} attempt(s)."
    if str(status).upper() == "VIDEO_FAILED":
        return "Video generation failed."
    if len(text) > 140:
        return text[:137] + "..."
    return text


def render_candidate_summary_card(row: pd.Series) -> None:
    short_error = short_error_summary(
        row.get("error_message", ""),
        blueprint_status=str(row.get("blueprint_status", "")),
        status=str(row.get("status", "")),
    )
    st.markdown("### Candidate Summary")
    metric_cols = st.columns(3)
    metric_cols[0].metric("Video Status", str(row.get("status", "") or "-"))
    metric_cols[1].metric("Blueprint", str(row.get("blueprint_status", "") or "-"))
    metric_cols[2].metric("Analysis", str(row.get("analysis_status", "") or "-"))
    st.write(
        {
            "blueprint_attempts": row.get("blueprint_attempts", ""),
            "control_compliance": row.get("control_compliance_ok", ""),
            "keep_suggestion": row.get("analysis_keep_recommendation", ""),
        }
    )
    if short_error:
        st.warning(short_error)


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe_value(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe_value(v) for v in value]
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return value


def show_video_if_exists(video_path: str) -> None:
    if video_path and Path(video_path).exists():
        st.video(video_path)
    else:
        st.warning("Video file is missing or the path is invalid.")


def render_analysis_panel(analysis: Dict[str, Any], title: str = "Video Analysis") -> None:
    st.markdown(f"### {title}")

    if not analysis:
        st.info("No analysis result is available yet.")
        return

    if analysis.get("analysis_status") != "ANALYSIS_READY":
        st.warning(f"Analysis status: {analysis.get('analysis_status', 'UNKNOWN')}")
        if analysis.get("error_message"):
            st.code(str(analysis.get("error_message", "")))
        return

    result = analysis.get("analysis_result", {}) or {}
    observations = result.get("objective_observations", {}) or {}
    scores = result.get("raise_scoring", {}) or result.get("scores", {}) or {}
    score_justifications = result.get("score_justifications", {}) or {}
    diagnostic_focus = result.get("diagnostic_focus", {}) or {}

    st.markdown("**Summary**")
    st.write(result.get("summary", "") or "No summary available.")

    detail_cols = st.columns(2)
    with detail_cols[0]:
        st.markdown("**Main Issue**")
        st.write(result.get("main_issue", "") or "No main issue identified.")
        st.markdown("**Keep Recommendation**")
        st.write(result.get("keep_recommendation", "") or "-")
    with detail_cols[1]:
        st.markdown("**Revision Priority**")
        st.write(result.get("revision_priority", "") or "-")
        if diagnostic_focus.get("strongest_dimensions"):
            st.markdown("**Strongest Dimensions**")
            for item in diagnostic_focus.get("strongest_dimensions", []):
                st.markdown(f"- {item}")
        if diagnostic_focus.get("weakest_dimensions"):
            st.markdown("**Weakest Dimensions**")
            for item in diagnostic_focus.get("weakest_dimensions", []):
                st.markdown(f"- {item}")

    st.markdown("**Objective Observations**")
    observation_lines = [
        f"Scene: {observations.get('scene_summary', '')}",
        f"Narrative flow: {observations.get('narrative_flow_summary', '')}",
        f"Product first appearance: {observations.get('product_first_appearance_seconds', '')} seconds",
        f"Speaking style: {observations.get('speaking_style', '')}",
        f"Product visible most of video: {observations.get('product_visible_most_of_video', False)}",
        f"Product is visual focal point: {observations.get('product_is_visual_focal_point', False)}",
        f"Unnatural motion present: {observations.get('unnatural_motion_present', False)}",
        f"Character identity consistent: {observations.get('character_identity_consistent', False)}",
        f"Object continuity ok: {observations.get('object_continuity_ok', True)}",
        f"Hand-object interaction consistent: {observations.get('hand_object_interaction_consistent', True)}",
        f"Unexpected extra objects present: {observations.get('unexpected_extra_objects_present', False)}",
        f"Prop identity stable: {observations.get('prop_identity_stable', True)}",
        f"Temporal coherence ok: {observations.get('temporal_coherence_ok', True)}",
    ]
    for line in observation_lines:
        st.markdown(f"- {line}")

    if observations.get("key_observations"):
        st.markdown("**Key Observations**")
        for item in observations.get("key_observations", []):
            st.markdown(f"- {item}")

    if result.get("prompt_revision_suggestions"):
        st.markdown("**Machine Revision Suggestions**")
        for item in result.get("prompt_revision_suggestions", []):
            st.markdown(f"- {item}")

    with st.expander("View Full Analysis JSON", expanded=False):
        st.json(analysis)
    if score_justifications:
        with st.expander("View Score Justifications", expanded=False):
            just_rows = []
            for k in RAISE_SCORE_KEYS:
                just_rows.append({
                    "dimension": k,
                    "score": scores.get(k, ""),
                    "justification": score_justifications.get(k, ""),
                })
            st.dataframe(pd.DataFrame(just_rows), use_container_width=True, height=320)


def safe_selectbox(label: str, options: List[Any], key: str):
    if not options:
        st.warning(f"No options available for {label}.")
        return None
    return st.selectbox(label, options=options, key=key)


runs_df = load_runs_cached()
run_options = runs_df["run_id"].tolist() if not runs_df.empty else []

if "selected_run_id" not in st.session_state:
    st.session_state.selected_run_id = run_options[0] if run_options else ""
if "create_new_run" not in st.session_state:
    st.session_state.create_new_run = (len(run_options) == 0)
if "pending_run_switch" in st.session_state:
    st.session_state.selected_run_id = str(st.session_state.pop("pending_run_switch") or "")
    st.session_state.create_new_run = False

st.sidebar.header("Run")
create_new = st.sidebar.checkbox("Create New Run", key="create_new_run")

if not create_new and run_options:
    default_index = run_options.index(st.session_state.selected_run_id) if st.session_state.selected_run_id in run_options else 0
    st.session_state.selected_run_id = st.sidebar.selectbox("Select Existing Run", options=run_options, index=default_index)
elif create_new:
    st.session_state.selected_run_id = ""

selected_run_id = st.session_state.selected_run_id

tab_setup, tab_manager, tab_screening, tab_final = st.tabs(
    ["Run Setup & Generate", "Candidate Manager", "Screening", "Final Selection"]
)

with tab_setup:
    st.subheader("1) Run Setup & Generation")
    st.markdown("<div class='section-note'>Create a run, define shared stimulus settings, and generate a fixed candidate pool for later review.</div>", unsafe_allow_html=True)
    st.info("Use this page to create a run, save shared settings, upload shared image references, and generate the fixed candidate pool.")

    base_config: Dict[str, Any] = {
        "run_id": "",
        "model": DEFAULT_MODEL,
        "analysis_model": DEFAULT_ANALYSIS_MODEL,
        "product_name": "",
        "product_category": "",
        "influencer_profile": "",
        "scene_context": "",
        "duration_hint": "8-second short-form vertical video",
        "aspect_ratio": "9:16",
        "visual_style_lock": "TikTok-native realism, smartphone aesthetics, natural lighting, platform-authentic styling",
        "camera_style": "handheld vertical smartphone framing, casual creator-style composition, slight movement, not cinematic",
        "tone_lock": "realistic, casual, relatable, social-media-native, not glossy commercial",
        "negative_constraints": "avoid luxury beauty ad style, avoid cinematic ad polish, avoid unrealistic transitions, avoid product-only packshot dominance unless required by condition",
        "n_candidates_high_integration": 2,
        "n_candidates_hard_sell": 2,
        "video_input_mode": "text_only",
        "use_storyboard_panel_as_first_frame": True,
        "use_storyboard_panel_as_last_frame": False,
        "selected_reference_asset_types": ["scene_reference", "product_reference", "character_reference", "style_reference"],
        "high_integration_controls": default_high_integration_controls(),
        "hard_sell_controls": default_hard_sell_controls(),
    }
    if selected_run_id:
        base_config.update(load_run_config(selected_run_id))

    with st.form("run_setup_form"):
        c1, c2, c3 = st.columns(3)
        run_id_input = c1.text_input("run_id (leave blank to auto-create)", value="" if create_new else str(base_config.get("run_id", "")))
        model = c2.text_input("generation model", value=str(base_config.get("model", DEFAULT_MODEL)))
        analysis_model = c3.text_input("analysis model", value=str(base_config.get("analysis_model", DEFAULT_ANALYSIS_MODEL)))

        c4, c5 = st.columns(2)
        product_name = c4.text_input("product_name", value=str(base_config.get("product_name", "")))
        product_category = c5.text_input("product_category", value=str(base_config.get("product_category", "")))

        influencer_profile = st.text_area("influencer_profile", value=str(base_config.get("influencer_profile", "")), height=90)
        scene_context = st.text_area("scene_context", value=str(base_config.get("scene_context", "")), height=100)

        c6, c7 = st.columns(2)
        duration_hint = c6.text_input("duration_hint", value=str(base_config.get("duration_hint", "8-second short-form vertical video")))
        aspect_ratio = c7.text_input("aspect_ratio", value=str(base_config.get("aspect_ratio", "9:16")))

        visual_style_lock = st.text_area("visual_style_lock", value=str(base_config.get("visual_style_lock", "")), height=80)
        camera_style = st.text_area("camera_style", value=str(base_config.get("camera_style", "")), height=80)
        tone_lock = st.text_area("tone_lock", value=str(base_config.get("tone_lock", "")), height=80)
        negative_constraints = st.text_area("negative_constraints", value=str(base_config.get("negative_constraints", "")), height=100)

        st.markdown("### Multimodal Video Inputs")
        video_input_mode = st.selectbox(
            "Default generation mode",
            options=VIDEO_INPUT_MODES,
            index=VIDEO_INPUT_MODES.index(str(base_config.get("video_input_mode", "text_only"))) if str(base_config.get("video_input_mode", "text_only")) in VIDEO_INPUT_MODES else 0,
        )
        selected_reference_asset_types = st.multiselect(
            "Default shared reference asset types",
            options=MULTIMODAL_ASSET_TYPES,
            default=[item for item in (base_config.get("selected_reference_asset_types") or []) if item in MULTIMODAL_ASSET_TYPES] or ["scene_reference", "product_reference", "character_reference", "style_reference"],
        )
        mm1, mm2 = st.columns(2)
        use_storyboard_panel_as_first_frame = mm1.checkbox(
            "Prefer storyboard panel as first frame when available",
            value=bool(base_config.get("use_storyboard_panel_as_first_frame", True)),
        )
        use_storyboard_panel_as_last_frame = mm2.checkbox(
            "Use storyboard panel as optional last frame when available",
            value=bool(base_config.get("use_storyboard_panel_as_last_frame", False)),
        )
        st.caption("Shared reference uploads are optional. If present, they can be reused as Veo multimodal inputs across both conditions.")
        up1, up2 = st.columns(2)
        scene_reference_upload = up1.file_uploader("Scene reference image", type=["png", "jpg", "jpeg", "webp"], key="scene_reference_upload")
        product_reference_upload = up2.file_uploader("Product reference image", type=["png", "jpg", "jpeg", "webp"], key="product_reference_upload")
        up3, up4 = st.columns(2)
        character_reference_upload = up3.file_uploader("Character reference image", type=["png", "jpg", "jpeg", "webp"], key="character_reference_upload")
        style_reference_upload = up4.file_uploader("Style reference image", type=["png", "jpg", "jpeg", "webp"], key="style_reference_upload")

        st.markdown("### Explicit Condition Controls")
        hi_controls = dict(base_config.get("high_integration_controls") or default_high_integration_controls())
        hs_controls = dict(base_config.get("hard_sell_controls") or default_hard_sell_controls())
        hi_col, hs_col = st.columns(2)
        with hi_col:
            st.markdown("**High Integration Controls**")
            hi_timing = st.number_input(
                "HI product_entry_timing_seconds",
                min_value=0.0,
                value=float(hi_controls.get("product_entry_timing_seconds", 3.5)),
                step=0.1,
            )
            hi_entry_mode = select_enum("HI product_entry_mode", PRODUCT_ENTRY_MODE_OPTIONS, str(hi_controls.get("product_entry_mode", "")), "hi_entry_mode")
            hi_speaking = select_enum("HI speaking_mode", SPEAKING_MODE_OPTIONS, str(hi_controls.get("speaking_mode", "")), "hi_speaking")
            hi_focus = select_enum("HI product_focus_pattern", PRODUCT_FOCUS_PATTERN_OPTIONS, str(hi_controls.get("product_focus_pattern", "")), "hi_focus")
            hi_intensity = select_enum("HI persuasive_intensity", PERSUASIVE_INTENSITY_OPTIONS, str(hi_controls.get("persuasive_intensity", "")), "hi_intensity")
            hi_interrupt = select_enum("HI narrative_interruption", NARRATIVE_INTERRUPTION_OPTIONS, str(hi_controls.get("narrative_interruption", "")), "hi_interrupt")
            hi_address = select_enum("HI direct_address_to_camera", DIRECT_ADDRESS_OPTIONS, str(hi_controls.get("direct_address_to_camera", "")), "hi_address")
            hi_cta = select_enum("HI call_to_action_presence", CALL_TO_ACTION_OPTIONS, str(hi_controls.get("call_to_action_presence", "")), "hi_cta")
            hi_centrality = select_enum("HI product_centrality", PRODUCT_CENTRALITY_OPTIONS, str(hi_controls.get("product_centrality", "")), "hi_centrality")
        with hs_col:
            st.markdown("**Hard Sell Controls**")
            hs_timing = st.number_input(
                "HS product_entry_timing_seconds",
                min_value=0.0,
                value=float(hs_controls.get("product_entry_timing_seconds", 0.8)),
                step=0.1,
            )
            hs_entry_mode = select_enum("HS product_entry_mode", PRODUCT_ENTRY_MODE_OPTIONS, str(hs_controls.get("product_entry_mode", "")), "hs_entry_mode")
            hs_speaking = select_enum("HS speaking_mode", SPEAKING_MODE_OPTIONS, str(hs_controls.get("speaking_mode", "")), "hs_speaking")
            hs_focus = select_enum("HS product_focus_pattern", PRODUCT_FOCUS_PATTERN_OPTIONS, str(hs_controls.get("product_focus_pattern", "")), "hs_focus")
            hs_intensity = select_enum("HS persuasive_intensity", PERSUASIVE_INTENSITY_OPTIONS, str(hs_controls.get("persuasive_intensity", "")), "hs_intensity")
            hs_interrupt = select_enum("HS narrative_interruption", NARRATIVE_INTERRUPTION_OPTIONS, str(hs_controls.get("narrative_interruption", "")), "hs_interrupt")
            hs_address = select_enum("HS direct_address_to_camera", DIRECT_ADDRESS_OPTIONS, str(hs_controls.get("direct_address_to_camera", "")), "hs_address")
            hs_cta = select_enum("HS call_to_action_presence", CALL_TO_ACTION_OPTIONS, str(hs_controls.get("call_to_action_presence", "")), "hs_cta")
            hs_centrality = select_enum("HS product_centrality", PRODUCT_CENTRALITY_OPTIONS, str(hs_controls.get("product_centrality", "")), "hs_centrality")

        c9, c10 = st.columns(2)
        n_hi = c9.number_input("n_candidates_high_integration", min_value=0, value=int(base_config.get("n_candidates_high_integration", 2)), step=1)
        n_hs = c10.number_input("n_candidates_hard_sell", min_value=0, value=int(base_config.get("n_candidates_hard_sell", 2)), step=1)

        save_only = st.form_submit_button("Save Run Configuration")
        save_and_generate = st.form_submit_button("Save and Generate Candidate Pool")

    if save_only or save_and_generate:
        save_result = study2_agent.create_or_update_run(
            {
                "run_id": run_id_input.strip(),
                "model": model.strip() or DEFAULT_MODEL,
                "analysis_model": analysis_model.strip() or DEFAULT_ANALYSIS_MODEL,
                "product_name": product_name.strip(),
                "product_category": product_category.strip(),
                "influencer_profile": influencer_profile.strip(),
                "scene_context": scene_context.strip(),
                "duration_hint": duration_hint.strip(),
                "aspect_ratio": aspect_ratio.strip(),
                "visual_style_lock": visual_style_lock.strip(),
                "camera_style": camera_style.strip(),
                "tone_lock": tone_lock.strip(),
                "negative_constraints": negative_constraints.strip(),
                "n_candidates_high_integration": int(n_hi),
                "n_candidates_hard_sell": int(n_hs),
                "video_input_mode": video_input_mode,
                "use_storyboard_panel_as_first_frame": bool(use_storyboard_panel_as_first_frame),
                "use_storyboard_panel_as_last_frame": bool(use_storyboard_panel_as_last_frame),
                "selected_reference_asset_types": list(selected_reference_asset_types),
                "high_integration_controls": {
                    "product_entry_timing_seconds": float(hi_timing),
                    "product_entry_mode": hi_entry_mode,
                    "speaking_mode": hi_speaking,
                    "product_focus_pattern": hi_focus,
                    "persuasive_intensity": hi_intensity,
                    "narrative_interruption": hi_interrupt,
                    "direct_address_to_camera": hi_address,
                    "call_to_action_presence": hi_cta,
                    "product_centrality": hi_centrality,
                },
                "hard_sell_controls": {
                    "product_entry_timing_seconds": float(hs_timing),
                    "product_entry_mode": hs_entry_mode,
                    "speaking_mode": hs_speaking,
                    "product_focus_pattern": hs_focus,
                    "persuasive_intensity": hs_intensity,
                    "narrative_interruption": hs_interrupt,
                    "direct_address_to_camera": hs_address,
                    "call_to_action_presence": hs_cta,
                    "product_centrality": hs_centrality,
                },
            }
        )

        if not save_result["ok"]:
            st.error(save_result["agent_message"])
        else:
            cfg = save_result["run_config"]
            st.session_state["pending_run_switch"] = str(cfg["run_id"])
            selected_run_id = str(cfg["run_id"])
            shared_asset_dir = get_run_asset_dir(selected_run_id)
            uploaded_any = False
            for stem, uploaded in [
                ("scene_reference", scene_reference_upload),
                ("product_reference", product_reference_upload),
                ("character_reference", character_reference_upload),
                ("style_reference", style_reference_upload),
            ]:
                if uploaded is not None:
                    save_uploaded_image_to_stem(uploaded, shared_asset_dir, stem)
                    uploaded_any = True
            clear_cache()
            st.success(f"Configuration saved: {selected_run_id}")
            if uploaded_any:
                st.info("Shared reference images were saved for this run.")

            if save_and_generate:
                current_df = load_run_table_cached(selected_run_id)
                if not current_df.empty:
                    st.error("This run already has existing candidates. To preserve a fixed candidate pool, create a new run instead of batch-generating again in the same run.")
                else:
                    with st.spinner("Generating the fixed candidate pool..."):
                        generation_result = study2_agent.generate_candidate_pool(
                            run_id=selected_run_id,
                            auto_analyze=False,
                        )
                    clear_cache()
                    if generation_result["ok"]:
                        st.success("The fixed candidate pool has been generated. Continue to Candidate Manager or Screening.")
                        st.rerun()
                    st.error(generation_result["agent_message"])
            else:
                st.rerun()

    if selected_run_id:
        st.markdown("### Current run config")
        st.json(load_run_config(selected_run_id))
        shared_asset_dir = get_run_asset_dir(selected_run_id)
        with st.expander("View Existing Shared Reference Images", expanded=False):
            found_any = False
            for stem in ["scene_reference", "product_reference", "character_reference", "style_reference"]:
                for ext in [".png", ".jpg", ".jpeg", ".webp"]:
                    candidate = shared_asset_dir / f"{stem}{ext}"
                    if candidate.exists():
                        found_any = True
                        st.markdown(f"**{stem}**")
                        st.image(str(candidate), use_container_width=True)
                        break
            if not found_any:
                st.info("No shared reference images have been uploaded or manually placed yet.")

run_df = load_run_table_cached(selected_run_id) if selected_run_id else pd.DataFrame()

if selected_run_id:
    run_summary = study2_agent.get_run_summary(selected_run_id)
    st.sidebar.markdown("### Current run Summary")
    st.sidebar.write(
        {
            **run_summary,
            "n_rows": int(len(run_df)),
            "config_exists": get_run_config_path(selected_run_id).exists(),
        }
    )

with tab_manager:
    st.subheader("2) Candidate Manager")
    st.markdown("<div class='section-note'>Review one candidate at a time, inspect machine analysis, and regenerate only when a targeted revision is necessary.</div>", unsafe_allow_html=True)
    st.info("This is now the main iteration workspace: analyze the candidate, inspect machine RAISE, save a working human review, refine revision instructions, regenerate, and repeat.")

    if not selected_run_id:
        st.info("Select or create a run from the left sidebar first.")
    elif run_df.empty:
        st.warning("This run does not contain any candidates yet. Generate them from the first tab.")
    else:
        c1, c2 = st.columns(2)
        cond_options = sorted(run_df["condition"].astype(str).unique().tolist())
        cond = safe_selectbox("Condition", cond_options, key="mgr_cond")
        if cond is not None:
            idxs = sorted(run_df.loc[run_df["condition"].astype(str) == cond, "candidate_index"].astype(int).unique().tolist())
            cand_idx = safe_selectbox("candidate_index", idxs, key="mgr_idx")
        else:
            cand_idx = None

        if cond is not None and cand_idx is not None:
            row = pick_row(run_df, cond, int(cand_idx))

            if row is not None:
                left, right = st.columns([1.2, 1])
                analysis = load_analysis_cached(selected_run_id, cond, int(cand_idx))
                candidate_status = str(row.get("status", "")).strip().upper()
                blueprint_status = str(row.get("blueprint_status", "")).strip().upper()
                output_video_path = str(row.get("output_video_path", "") or "")
                video_ready_for_analysis = candidate_status == "VIDEO_READY" and bool(output_video_path) and Path(output_video_path).exists()
                with left:
                    st.markdown(f"### {condition_label(cond)} / cand_{int(cand_idx):02d}")
                    if video_ready_for_analysis:
                        show_video_if_exists(output_video_path)
                    elif output_video_path and Path(output_video_path).exists():
                        st.warning("A previous video file exists on disk, but this candidate's current status is not VIDEO_READY. The latest attempt did not complete successfully.")
                    else:
                        st.info("No current video is available for this candidate yet.")
                    render_analysis_panel(analysis, title="Current Candidate Analysis")
                with right:
                    saved_condition_controls = get_condition_controls(load_run_config(selected_run_id), cond)
                    blueprint_payload = load_json_from_path(row.get("blueprint_json_path", ""))
                    blueprint_validation = load_json_from_path(row.get("blueprint_validation_json_path", ""))
                    asset_pack = load_json_from_path(row.get("asset_pack_json_path", ""))
                    storyboard_package = load_json_from_path(row.get("storyboard_plan_json_path", ""))
                    execution_script = load_json_from_path(row.get("execution_script_json_path", ""))
                    execution_script_text = load_text_from_path(row.get("execution_script_text_path", ""))
                    final_generation_package = load_json_from_path(row.get("final_generation_package_json_path", ""))
                    validated_blueprint = (
                        blueprint_validation.get("normalized_blueprint", {})
                        if isinstance(blueprint_validation.get("normalized_blueprint"), dict)
                        else {}
                    )
                    validation_errors = blueprint_validation.get("errors", []) if isinstance(blueprint_validation.get("errors"), list) else []
                    validation_warnings = blueprint_validation.get("warnings", []) if isinstance(blueprint_validation.get("warnings"), list) else []
                    control_mismatches = blueprint_validation.get("control_mismatches", []) if isinstance(blueprint_validation.get("control_mismatches"), list) else []
                    machine_snapshot = _machine_analysis_snapshot(analysis)
                    manager_state = _sync_candidate_manager_state(selected_run_id, cond, int(cand_idx), row, analysis)
                    widget_keys = manager_state["widget_keys"]
                    score_source = manager_state["score_source"]
                    existing_mm = final_generation_package.get("resolved_multimodal_inputs", {}) if isinstance(final_generation_package.get("resolved_multimodal_inputs"), dict) else {}
                    if existing_mm:
                        if not st.session_state.get(widget_keys["selected_storyboard_panel_ids"]):
                            st.session_state[widget_keys["selected_storyboard_panel_ids"]] = existing_mm.get("selected_storyboard_panel_ids", []) or []
                        if not st.session_state.get(widget_keys["selected_asset_types"]):
                            st.session_state[widget_keys["selected_asset_types"]] = final_generation_package.get("selected_asset_types", []) or st.session_state.get(widget_keys["selected_asset_types"], [])
                        if not st.session_state.get(widget_keys["selected_first_frame_panel_id"]):
                            first_frame_path = str(existing_mm.get("first_frame_image_path", "") or "")
                            for panel in (storyboard_package.get("panels", []) if isinstance(storyboard_package.get("panels"), list) else []):
                                if str(panel.get("image_path", "") or "") == first_frame_path:
                                    st.session_state[widget_keys["selected_first_frame_panel_id"]] = str(panel.get("panel_id", "") or "")
                                    break
                        if not st.session_state.get(widget_keys["selected_last_frame_panel_id"]):
                            last_frame_path = str(existing_mm.get("last_frame_image_path", "") or "")
                            for panel in (storyboard_package.get("panels", []) if isinstance(storyboard_package.get("panels"), list) else []):
                                if str(panel.get("image_path", "") or "") == last_frame_path:
                                    st.session_state[widget_keys["selected_last_frame_panel_id"]] = str(panel.get("panel_id", "") or "")
                                    break
                    render_candidate_summary_card(row)
                    if str(row.get("blueprint_summary", "")).strip():
                        st.caption(f"Blueprint summary: {row.get('blueprint_summary', '')}")
                    if validated_blueprint:
                        beat_count = len((validated_blueprint.get("temporal_plan", {}) or {}).get("beats", []) or [])
                        panel_count = len((storyboard_package.get("panels", []) if isinstance(storyboard_package.get("panels"), list) else []))
                        asset_count = len((asset_pack.get("assets", []) if isinstance(asset_pack.get("assets"), list) else []))
                        st.caption(f"Planning stack: {beat_count} beat(s) · {panel_count} storyboard panel(s) · {asset_count} asset reference slot(s)")
                    if control_mismatches:
                        st.warning("Control compliance issues detected.")
                    if validation_errors and str(row.get("blueprint_status", "")).upper() == "BLUEPRINT_FAILED":
                        st.warning("Blueprint validation failed. Review the validation expander for details.")
                    elif validation_warnings:
                        st.caption("Blueprint validation warnings are available in the validation expander.")

                    st.markdown("### Machine Diagnostic Summary")
                    if machine_snapshot:
                        st.markdown(f"**Keep Recommendation:** {machine_snapshot.get('keep_recommendation', '') or '-'}")
                        st.markdown(f"**Revision Priority:** {machine_snapshot.get('revision_priority', '') or '-'}")
                        st.markdown(f"**Main Issue:** {machine_snapshot.get('main_issue', '') or 'No main issue identified.'}")
                        if machine_snapshot.get("strongest_dimensions"):
                            st.markdown("**Strongest Dimensions**")
                            for item in machine_snapshot.get("strongest_dimensions", []):
                                st.markdown(f"- {item}")
                        if machine_snapshot.get("weakest_dimensions"):
                            st.markdown("**Weakest Dimensions**")
                            for item in machine_snapshot.get("weakest_dimensions", []):
                                st.markdown(f"- {item}")
                    else:
                        st.info("No machine analysis is available yet for this candidate.")

                    action_col1, action_col2 = st.columns(2)
                    if not video_ready_for_analysis:
                        if blueprint_status == "BLUEPRINT_FAILED":
                            action_col1.error("This candidate never reached video generation because blueprint creation failed. Regenerate it first.")
                        else:
                            action_col1.warning("Analyze is only available after a video has been generated successfully.")
                    if action_col1.button(
                        "Analyze Current Candidate",
                        key="analyze_btn",
                        use_container_width=True,
                        disabled=not video_ready_for_analysis,
                    ):
                        with st.spinner("Analyzing..."):
                            study2_agent.analyze_candidate(
                                run_id=selected_run_id,
                                condition=cond,
                                candidate_index=int(cand_idx),
                            )
                        clear_cache()
                        st.success("Analysis completed.")
                        st.rerun()

                    st.markdown("### Working Human Review")
                    if analysis.get("analysis_status") == "ANALYSIS_READY":
                        st.caption("The latest machine analysis has refreshed the editable score baseline below. Adjust these sliders directly before regenerating.")
                    elif score_source == "saved_review" or any(str(row.get(key, "")).strip() != "" for key in RAISE_SCORE_KEYS) or str(row.get("notes", "")).strip():
                        st.caption("Current machine analysis is unavailable; the sliders below may reflect the last saved review values.")
                    else:
                        st.caption("No machine analysis is available yet. The sliders below use fallback defaults until you analyze this candidate.")
                    a, b = st.columns(2)
                    realism = a.slider("Realism", 1, 7, value=int(st.session_state[widget_keys["realism_score_1to7"]]), key=widget_keys["realism_score_1to7"])
                    if analysis.get("analysis_status") == "ANALYSIS_READY":
                        a.caption(f"machine baseline: {manager_state['machine_scores'].get('realism_score_1to7', 4)}")
                    platform_fit = b.slider("Platform Fit", 1, 7, value=int(st.session_state[widget_keys["platform_fit_score_1to7"]]), key=widget_keys["platform_fit_score_1to7"])
                    if analysis.get("analysis_status") == "ANALYSIS_READY":
                        b.caption(f"machine baseline: {manager_state['machine_scores'].get('platform_fit_score_1to7', 4)}")
                    c, d = st.columns(2)
                    integration = c.slider("Integration", 1, 7, value=int(st.session_state[widget_keys["integration_score_1to7"]]), key=widget_keys["integration_score_1to7"])
                    if analysis.get("analysis_status") == "ANALYSIS_READY":
                        c.caption(f"machine baseline: {manager_state['machine_scores'].get('integration_score_1to7', 4)}")
                    commercial_salience = d.slider("Commercial Salience", 1, 7, value=int(st.session_state[widget_keys["commercial_salience_score_1to7"]]), key=widget_keys["commercial_salience_score_1to7"])
                    if analysis.get("analysis_status") == "ANALYSIS_READY":
                        d.caption(f"machine baseline: {manager_state['machine_scores'].get('commercial_salience_score_1to7', 4)}")
                    e, f = st.columns(2)
                    character_consistency = e.slider("Character Consistency", 1, 7, value=int(st.session_state[widget_keys["character_consistency_score_1to7"]]), key=widget_keys["character_consistency_score_1to7"])
                    if analysis.get("analysis_status") == "ANALYSIS_READY":
                        e.caption(f"machine baseline: {manager_state['machine_scores'].get('character_consistency_score_1to7', 4)}")
                    f.empty()
                    g, h = st.columns(2)
                    ai_detection = g.slider("AI Naturalness", 1, 7, value=int(st.session_state[widget_keys[AI_NATURALNESS_KEY]]), key=widget_keys[AI_NATURALNESS_KEY])
                    if analysis.get("analysis_status") == "ANALYSIS_READY":
                        g.caption(f"machine baseline: {manager_state['machine_scores'].get(AI_NATURALNESS_KEY, 4)}")
                    uncanniness = h.slider("Non-Uncanny Naturalness", 1, 7, value=int(st.session_state[widget_keys[NON_UNCANNY_NATURALNESS_KEY]]), key=widget_keys[NON_UNCANNY_NATURALNESS_KEY])
                    if analysis.get("analysis_status") == "ANALYSIS_READY":
                        h.caption(f"machine baseline: {manager_state['machine_scores'].get(NON_UNCANNY_NATURALNESS_KEY, 4)}")
                    keep_options = ["", "yes", "no"]
                    current_keep_value = str(st.session_state.get(widget_keys["working_keep"], "") or "")
                    working_keep = st.selectbox(
                        "Working Keep Intention",
                        options=keep_options,
                        index=keep_options.index(current_keep_value) if current_keep_value in keep_options else 0,
                        key=widget_keys["working_keep"],
                    )
                    working_notes = st.text_area(
                        "Working Notes",
                        value=str(st.session_state.get(widget_keys["working_notes"], "") or ""),
                        height=140,
                        key=widget_keys["working_notes"],
                    )
                    working_saved = st.button("Save Working Review", key=f"save_working_review_{selected_run_id}_{cond}_{cand_idx}")
                    if working_saved:
                        study2_agent.save_screening_decision(
                            run_id=selected_run_id,
                            condition=cond,
                            candidate_index=int(cand_idx),
                            output_video_path=str(row.get("output_video_path", "")),
                            ratings={
                                "realism_score_1to7": realism,
                                "platform_fit_score_1to7": platform_fit,
                                "integration_score_1to7": integration,
                                "commercial_salience_score_1to7": commercial_salience,
                                "character_consistency_score_1to7": character_consistency,
                                AI_NATURALNESS_KEY: ai_detection,
                                NON_UNCANNY_NATURALNESS_KEY: uncanniness,
                            },
                            overall_keep_yes_no=working_keep,
                            notes=working_notes,
                        )
                        clear_cache()
                        st.success("Working review saved for iteration.")
                        st.rerun()

                    st.markdown("### Revision And Regeneration")
                    if analysis.get("analysis_status") == "ANALYSIS_READY":
                        revision_suggestions = analysis.get("analysis_result", {}).get("prompt_revision_suggestions", []) or []
                        if revision_suggestions:
                            st.markdown(f"**Revision Priority:** {machine_snapshot.get('revision_priority', '') or '-'}")
                            st.markdown("**Machine Revision Suggestions**")
                            for item in revision_suggestions:
                                st.markdown(f"- {item}")

                    default_revision_text = _revision_text_default(row, analysis)
                    inferred_route = manager_state.get("inferred_route", "surface_regenerate")
                    current_route_value = str(st.session_state.get(widget_keys["regeneration_route"], inferred_route) or inferred_route)
                    route_help = {
                        "structural_regenerate": "Structural regenerate = fix condition logic, narrative structure, product entry, product centrality, and manipulation validity.",
                        "surface_regenerate": "Surface regenerate = preserve structure but improve realism, motion, text rendering, AI naturalness, and non-uncanny human-like performance.",
                    }
                    st.caption(f"Suggested route: `{inferred_route}` based on the latest analysis. You can override it below.")
                    regeneration_route = st.selectbox(
                        "Regeneration Route",
                        options=REGENERATION_ROUTE_OPTIONS,
                        index=REGENERATION_ROUTE_OPTIONS.index(current_route_value) if current_route_value in REGENERATION_ROUTE_OPTIONS else 1,
                        key=widget_keys["regeneration_route"],
                    )
                    st.caption(route_help.get(regeneration_route, ""))

                    st.markdown("### Multimodal Video Inputs")
                    run_config = load_run_config(selected_run_id)
                    panel_rows = storyboard_package.get("panels", []) if isinstance(storyboard_package.get("panels"), list) else []
                    panel_ids = [str(panel.get("panel_id")) for panel in panel_rows if str(panel.get("panel_id", "")).strip()]
                    default_generation_mode = str(st.session_state.get(widget_keys["video_input_mode"], run_config.get("video_input_mode", "text_only")) or "text_only")
                    generation_mode = st.selectbox(
                        "Video input mode",
                        options=VIDEO_INPUT_MODES,
                        index=VIDEO_INPUT_MODES.index(default_generation_mode) if default_generation_mode in VIDEO_INPUT_MODES else 0,
                        key=widget_keys["video_input_mode"],
                    )
                    asset_type_selection = st.multiselect(
                        "Reference asset types to use",
                        options=MULTIMODAL_ASSET_TYPES,
                        default=[item for item in st.session_state.get(widget_keys["selected_asset_types"], run_config.get("selected_reference_asset_types", [])) if item in MULTIMODAL_ASSET_TYPES],
                        key=widget_keys["selected_asset_types"],
                    )
                    mmc1, mmc2 = st.columns(2)
                    use_panel_first_frame = mmc1.checkbox(
                        "Use storyboard panel as first frame",
                        value=bool(st.session_state.get(widget_keys["use_storyboard_panel_as_first_frame"], run_config.get("use_storyboard_panel_as_first_frame", True))),
                        key=widget_keys["use_storyboard_panel_as_first_frame"],
                    )
                    use_panel_last_frame = mmc2.checkbox(
                        "Use storyboard panel as optional last frame",
                        value=bool(st.session_state.get(widget_keys["use_storyboard_panel_as_last_frame"], run_config.get("use_storyboard_panel_as_last_frame", False))),
                        key=widget_keys["use_storyboard_panel_as_last_frame"],
                    )
                    selected_storyboard_panel_ids = st.multiselect(
                        "Storyboard panels eligible for generation",
                        options=panel_ids,
                        default=[item for item in st.session_state.get(widget_keys["selected_storyboard_panel_ids"], []) if item in panel_ids],
                        key=widget_keys["selected_storyboard_panel_ids"],
                    )
                    first_frame_panel_id = st.selectbox(
                        "First-frame storyboard panel",
                        options=[""] + panel_ids,
                        index=([""] + panel_ids).index(str(st.session_state.get(widget_keys["selected_first_frame_panel_id"], "") or "")) if str(st.session_state.get(widget_keys["selected_first_frame_panel_id"], "") or "") in ([""] + panel_ids) else 0,
                        key=widget_keys["selected_first_frame_panel_id"],
                    )
                    last_frame_panel_id = st.selectbox(
                        "Optional last-frame storyboard panel",
                        options=[""] + panel_ids,
                        index=([""] + panel_ids).index(str(st.session_state.get(widget_keys["selected_last_frame_panel_id"], "") or "")) if str(st.session_state.get(widget_keys["selected_last_frame_panel_id"], "") or "") in ([""] + panel_ids) else 0,
                        key=widget_keys["selected_last_frame_panel_id"],
                    )

                    if panel_rows:
                        with st.expander("Upload / Replace Storyboard Panel Images", expanded=False):
                            upload_cols = st.columns(2)
                            upload_index = 0
                            for panel in panel_rows:
                                panel_id = str(panel.get("panel_id", "") or "")
                                if not panel_id:
                                    continue
                                image_path = str(panel.get("image_path", "") or "")
                                col = upload_cols[upload_index % 2]
                                with col:
                                    st.markdown(f"**{panel_id}**")
                                    if image_path and Path(image_path).exists():
                                        st.image(image_path, use_container_width=True)
                                    uploaded_panel = st.file_uploader(
                                        f"Upload image for {panel_id}",
                                        type=["png", "jpg", "jpeg", "webp"],
                                        key=f"upload_{selected_run_id}_{cond}_{cand_idx}_{panel_id}",
                                    )
                                    save_panel_upload = st.button(
                                        f"Save {panel_id} image",
                                        key=f"save_panel_{selected_run_id}_{cond}_{cand_idx}_{panel_id}",
                                        disabled=uploaded_panel is None,
                                    )
                                    if uploaded_panel is not None and save_panel_upload:
                                        saved_path = save_uploaded_image_to_stem(uploaded_panel, Path(image_path).parent, panel_id)
                                        st.success(f"Saved panel image: {Path(saved_path).name}")
                                        clear_cache()
                                        st.rerun()
                                upload_index += 1

                    multimodal_options = {
                        "generation_mode": generation_mode,
                        "selected_asset_types": asset_type_selection,
                        "selected_storyboard_panel_ids": selected_storyboard_panel_ids,
                        "selected_first_frame_panel_id": first_frame_panel_id,
                        "selected_last_frame_panel_id": last_frame_panel_id,
                        "use_storyboard_panel_as_first_frame": use_panel_first_frame,
                        "use_storyboard_panel_as_last_frame": use_panel_last_frame,
                    }
                    resolved_multimodal_inputs = resolve_multimodal_inputs(
                        run_id=selected_run_id,
                        condition=cond,
                        candidate_index=int(cand_idx),
                        asset_pack=asset_pack,
                        storyboard_package=storyboard_package,
                        options=multimodal_options,
                        config=run_config,
                    )
                    st.caption(
                        f"Resolved mode: `{resolved_multimodal_inputs.get('generation_mode', 'text_only')}` · multimodal ready: `{resolved_multimodal_inputs.get('multimodal_ready', False)}`"
                    )
                    if resolved_multimodal_inputs.get("multimodal_notes"):
                        for item in resolved_multimodal_inputs.get("multimodal_notes", []):
                            st.markdown(f"- {item}")
                    preview_rows = [
                        {"field": "first_frame_image_path", "value": resolved_multimodal_inputs.get("first_frame_image_path", "") or "-"},
                        {"field": "last_frame_image_path", "value": resolved_multimodal_inputs.get("last_frame_image_path", "") or "-"},
                        {"field": "reference_image_paths", "value": ", ".join(resolved_multimodal_inputs.get("reference_image_paths", [])) or "-"},
                        {"field": "selected_asset_types", "value": ", ".join(resolved_multimodal_inputs.get("selected_asset_types", [])) or "-"},
                        {"field": "selected_storyboard_panel_ids", "value": ", ".join(resolved_multimodal_inputs.get("selected_storyboard_panel_ids", [])) or "-"},
                    ]
                    st.dataframe(pd.DataFrame(preview_rows), use_container_width=True, height=210)

                    with st.form(f"regenerate_form_{cond}_{cand_idx}"):
                        revision_instruction_text = st.text_area(
                            "Targeted Revision Instructions (edit the machine suggestions if needed before regenerating)",
                            value=str(st.session_state.get(widget_keys["revision_text"], default_revision_text) or default_revision_text),
                            height=200,
                            key=widget_keys["revision_text"],
                        )
                        submitted = st.form_submit_button("Regenerate Current Candidate (Overwrite and Re-analyze)")
                    if submitted:
                        with st.spinner("Regenerating..."):
                            study2_agent.regenerate_one_candidate(
                                run_id=selected_run_id,
                                condition=cond,
                                candidate_index=int(cand_idx),
                                revision_instruction_text=revision_instruction_text.strip() or None,
                                edited_score_targets=_current_editable_score_targets(widget_keys),
                                regeneration_route=regeneration_route,
                                multimodal_options=multimodal_options,
                                auto_analyze=True,
                            )
                        clear_cache()
                        st.success("The current candidate has been overwritten and re-analyzed.")
                        st.rerun()

                    st.markdown("### Technical Details")
                    with st.expander("View Full Error Details", expanded=False):
                        full_error_text = str(row.get("error_message", "") or "").strip()
                        if full_error_text:
                            st.code(full_error_text)
                        else:
                            st.info("No error details are available.")

                    with st.expander("View Saved Condition Controls", expanded=False):
                        st.json(saved_condition_controls)

                    with st.expander("View Blueprint V2", expanded=False):
                        if validated_blueprint:
                            temporal_beats = (validated_blueprint.get("temporal_plan", {}) or {}).get("beats", []) or []
                            if temporal_beats:
                                st.markdown("**Temporal Beats**")
                                for beat in temporal_beats:
                                    st.markdown(
                                        f"- `{beat.get('beat_id', '')}` {beat.get('approximate_start_seconds', 0)}s-{beat.get('approximate_end_seconds', 0)}s: {beat.get('beat_goal', '')}"
                                    )
                            storyboard_panels = (validated_blueprint.get("storyboard_plan", {}) or {}).get("panels", []) or []
                            if storyboard_panels:
                                st.markdown("**Storyboard Plan**")
                                for panel in storyboard_panels:
                                    st.markdown(
                                        f"- `{panel.get('panel_id', '')}` linked to `{panel.get('linked_beat_id', '')}`: {panel.get('panel_goal', '')}"
                                    )
                            st.json(validated_blueprint)
                        elif isinstance(blueprint_payload.get("blueprint"), dict) and blueprint_payload.get("blueprint"):
                            st.json(blueprint_payload.get("blueprint"))
                        else:
                            st.info("No blueprint data is available for this candidate. Older runs may not have blueprint artifacts.")

                    with st.expander("View Asset Pack", expanded=False):
                        if asset_pack:
                            st.markdown(f"**Asset mode:** {asset_pack.get('generation_mode', '-')}")
                            assets = asset_pack.get("assets", []) if isinstance(asset_pack.get("assets"), list) else []
                            if assets:
                                st.dataframe(pd.DataFrame(assets), use_container_width=True, height=220)
                            if asset_pack.get("notes_on_asset_usage"):
                                st.caption(str(asset_pack.get("notes_on_asset_usage")))
                            st.json(asset_pack)
                        else:
                            st.info("No asset pack artifact is available for this candidate yet.")

                    with st.expander("View Storyboard Panels", expanded=False):
                        panels = storyboard_package.get("panels", []) if isinstance(storyboard_package.get("panels"), list) else []
                        if panels:
                            for panel in panels:
                                st.markdown(
                                    f"**{panel.get('panel_id', '')}** · beat `{panel.get('linked_beat_id', '')}` · {panel.get('panel_summary', '')}"
                                )
                                st.caption(f"Visual focus: {panel.get('visual_focus', '')}")
                                if panel.get("required_elements"):
                                    for item in panel.get("required_elements", []):
                                        st.markdown(f"- {item}")
                                image_path = str(panel.get("image_path", "") or "")
                                if image_path and Path(image_path).exists():
                                    st.image(image_path, use_container_width=True)
                            st.json(storyboard_package)
                        else:
                            st.info("No storyboard panel artifact is available for this candidate yet.")

                    with st.expander("View Execution Script", expanded=False):
                        if execution_script:
                            st.markdown(f"**Overall scene intent:** {execution_script.get('overall_scene_intent', '-')}")
                            st.markdown(f"**Route emphasis:** {execution_script.get('route_emphasis', '-')}")
                            if execution_script.get("beat_by_beat_action_instructions"):
                                st.markdown("**Beat-by-beat instructions**")
                                for item in execution_script.get("beat_by_beat_action_instructions", []):
                                    st.markdown(
                                        f"- `{item.get('beat_id', '')}`: {item.get('action_instruction', '')} | camera: {item.get('camera_note', '')}"
                                    )
                            if execution_script_text:
                                st.text_area(
                                    "Readable Execution Script",
                                    value=execution_script_text,
                                    height=260,
                                    key=f"mgr_execution_script_{cond}_{cand_idx}",
                                )
                            st.json(execution_script)
                        else:
                            st.info("No execution script artifact is available for this candidate yet.")

                    with st.expander("View Final Generation Package", expanded=False):
                        if final_generation_package:
                            st.markdown(f"**Package version:** {final_generation_package.get('package_version', '-')}")
                            st.markdown(f"**Prompt-ready summary:** {final_generation_package.get('prompt_ready_summary', '-')}")
                            st.json(final_generation_package)
                        else:
                            st.info("No final generation package artifact is available for this candidate yet.")

                    with st.expander("View Blueprint Validation", expanded=False):
                        if blueprint_validation:
                            st.json(blueprint_validation)
                            if control_mismatches:
                                st.write({"control_mismatches": control_mismatches})
                        else:
                            st.info("No blueprint validation file is available for this candidate.")

                    with st.expander("View Prompt Details", expanded=False):
                        canonical_prompt = build_condition_prompt(load_run_config(selected_run_id), cond)
                        st.text_area("Locked Canonical Condition Prompt", value=canonical_prompt, height=260, key="mgr_prompt_view")
                        if str(row.get("final_generation_prompt", "")).strip():
                            st.text_area(
                                "Composed Final Generation Prompt",
                                value=str(row.get("final_generation_prompt", "")),
                                height=240,
                                key=f"mgr_final_prompt_{cond}_{cand_idx}",
                            )
                        existing_revision_text = str(row.get("regeneration_instruction_text", "") or "").strip()
                        if existing_revision_text:
                            st.text_area(
                                "Previous Regeneration Instructions Used",
                                value=existing_revision_text,
                                height=120,
                                key="mgr_prev_regen_view",
                            )

                    with st.expander("View Raw Candidate Status", expanded=False):
                        st.json({k: _json_safe_value(v) for k, v in row.to_dict().items()})

with tab_screening:
    st.subheader("3) Screening")
    st.markdown("<div class='section-note'>Machine-generated RAISE suggestions are shown as editable defaults. Human review remains the final judgment.</div>", unsafe_allow_html=True)
    st.info("Use this page as the lighter final confirmation stage after iteration in Candidate Manager. Review likely-good candidates, confirm final scores, and make keep/no decisions.")

    if not selected_run_id:
        st.info("Select or create a run from the left sidebar first.")
    elif run_df.empty:
        st.warning("This run does not contain any candidates.")
    else:
        mode = st.radio("Review Mode", options=["Single Candidate Review", "Cross-Condition Comparison"], horizontal=True)
        if mode == "Single Candidate Review":
            c1, c2 = st.columns(2)
            cond = safe_selectbox("Condition", sorted(run_df["condition"].astype(str).unique().tolist()), key="screen_cond")
            if cond is not None:
                idxs = sorted(run_df.loc[run_df["condition"].astype(str) == cond, "candidate_index"].astype(int).unique().tolist())
                cand_idx = safe_selectbox("Candidate", idxs, key="screen_idx")
                target_rows = [(cond, int(cand_idx))] if cand_idx is not None else []
            else:
                target_rows = []
        else:
            c1, c2 = st.columns(2)
            hi_idxs = sorted(run_df.loc[run_df["condition"].astype(str) == "high_integration", "candidate_index"].astype(int).unique().tolist())
            hs_idxs = sorted(run_df.loc[run_df["condition"].astype(str) == "hard_sell", "candidate_index"].astype(int).unique().tolist())
            hi = c1.selectbox("High Integration Candidate", options=hi_idxs, key="screen_hi") if hi_idxs else None
            hs = c2.selectbox("Hard Sell Candidate", options=hs_idxs, key="screen_hs") if hs_idxs else None
            target_rows = []
            if hi is not None:
                target_rows.append(("high_integration", int(hi)))
            if hs is not None:
                target_rows.append(("hard_sell", int(hs)))

        if not target_rows:
            st.info("No candidates available for the current review mode.")
        else:
            cols = st.columns(len(target_rows))
            for col, (cond, cand_idx) in zip(cols, target_rows):
                row = pick_row(run_df, cond, cand_idx)
                if row is None:
                    continue
                analysis = load_analysis_cached(selected_run_id, cond, int(cand_idx))
                with col:
                    st.markdown(f"### {condition_label(cond)} / cand_{cand_idx:02d}")
                    video_path = str(row.get("output_video_path", ""))
                    show_video_if_exists(video_path)
                    with st.expander("Machine Analysis Reference", expanded=False):
                        render_analysis_panel(analysis, title="Machine Analysis Reference")

                    with st.form(f"screen_form_{cond}_{cand_idx}"):
                        a, b, c = st.columns(3)
                        realism = a.slider("Realism", 1, 7, _score_default(row, analysis, "realism_score_1to7"))
                        platform_fit = b.slider("Platform Fit", 1, 7, _score_default(row, analysis, "platform_fit_score_1to7"))
                        integration = c.slider("Integration", 1, 7, _score_default(row, analysis, "integration_score_1to7"))
                        d, e = st.columns(2)
                        commercial_salience = d.slider("Commercial Salience", 1, 7, _score_default(row, analysis, "commercial_salience_score_1to7"))
                        character_consistency = e.slider("Character Consistency", 1, 7, _score_default(row, analysis, "character_consistency_score_1to7"))
                        g, h = st.columns(2)
                        ai_detection = g.slider("AI Naturalness", 1, 7, _score_default(row, analysis, AI_NATURALNESS_KEY))
                        uncanniness = h.slider("Non-Uncanny Naturalness", 1, 7, _score_default(row, analysis, NON_UNCANNY_NATURALNESS_KEY))
                        keep_options = ["", "yes", "no"]
                        current_keep = _keep_default(row, analysis)
                        keep = st.selectbox("Keep?", options=keep_options, index=keep_options.index(current_keep) if current_keep in keep_options else 0)
                        notes = st.text_area("Notes", value=_notes_default(row, analysis), height=110)
                        submitted = st.form_submit_button("Save Review Scores")
                    if submitted:
                        study2_agent.save_screening_decision(
                            run_id=selected_run_id,
                            condition=cond,
                            candidate_index=cand_idx,
                            output_video_path=video_path,
                            ratings={
                                "realism_score_1to7": realism,
                                "platform_fit_score_1to7": platform_fit,
                                "integration_score_1to7": integration,
                                "commercial_salience_score_1to7": commercial_salience,
                                "character_consistency_score_1to7": character_consistency,
                                AI_NATURALNESS_KEY: ai_detection,
                                NON_UNCANNY_NATURALNESS_KEY: uncanniness,
                            },
                            overall_keep_yes_no=keep,
                            notes=notes,
                        )
                        clear_cache()
                        st.success(f"{condition_label(cond)} / cand_{cand_idx:02d} 已保存。")
                        st.rerun()

with tab_final:
    st.subheader("4) Final Selection")
    st.markdown("<div class='section-note'>Compare the strongest retained candidates and mark the final stimuli for export.</div>", unsafe_allow_html=True)
    st.info("This page displays candidates marked keep=yes and supports final selection review.")
    if not selected_run_id:
        st.info("Select or create a run from the left sidebar first.")
    elif run_df.empty:
        st.warning("This run does not contain any candidates.")
    else:
        if "overall_keep_yes_no" in run_df.columns:
            keep_yes_df = run_df[run_df["overall_keep_yes_no"].fillna("").astype(str).str.lower() == "yes"].copy()
        else:
            keep_yes_df = pd.DataFrame()
        if keep_yes_df.empty:
            st.info("There are no candidates marked keep=yes yet. Complete Screening first.")
        else:
            summary_cols = st.columns(2)
            hi_df = keep_yes_df[keep_yes_df["condition"].astype(str) == "high_integration"].copy()
            hs_df = keep_yes_df[keep_yes_df["condition"].astype(str) == "hard_sell"].copy()
            with summary_cols[0]:
                st.markdown("### High Integration keep=yes")
                st.dataframe(hi_df[[c for c in ["candidate_index", "notes", "realism_score_1to7", "platform_fit_score_1to7", "integration_score_1to7"] if c in hi_df.columns]], use_container_width=True, height=220)
            with summary_cols[1]:
                st.markdown("### Hard Sell keep=yes")
                st.dataframe(hs_df[[c for c in ["candidate_index", "notes", "realism_score_1to7", "platform_fit_score_1to7", "commercial_salience_score_1to7"] if c in hs_df.columns]], use_container_width=True, height=220)

            st.markdown("### Preview keep=yes video")
            c1, c2 = st.columns(2)
            hi_options = sorted(hi_df["candidate_index"].astype(int).unique().tolist()) if not hi_df.empty else []
            hs_options = sorted(hs_df["candidate_index"].astype(int).unique().tolist()) if not hs_df.empty else []
            selected_hi = c1.selectbox("Select High Integration Video", options=hi_options, key="final_hi") if hi_options else None
            selected_hs = c2.selectbox("Select Hard Sell Video", options=hs_options, key="final_hs") if hs_options else None
            preview_cols = st.columns(2)
            if selected_hi is not None:
                row_hi = pick_row(keep_yes_df, "high_integration", int(selected_hi))
                with preview_cols[0]:
                    st.markdown(f"#### High Integration / cand_{int(selected_hi):02d}")
                    if row_hi is not None:
                        show_video_if_exists(str(row_hi.get("output_video_path", "")))
                        render_analysis_panel(load_analysis_cached(selected_run_id, "high_integration", int(selected_hi)), title="Analysis Reference")
                        if st.button("Mark This High Integration Candidate as Final Selected", key="mark_final_hi"):
                            study2_agent.save_final_selection(
                                run_id=selected_run_id,
                                condition="high_integration",
                                candidate_index=int(selected_hi),
                                output_video_path=str(row_hi.get("output_video_path", "")),
                                final_selected_yes_no="yes",
                                experiment_label="HI_final",
                                selection_notes=str(row_hi.get("notes", "")),
                            )
                            clear_cache()
                            st.success("This High Integration candidate has been marked as final selected.")
                            st.rerun()
            if selected_hs is not None:
                row_hs = pick_row(keep_yes_df, "hard_sell", int(selected_hs))
                with preview_cols[1]:
                    st.markdown(f"#### Hard Sell / cand_{int(selected_hs):02d}")
                    if row_hs is not None:
                        show_video_if_exists(str(row_hs.get("output_video_path", "")))
                        render_analysis_panel(load_analysis_cached(selected_run_id, "hard_sell", int(selected_hs)), title="Analysis Reference")
                        if st.button("Mark This Hard Sell Candidate as Final Selected", key="mark_final_hs"):
                            study2_agent.save_final_selection(
                                run_id=selected_run_id,
                                condition="hard_sell",
                                candidate_index=int(selected_hs),
                                output_video_path=str(row_hs.get("output_video_path", "")),
                                final_selected_yes_no="yes",
                                experiment_label="HS_final",
                                selection_notes=str(row_hs.get("notes", "")),
                            )
                            clear_cache()
                            st.success("This Hard Sell candidate has been marked as final selected.")
                            st.rerun()

            if "final_selected_yes_no" in run_df.columns:
                final_yes_df = run_df[run_df["final_selected_yes_no"].fillna("").astype(str).str.lower() == "yes"].copy()
            else:
                final_yes_df = pd.DataFrame()
            st.markdown("### Current final_selected=yes")
            if final_yes_df.empty:
                st.caption("No final selections yet.")
            else:
                st.dataframe(final_yes_df[[c for c in ["condition", "candidate_index", "final_selected_yes_no", "notes", "output_video_path"] if c in final_yes_df.columns]], use_container_width=True, height=180)

            if st.button("Export final_selected_export.csv"):
                export_result = study2_agent.export_final_manifest(run_id=selected_run_id)
                clear_cache()
                st.success(f"Exported: {export_result['export_path']}")
                st.code(str(export_result["export_path"]))

st.markdown("---")
st.caption(f"BASE_OUTPUT_DIR: {BASE_OUTPUT_DIR}")
