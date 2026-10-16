from __future__ import annotations

import csv
import json
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from dotenv import load_dotenv

try:
    from google import genai
    from google.genai import types as genai_types
except Exception:  # pragma: no cover
    genai = None
    genai_types = None


# =========================================================
# Config
# =========================================================

load_dotenv()

DEFAULT_MODEL = "veo-3.1-lite-generate-preview"
DEFAULT_ANALYSIS_MODEL = "gemini-2.5-flash"
DEFAULT_BLUEPRINT_MODEL = DEFAULT_ANALYSIS_MODEL
DEFAULT_BLUEPRINT_MAX_ATTEMPTS = 3
BASE_OUTPUT_DIR = Path(__file__).resolve().parent / "study2_outputs"
BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONDITIONS = ["high_integration", "hard_sell"]
REGENERATION_ROUTES = {"structural_regenerate", "surface_regenerate"}
VIDEO_INPUT_MODES = [
    "text_only",
    "first_frame_to_video",
    "reference_images_to_video",
    "first_frame_plus_reference_images",
]
MULTIMODAL_ASSET_TYPES = [
    "scene_reference",
    "product_reference",
    "character_reference",
    "style_reference",
    "condition_cue_reference",
]
AI_NATURALNESS_KEY = "ai_naturalness_score_1to7"
NON_UNCANNY_NATURALNESS_KEY = "non_uncanny_naturalness_score_1to7"
LEGACY_SCORE_KEY_ALIASES = {
    AI_NATURALNESS_KEY: ["ai_detection_score_1to7"],
    NON_UNCANNY_NATURALNESS_KEY: ["uncanniness_score_1to7"],
}
RATING_COLUMNS = [
    "realism_score_1to7",
    "platform_fit_score_1to7",
    "integration_score_1to7",
    "commercial_salience_score_1to7",
    "character_consistency_score_1to7",
    AI_NATURALNESS_KEY,
    NON_UNCANNY_NATURALNESS_KEY,
    "overall_keep_yes_no",
    "notes",
]
FINAL_SELECTION_COLUMNS = [
    "run_id",
    "condition",
    "candidate_index",
    "output_video_path",
    "final_selected_yes_no",
    "experiment_label",
    "selection_notes",
]

MANIFEST_COLUMNS = [
    "run_id",
    "candidate_index",
    "condition",
    "model",
    "analysis_model",
    "blueprint_model",
    "product_name",
    "product_category",
    "influencer_profile",
    "scene_context",
    "disclosure_text",
    "duration_hint",
    "aspect_ratio",
    "visual_style_lock",
    "camera_style",
    "tone_lock",
    "negative_constraints",
    "base_stimulus_brief",
    "base_prompt",
    "condition_prompt",
    "regeneration_instruction_text",
    "regeneration_route",
    "blueprint_json_path",
    "blueprint_validation_json_path",
    "blueprint_status",
    "blueprint_attempts",
    "blueprint_summary",
    "blueprint_reused_for_surface_regenerate",
    "reused_blueprint_source",
    "condition_controls_json_compact",
    "edited_score_targets_json_compact",
    "video_input_mode",
    "multimodal_generation_enabled",
    "resolved_multimodal_inputs_json_compact",
    "first_frame_image_path_used",
    "last_frame_image_path_used",
    "reference_image_paths_used_json_compact",
    "storyboard_panel_used_for_generation",
    "asset_types_used_for_generation",
    "multimodal_input_summary",
    "control_compliance_ok",
    "control_mismatch_summary",
    "validated_condition_signal_summary",
    "validated_blueprint_json_compact",
    "planning_stack_version",
    "asset_pack_json_path",
    "asset_manifest_csv_path",
    "storyboard_plan_json_path",
    "storyboard_panel_dir",
    "execution_script_json_path",
    "execution_script_text_path",
    "final_generation_package_json_path",
    "final_generation_prompt",
    "output_dir",
    "output_video_path",
    "metadata_json_path",
    "analysis_json_path",
    "status",
    "analysis_status",
    "error_message",
    "created_at_epoch",
    "updated_at_epoch",
]

SCREENING_COLUMNS = [
    "run_id",
    "candidate_index",
    "condition",
    "output_video_path",
    *RATING_COLUMNS,
]

BLUEPRINT_SPEAKING_MODES = {"narrative", "direct_pitch", "mixed", "no_speech"}
BLUEPRINT_PERSUASIVE_INTENSITIES = {"low", "medium", "high"}
BLUEPRINT_PRODUCT_ENTRY_MODES = {"incidental", "embedded_use", "direct_showcase", "pitch_opening"}
BLUEPRINT_PRODUCT_FOCUS_PATTERNS = {"background", "intermittent", "shared_focus", "dominant_focus"}
BLUEPRINT_NARRATIVE_INTERRUPTION_LEVELS = {"none", "mild", "strong"}
BLUEPRINT_DIRECT_ADDRESS_LEVELS = {"no", "partial", "yes"}
BLUEPRINT_CALL_TO_ACTION_LEVELS = {"none", "soft", "explicit"}
BLUEPRINT_PRODUCT_CENTRALITY_LEVELS = {"contextual", "supportive", "central", "dominant"}
BLUEPRINT_REQUIRED_FIELDS = [
    "condition",
    "scene_opening",
    "narrative_beat",
    "product_entry_timing_seconds",
    "product_entry_mode",
    "speaking_mode",
    "camera_behavior",
    "product_focus_pattern",
    "persuasive_intensity",
    "narrative_interruption",
    "direct_address_to_camera",
    "call_to_action_presence",
    "product_centrality",
    "closing_beat",
    "condition_rationale",
    "prompt_ready_summary",
]

STRUCTURAL_FAILURE_MODE_HINTS = {
    "weak_condition_signal",
    "condition_logic_drift",
    "wrong_product_entry_timing",
    "wrong_speaking_mode",
    "wrong_product_centrality",
    "wrong_commercial_salience_pattern",
    "narrative_product_integration_mismatch",
    "manipulation_validity_failure",
}
SURFACE_FAILURE_MODE_HINTS = {
    "lip_sync_mismatch",
    "unnatural_expression_timing",
    "stiff_body_motion",
    "synthetic_face_cues",
    "unrealistic_text_reflection",
    "scripted_delivery",
    "overpolished_ad_tone",
    "robotic_gesture_timing",
    "uncanny_movement",
    "object_continuity_failure",
    "prop_switching_artifact",
    "extra_object_artifact",
    "hand_object_inconsistency",
    "temporal_coherence_failure",
    "duplicate_object_artifact",
}
STABLE_SKELETON_FIELDS = [
    "product_name",
    "product_category",
    "influencer_profile",
    "scene_context",
    "duration_hint",
    "aspect_ratio",
    "visual_style_lock",
    "camera_style",
    "tone_lock",
    "negative_constraints",
]
ASSET_MANIFEST_COLUMNS = [
    "asset_type",
    "scope",
    "required",
    "status",
    "expected_path",
    "notes",
]


# =========================================================
# IO helpers
# =========================================================


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def write_csv_rows(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def safe_read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def get_score_with_backward_compat(payload: Dict[str, Any], key: str, default: Any = None) -> Any:
    if not isinstance(payload, dict):
        return default
    if key in payload and str(payload.get(key, "")).strip() != "":
        return payload.get(key, default)
    for alias in LEGACY_SCORE_KEY_ALIASES.get(key, []):
        if alias in payload and str(payload.get(alias, "")).strip() != "":
            return payload.get(alias, default)
    return default


def remap_score_keys(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    out = dict(payload)
    for canonical_key, aliases in LEGACY_SCORE_KEY_ALIASES.items():
        if str(out.get(canonical_key, "")).strip():
            continue
        for alias in aliases:
            alias_value = out.get(alias, "")
            if str(alias_value).strip() != "":
                out[canonical_key] = alias_value
                break
    return out


# =========================================================
# Generic helpers
# =========================================================


def build_run_id(prefix: str = "study2") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def numeric_score(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or str(value).strip() == "":
            return default
        return float(value)
    except Exception:
        return default


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def _clean_keep(value: Any) -> str:
    text = _clean_text(value).lower()
    if text in {"yes", "keep", "true", "1"}:
        return "yes"
    if text in {"no", "reject", "false", "0"}:
        return "no"
    return ""


def _clamp_score(value: Any, default: int = 4) -> int:
    try:
        score = int(round(float(value)))
    except Exception:
        return default
    return max(1, min(7, score))


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    if not text:
        return []
    return [text]


def _compact_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        return _clean_text(value)


def _extract_first_number(text: Any, default: float = 8.0) -> float:
    match = re.search(r"(\d+(?:\.\d+)?)", _clean_text(text))
    if not match:
        return default
    try:
        return float(match.group(1))
    except Exception:
        return default


def _clamp_float(value: Any, min_value: float, max_value: float, default: float) -> float:
    try:
        parsed = float(value)
    except Exception:
        parsed = default
    return max(min_value, min(max_value, parsed))


def _normalize_enum(value: Any, allowed: set[str], default: str) -> str:
    text = _clean_text(value).lower().replace("-", "_").replace(" ", "_")
    return text if text in allowed else default


def default_high_integration_controls() -> Dict[str, Any]:
    return {
        "product_entry_timing_seconds": 3.5,
        "product_entry_mode": "embedded_use",
        "speaking_mode": "narrative",
        "product_focus_pattern": "intermittent",
        "persuasive_intensity": "low",
        "narrative_interruption": "none",
        "direct_address_to_camera": "no",
        "call_to_action_presence": "none",
        "product_centrality": "supportive",
    }


def default_hard_sell_controls() -> Dict[str, Any]:
    return {
        "product_entry_timing_seconds": 0.8,
        "product_entry_mode": "pitch_opening",
        "speaking_mode": "direct_pitch",
        "product_focus_pattern": "dominant_focus",
        "persuasive_intensity": "high",
        "narrative_interruption": "strong",
        "direct_address_to_camera": "yes",
        "call_to_action_presence": "explicit",
        "product_centrality": "dominant",
    }


def normalize_condition_controls(
    controls: Optional[Dict[str, Any]],
    condition: Literal["high_integration", "hard_sell"],
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    base = default_high_integration_controls() if condition == "high_integration" else default_hard_sell_controls()
    merged = dict(base)
    if isinstance(controls, dict):
        merged.update(controls)
    duration_seconds = max(4.0, _extract_first_number((config or {}).get("duration_hint"), 8.0))
    return {
        "product_entry_timing_seconds": round(
            _clamp_float(merged.get("product_entry_timing_seconds"), 0.0, duration_seconds, float(base["product_entry_timing_seconds"])),
            2,
        ),
        "product_entry_mode": _normalize_enum(merged.get("product_entry_mode"), BLUEPRINT_PRODUCT_ENTRY_MODES, base["product_entry_mode"]),
        "speaking_mode": _normalize_enum(merged.get("speaking_mode"), BLUEPRINT_SPEAKING_MODES, base["speaking_mode"]),
        "product_focus_pattern": _normalize_enum(
            merged.get("product_focus_pattern"),
            BLUEPRINT_PRODUCT_FOCUS_PATTERNS,
            base["product_focus_pattern"],
        ),
        "persuasive_intensity": _normalize_enum(
            merged.get("persuasive_intensity"),
            BLUEPRINT_PERSUASIVE_INTENSITIES,
            base["persuasive_intensity"],
        ),
        "narrative_interruption": _normalize_enum(
            merged.get("narrative_interruption"),
            BLUEPRINT_NARRATIVE_INTERRUPTION_LEVELS,
            base["narrative_interruption"],
        ),
        "direct_address_to_camera": _normalize_enum(
            merged.get("direct_address_to_camera"),
            BLUEPRINT_DIRECT_ADDRESS_LEVELS,
            base["direct_address_to_camera"],
        ),
        "call_to_action_presence": _normalize_enum(
            merged.get("call_to_action_presence"),
            BLUEPRINT_CALL_TO_ACTION_LEVELS,
            base["call_to_action_presence"],
        ),
        "product_centrality": _normalize_enum(
            merged.get("product_centrality"),
            BLUEPRINT_PRODUCT_CENTRALITY_LEVELS,
            base["product_centrality"],
        ),
    }


def validate_condition_controls(
    controls: Optional[Dict[str, Any]],
    condition: Literal["high_integration", "hard_sell"],
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    normalized = normalize_condition_controls(controls, condition, config)
    errors: List[str] = []

    if condition == "high_integration":
        if normalized["speaking_mode"] == "direct_pitch":
            errors.append("high_integration_controls.speaking_mode should not be direct_pitch.")
        if normalized["persuasive_intensity"] == "high":
            errors.append("high_integration_controls.persuasive_intensity should not be high.")
    else:
        if normalized["persuasive_intensity"] != "high":
            errors.append("hard_sell_controls.persuasive_intensity should be high.")
        if normalized["direct_address_to_camera"] == "no":
            errors.append("hard_sell_controls.direct_address_to_camera should not be 'no'.")

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "normalized_controls": normalized,
    }


def get_condition_controls(
    config: Dict[str, Any],
    condition: Literal["high_integration", "hard_sell"],
) -> Dict[str, Any]:
    key = "high_integration_controls" if condition == "high_integration" else "hard_sell_controls"
    return normalize_condition_controls(config.get(key), condition, config)


def _contains_any(text: str, keywords: List[str]) -> bool:
    lower_text = _clean_text(text).lower()
    return any(keyword in lower_text for keyword in keywords)


def _normalize_short_list(value: Any, max_items: int) -> List[str]:
    items = _as_list(value)
    return [str(item).strip() for item in items[:max_items] if str(item).strip()]


def _clean_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    text = _clean_text(value).lower()
    if text in {"true", "yes", "1"}:
        return True
    if text in {"false", "no", "0"}:
        return False
    return default


def _existing_reference_path(directory: Path, stem: str) -> str:
    for suffix in [".png", ".jpg", ".jpeg", ".webp"]:
        candidate = directory / f"{stem}{suffix}"
        if candidate.exists():
            return str(candidate)
    return str(directory / f"{stem}.png")


def _build_default_temporal_beats(
    condition: Literal["high_integration", "hard_sell"],
    controls: Dict[str, Any],
    duration_seconds: float,
) -> List[Dict[str, Any]]:
    entry_seconds = _clamp_float(controls.get("product_entry_timing_seconds"), 0.0, duration_seconds, duration_seconds / 2.0)
    beat_1_end = round(min(max(entry_seconds, 1.6), max(2.0, duration_seconds * 0.35)), 2)
    beat_2_end = round(min(max(beat_1_end + 1.6, duration_seconds * 0.7), max(duration_seconds - 1.2, beat_1_end + 1.6)), 2)
    return [
        {
            "beat_id": "beat_01",
            "beat_label": "setup",
            "approximate_start_seconds": 0.0,
            "approximate_end_seconds": beat_1_end,
            "beat_goal": "Establish the creator, scene, and opening context before the main persuasive move fully develops.",
            "product_visibility_role": "background" if condition == "high_integration" else "emerging_focus",
            "speaking_behavior": "narrative_setup" if controls.get("speaking_mode") != "no_speech" else "silent_visual_setup",
            "camera_intent": "creator-native handheld setup framing",
            "condition_signal_role": "soft_setup" if condition == "high_integration" else "fast_pitch_setup",
            "notes": "Lock identity, scene logic, and initial continuity cues.",
        },
        {
            "beat_id": "beat_02",
            "beat_label": "product_entry",
            "approximate_start_seconds": beat_1_end,
            "approximate_end_seconds": beat_2_end,
            "beat_goal": "Introduce the product in the condition-appropriate way while preserving scene plausibility.",
            "product_visibility_role": controls.get("product_focus_pattern") or "shared_focus",
            "speaking_behavior": controls.get("speaking_mode") or "mixed",
            "camera_intent": "follow product interaction without losing creator continuity",
            "condition_signal_role": "core_manipulation_signal",
            "notes": "Product entry timing and focus pattern must align with condition controls.",
        },
        {
            "beat_id": "beat_03",
            "beat_label": "reinforcement",
            "approximate_start_seconds": beat_2_end,
            "approximate_end_seconds": round(duration_seconds, 2),
            "beat_goal": "Reinforce the condition logic and land the ending beat without breaking continuity.",
            "product_visibility_role": controls.get("product_centrality") or "supportive",
            "speaking_behavior": "direct_recommendation" if condition == "hard_sell" else "casual_followthrough",
            "camera_intent": "stable payoff framing with natural performance timing",
            "condition_signal_role": "closing_condition_confirmation",
            "notes": "Preserve realism, continuity, and platform-native performance until the end.",
        },
    ]


def _normalize_temporal_beats(
    beats_raw: Any,
    condition: Literal["high_integration", "hard_sell"],
    controls: Dict[str, Any],
    duration_seconds: float,
) -> List[Dict[str, Any]]:
    source = beats_raw if isinstance(beats_raw, list) else []
    default_beats = _build_default_temporal_beats(condition, controls, duration_seconds)
    if not source:
        source = default_beats

    beats: List[Dict[str, Any]] = []
    previous_end = 0.0
    for index, item in enumerate(source[:4], start=1):
        raw = item if isinstance(item, dict) else {}
        fallback = default_beats[min(index - 1, len(default_beats) - 1)]
        start = round(_clamp_float(raw.get("approximate_start_seconds"), 0.0, duration_seconds, previous_end), 2)
        end_default = fallback.get("approximate_end_seconds", duration_seconds)
        end = round(_clamp_float(raw.get("approximate_end_seconds"), start, duration_seconds, end_default), 2)
        if end <= start:
            end = round(min(duration_seconds, start + max(0.8, duration_seconds / 6.0)), 2)
        beats.append(
            {
                "beat_id": _clean_text(raw.get("beat_id")) or f"beat_{index:02d}",
                "beat_label": _clean_text(raw.get("beat_label")) or fallback.get("beat_label", f"beat_{index:02d}"),
                "approximate_start_seconds": start,
                "approximate_end_seconds": end,
                "beat_goal": _clean_text(raw.get("beat_goal")) or fallback.get("beat_goal", ""),
                "product_visibility_role": _clean_text(raw.get("product_visibility_role")) or fallback.get("product_visibility_role", ""),
                "speaking_behavior": _clean_text(raw.get("speaking_behavior")) or fallback.get("speaking_behavior", ""),
                "camera_intent": _clean_text(raw.get("camera_intent")) or fallback.get("camera_intent", ""),
                "condition_signal_role": _clean_text(raw.get("condition_signal_role")) or fallback.get("condition_signal_role", ""),
                "notes": _clean_text(raw.get("notes")) or fallback.get("notes", ""),
            }
        )
        previous_end = end
    return beats[:4]


def _build_default_asset_plan(condition: Literal["high_integration", "hard_sell"]) -> Dict[str, Any]:
    return {
        "scene_reference_required": True,
        "product_reference_required": True,
        "character_reference_required": True,
        "style_reference_required": True,
        "optional_condition_cue_reference_required": condition == "hard_sell",
        "notes_on_asset_usage": (
            "Use shared scene, product, character, and style references across both conditions. "
            "Condition-specific cue references are optional and should only reinforce manipulation signaling without changing world identity."
        ),
    }


def _normalize_asset_plan(asset_plan_raw: Any, condition: Literal["high_integration", "hard_sell"]) -> Dict[str, Any]:
    raw = asset_plan_raw if isinstance(asset_plan_raw, dict) else {}
    defaults = _build_default_asset_plan(condition)
    return {
        "scene_reference_required": _clean_bool(raw.get("scene_reference_required"), defaults["scene_reference_required"]),
        "product_reference_required": _clean_bool(raw.get("product_reference_required"), defaults["product_reference_required"]),
        "character_reference_required": _clean_bool(raw.get("character_reference_required"), defaults["character_reference_required"]),
        "style_reference_required": _clean_bool(raw.get("style_reference_required"), defaults["style_reference_required"]),
        "optional_condition_cue_reference_required": _clean_bool(
            raw.get("optional_condition_cue_reference_required"),
            defaults["optional_condition_cue_reference_required"],
        ),
        "notes_on_asset_usage": _clean_text(raw.get("notes_on_asset_usage")) or defaults["notes_on_asset_usage"],
    }


def _build_default_storyboard_panels(
    beats: List[Dict[str, Any]],
    condition: Literal["high_integration", "hard_sell"],
) -> List[Dict[str, Any]]:
    panels: List[Dict[str, Any]] = []
    for index, beat in enumerate(beats[:4], start=1):
        panels.append(
            {
                "panel_id": f"panel_{index:02d}",
                "linked_beat_id": beat.get("beat_id", f"beat_{index:02d}"),
                "panel_goal": beat.get("beat_goal", ""),
                "visual_focus": beat.get("product_visibility_role", ""),
                "required_elements": [
                    "same influencer identity",
                    "same product identity",
                    "same scene world",
                    beat.get("camera_intent", ""),
                ],
                "should_directly_support_condition_signal": bool(index >= 2 or condition == "hard_sell"),
            }
        )
    return panels


def _normalize_storyboard_plan(
    storyboard_raw: Any,
    beats: List[Dict[str, Any]],
    condition: Literal["high_integration", "hard_sell"],
) -> Dict[str, Any]:
    raw = storyboard_raw if isinstance(storyboard_raw, dict) else {}
    source_panels = raw.get("panels") if isinstance(raw.get("panels"), list) else raw if isinstance(raw, list) else []
    default_panels = _build_default_storyboard_panels(beats, condition)
    panels: List[Dict[str, Any]] = []
    for index, item in enumerate((source_panels or default_panels)[:4], start=1):
        raw_panel = item if isinstance(item, dict) else {}
        fallback = default_panels[min(index - 1, len(default_panels) - 1)]
        required_elements = _normalize_short_list(raw_panel.get("required_elements"), 6) or fallback["required_elements"]
        panels.append(
            {
                "panel_id": _clean_text(raw_panel.get("panel_id")) or fallback["panel_id"],
                "linked_beat_id": _clean_text(raw_panel.get("linked_beat_id")) or fallback["linked_beat_id"],
                "panel_goal": _clean_text(raw_panel.get("panel_goal")) or fallback["panel_goal"],
                "visual_focus": _clean_text(raw_panel.get("visual_focus")) or fallback["visual_focus"],
                "required_elements": required_elements,
                "should_directly_support_condition_signal": _clean_bool(
                    raw_panel.get("should_directly_support_condition_signal"),
                    fallback["should_directly_support_condition_signal"],
                ),
            }
        )
    return {
        "panels": panels,
        "storyboard_generation_mode": _clean_text(raw.get("storyboard_generation_mode")) or "scaffold_ready",
    }


def _build_default_execution_plan(
    beats: List[Dict[str, Any]],
    condition: Literal["high_integration", "hard_sell"],
    controls: Dict[str, Any],
) -> Dict[str, Any]:
    direct_address = controls.get("direct_address_to_camera", "no")
    cta_presence = controls.get("call_to_action_presence", "none")
    return {
        "overall_scene_intent": (
            "Preserve a TikTok-native lived moment where the product appears organically."
            if condition == "high_integration"
            else "Deliver a more explicit creator-led pitch while keeping the same scene identity and creator world."
        ),
        "beat_by_beat_action_instructions": [
            {
                "beat_id": beat.get("beat_id", ""),
                "action_instruction": beat.get("beat_goal", ""),
                "camera_note": beat.get("camera_intent", ""),
            }
            for beat in beats
        ],
        "dialogue_outline": [beat.get("speaking_behavior", "") for beat in beats if beat.get("speaking_behavior")],
        "action_outline": [beat.get("beat_goal", "") for beat in beats if beat.get("beat_goal")],
        "product_interaction_outline": [
            f"{beat.get('beat_id', '')}: {beat.get('product_visibility_role', '')}"
            for beat in beats
            if beat.get("product_visibility_role")
        ],
        "direct_address_timing": (
            "Direct address should be absent or minimal." if direct_address == "no" else f"Direct address level should be {direct_address}."
        ),
        "cta_behavior": (
            "No explicit CTA should appear." if cta_presence == "none" else f"CTA presence should remain {cta_presence}."
        ),
        "camera_behavior_notes": [beat.get("camera_intent", "") for beat in beats if beat.get("camera_intent")],
        "realism_priorities": [
            "keep creator-native performance timing",
            "preserve stable identity and scene continuity",
            "avoid glossy commercial polish",
        ],
        "continuity_priorities": [
            "preserve product identity across all beats",
            "preserve hand-object consistency",
            "preserve temporal coherence and prop stability",
        ],
        "forbidden_drifts": [
            "do not change influencer identity",
            "do not change product appearance",
            "do not shift into a different setting or ad world",
        ],
    }


def _normalize_execution_plan(
    execution_raw: Any,
    beats: List[Dict[str, Any]],
    condition: Literal["high_integration", "hard_sell"],
    controls: Dict[str, Any],
) -> Dict[str, Any]:
    raw = execution_raw if isinstance(execution_raw, dict) else {}
    defaults = _build_default_execution_plan(beats, condition, controls)
    beat_instructions_raw = raw.get("beat_by_beat_action_instructions") if isinstance(raw.get("beat_by_beat_action_instructions"), list) else []
    beat_instructions = []
    for index, beat in enumerate(beats, start=1):
        raw_item = beat_instructions_raw[index - 1] if index - 1 < len(beat_instructions_raw) and isinstance(beat_instructions_raw[index - 1], dict) else {}
        default_item = defaults["beat_by_beat_action_instructions"][index - 1]
        beat_instructions.append(
            {
                "beat_id": _clean_text(raw_item.get("beat_id")) or default_item["beat_id"],
                "action_instruction": _clean_text(raw_item.get("action_instruction")) or default_item["action_instruction"],
                "camera_note": _clean_text(raw_item.get("camera_note")) or default_item["camera_note"],
            }
        )
    return {
        "overall_scene_intent": _clean_text(raw.get("overall_scene_intent")) or defaults["overall_scene_intent"],
        "beat_by_beat_action_instructions": beat_instructions,
        "dialogue_outline": _normalize_short_list(raw.get("dialogue_outline"), 8) or defaults["dialogue_outline"],
        "action_outline": _normalize_short_list(raw.get("action_outline"), 8) or defaults["action_outline"],
        "product_interaction_outline": _normalize_short_list(raw.get("product_interaction_outline"), 8) or defaults["product_interaction_outline"],
        "direct_address_timing": _clean_text(raw.get("direct_address_timing")) or defaults["direct_address_timing"],
        "cta_behavior": _clean_text(raw.get("cta_behavior")) or defaults["cta_behavior"],
        "camera_behavior_notes": _normalize_short_list(raw.get("camera_behavior_notes"), 8) or defaults["camera_behavior_notes"],
        "realism_priorities": _normalize_short_list(raw.get("realism_priorities"), 8) or defaults["realism_priorities"],
        "continuity_priorities": _normalize_short_list(raw.get("continuity_priorities"), 8) or defaults["continuity_priorities"],
        "forbidden_drifts": _normalize_short_list(raw.get("forbidden_drifts"), 8) or defaults["forbidden_drifts"],
    }


# =========================================================
# Path helpers
# =========================================================


def get_run_dir(run_id: str) -> Path:
    path = BASE_OUTPUT_DIR / run_id
    ensure_dir(path)
    return path


def get_manifest_path(run_id: str) -> Path:
    return get_run_dir(run_id) / "manifest.csv"


def get_screening_path(run_id: str) -> Path:
    return get_run_dir(run_id) / "screening.csv"


def get_final_selection_path(run_id: str) -> Path:
    return get_run_dir(run_id) / "final_selection.csv"


def get_run_config_path(run_id: str) -> Path:
    return get_run_dir(run_id) / "run_config.json"


def get_candidate_dir(run_id: str, condition: str, candidate_index: int) -> Path:
    path = get_run_dir(run_id) / condition / f"cand_{int(candidate_index):02d}"
    ensure_dir(path)
    return path


def get_metadata_path(run_id: str, condition: str, candidate_index: int) -> Path:
    return get_candidate_dir(run_id, condition, candidate_index) / "metadata.json"


def get_analysis_path(run_id: str, condition: str, candidate_index: int) -> Path:
    return get_candidate_dir(run_id, condition, candidate_index) / "analysis.json"


def get_blueprint_path(run_id: str, condition: str, candidate_index: int) -> Path:
    return get_candidate_dir(run_id, condition, candidate_index) / "blueprint.json"


def get_blueprint_validation_path(run_id: str, condition: str, candidate_index: int) -> Path:
    return get_candidate_dir(run_id, condition, candidate_index) / "blueprint_validation.json"


def get_run_asset_dir(run_id: str) -> Path:
    return get_run_dir(run_id) / "shared_assets"


def get_candidate_asset_dir(run_id: str, condition: str, candidate_index: int) -> Path:
    return get_candidate_dir(run_id, condition, candidate_index) / "assets"


def get_asset_pack_path(run_id: str, condition: str, candidate_index: int) -> Path:
    return get_candidate_dir(run_id, condition, candidate_index) / "asset_pack.json"


def get_asset_manifest_path(run_id: str, condition: str, candidate_index: int) -> Path:
    return get_candidate_dir(run_id, condition, candidate_index) / "asset_manifest.csv"


def get_storyboard_plan_path(run_id: str, condition: str, candidate_index: int) -> Path:
    return get_candidate_dir(run_id, condition, candidate_index) / "storyboard_plan.json"


def get_storyboard_panel_dir(run_id: str, condition: str, candidate_index: int) -> Path:
    return get_candidate_dir(run_id, condition, candidate_index) / "storyboard_panels"


def get_execution_script_path(run_id: str, condition: str, candidate_index: int) -> Path:
    return get_candidate_dir(run_id, condition, candidate_index) / "execution_script.json"


def get_execution_script_text_path(run_id: str, condition: str, candidate_index: int) -> Path:
    return get_candidate_dir(run_id, condition, candidate_index) / "execution_script.txt"


def get_final_generation_package_path(run_id: str, condition: str, candidate_index: int) -> Path:
    return get_candidate_dir(run_id, condition, candidate_index) / "final_generation_package.json"


def normalize_multimodal_generation_options(
    options: Optional[Dict[str, Any]],
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = config or {}
    raw = options if isinstance(options, dict) else {}
    generation_mode = _clean_text(raw.get("generation_mode") or raw.get("video_input_mode") or cfg.get("video_input_mode") or "text_only")
    if generation_mode not in VIDEO_INPUT_MODES:
        generation_mode = "text_only"
    selected_asset_types = [item for item in _normalize_short_list(raw.get("selected_asset_types") or cfg.get("selected_reference_asset_types"), 8) if item in MULTIMODAL_ASSET_TYPES]
    if not selected_asset_types:
        selected_asset_types = list(cfg.get("selected_reference_asset_types") or [])
        selected_asset_types = [item for item in selected_asset_types if item in MULTIMODAL_ASSET_TYPES] or ["scene_reference", "product_reference", "character_reference", "style_reference"]
    return {
        "generation_mode": generation_mode,
        "use_storyboard_panel_as_first_frame": _clean_bool(
            raw.get("use_storyboard_panel_as_first_frame"),
            _clean_bool(cfg.get("use_storyboard_panel_as_first_frame"), True),
        ),
        "use_storyboard_panel_as_last_frame": _clean_bool(
            raw.get("use_storyboard_panel_as_last_frame"),
            _clean_bool(cfg.get("use_storyboard_panel_as_last_frame"), False),
        ),
        "selected_first_frame_panel_id": _clean_text(raw.get("selected_first_frame_panel_id")),
        "selected_last_frame_panel_id": _clean_text(raw.get("selected_last_frame_panel_id")),
        "selected_storyboard_panel_ids": _normalize_short_list(raw.get("selected_storyboard_panel_ids"), 8),
        "selected_asset_types": selected_asset_types,
    }


# =========================================================
# Run config and storage
# =========================================================


def default_run_config() -> Dict[str, Any]:
    return {
        "run_id": "",
        "model": DEFAULT_MODEL,
        "analysis_model": DEFAULT_ANALYSIS_MODEL,
        "blueprint_model": DEFAULT_BLUEPRINT_MODEL,
        "blueprint_max_attempts": DEFAULT_BLUEPRINT_MAX_ATTEMPTS,
        "auto_analyze_after_generation": False,
        "product_name": "",
        "product_category": "",
        "influencer_profile": "",
        "scene_context": "",
        "disclosure_text": "",
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


def initialize_run_storage(run_id: str) -> None:
    ensure_dir(get_run_dir(run_id))
    if not get_manifest_path(run_id).exists():
        write_csv_rows(get_manifest_path(run_id), [], MANIFEST_COLUMNS)
    if not get_screening_path(run_id).exists():
        write_csv_rows(get_screening_path(run_id), [], SCREENING_COLUMNS)
    if not get_final_selection_path(run_id).exists():
        write_csv_rows(get_final_selection_path(run_id), [], FINAL_SELECTION_COLUMNS)


def save_run_config(run_config: Dict[str, Any]) -> Dict[str, Any]:
    run_id = str(run_config.get("run_id") or build_run_id())
    full = default_run_config()
    full.update(run_config)
    full["run_id"] = run_id
    initialize_run_storage(run_id)
    write_json(get_run_config_path(run_id), full)
    return full


def load_run_config(run_id: str) -> Dict[str, Any]:
    config = default_run_config()
    config.update(safe_read_json(get_run_config_path(run_id)))
    config["run_id"] = run_id
    return config


# =========================================================
# Prompt builders
# =========================================================


def build_base_stimulus_brief(config: Dict[str, Any]) -> str:
    return (
        f"Create a realistic TikTok-style {config['duration_hint']}\n"
        f"Aspect ratio: {config['aspect_ratio']}.\n"
        f"A fluent English-speaking Asian influencer.\n"
        f"Influencer profile: {config['influencer_profile']}.\n"
        f"Product: {config['product_name']}, a fictional {config['product_category']}.\n"
        f"Scene context: {config['scene_context']}.\n"
        f"Visual style lock: {config['visual_style_lock']}.\n"
        f"Camera style lock: {config['camera_style']}.\n"
        f"Tone lock: {config['tone_lock']}.\n"
        f"Negative constraints: {config['negative_constraints']}.\n\n"
        "This is experimental stimulus material, so performer identity, overall scene logic, production quality, lighting, framing, and platform feel must remain stable across conditions."
    )


def build_condition_delta(condition: Literal["high_integration", "hard_sell"]) -> str:
    if condition == "high_integration":
        return (
            "Condition: High integration / narrative.\n"
            "The product must be naturally embedded into the unfolding story.\n"
            "The influencer should continue the narrative flow while using or mentioning the product.\n"
            "The product should function like part of the lived moment rather than the sole focal point.\n"
            "The commercial message should feel seamless, organic, and story-led.\n"
            "Avoid abrupt direct selling language.\n"
            "Avoid pausing the narrative for an overt pitch."
        )
    return (
        "Condition: Low integration / hard sell.\n"
        "The influencer should interrupt the flow and directly address the camera.\n"
        "The product should become the visual and verbal focal point.\n"
        "Use clearly promotional framing and explicit recommendation language.\n"
        "The commercial intent should be more salient and more obviously persuasive.\n"
        "The product explanation should feel more like a deliberate ad segment than a naturally embedded story element."
    )


def build_condition_prompt(config: Dict[str, Any], condition: Literal["high_integration", "hard_sell"]) -> str:
    base_prompt = build_base_stimulus_brief(config)
    delta = build_condition_delta(condition)
    return f"{base_prompt}\n\n{delta}"


def compose_regeneration_prompt(
    config: Dict[str, Any],
    condition: Literal["high_integration", "hard_sell"],
    revision_instruction_text: Optional[str] = None,
) -> str:
    core_prompt = build_condition_prompt(config, condition)
    cleaned = _clean_text(revision_instruction_text)
    if not cleaned:
        return core_prompt

    return (
        f"{core_prompt}\n\n"
        "Targeted revision instructions for this regeneration only:\n"
        f"{cleaned}\n\n"
        "Important: preserve the same product, product category, influencer identity, scene context, platform-native style, aspect ratio, and condition logic defined above. "
        "Apply the targeted revisions without changing the core stimulus concept."
    )


def build_shared_stimulus_skeleton(config: Dict[str, Any]) -> Dict[str, Any]:
    skeleton = {
        "product_name": _clean_text(config.get("product_name")),
        "product_category": _clean_text(config.get("product_category")),
        "influencer_profile": _clean_text(config.get("influencer_profile")),
        "scene_context": _clean_text(config.get("scene_context")),
        "duration_hint": _clean_text(config.get("duration_hint")),
        "aspect_ratio": _clean_text(config.get("aspect_ratio")),
        "visual_style_lock": _clean_text(config.get("visual_style_lock")),
        "camera_style": _clean_text(config.get("camera_style")),
        "tone_lock": _clean_text(config.get("tone_lock")),
        "negative_constraints": _clean_text(config.get("negative_constraints")),
    }
    skeleton["stable_character_core"] = f"Influencer profile lock: {skeleton['influencer_profile']}"
    skeleton["stable_scene_core"] = f"Scene lock: {skeleton['scene_context']}"
    skeleton["stable_visual_core"] = (
        f"Style lock: {skeleton['visual_style_lock']}; camera lock: {skeleton['camera_style']}; tone lock: {skeleton['tone_lock']}"
    )
    return skeleton


def infer_regeneration_route(analysis_result: Dict[str, Any]) -> str:
    if not isinstance(analysis_result, dict):
        return "surface_regenerate"

    failure_modes = {
        _clean_text(item).lower().replace(" ", "_")
        for item in _as_list(analysis_result.get("failure_modes"))
        if _clean_text(item)
    }
    main_issue = _clean_text(analysis_result.get("main_issue")).lower()
    revision_priority = _clean_text(analysis_result.get("revision_priority")).lower()
    summary_blob = " ".join([main_issue, revision_priority, _clean_text(analysis_result.get("summary"))]).lower()

    structural_hits = len(failure_modes & STRUCTURAL_FAILURE_MODE_HINTS)
    surface_hits = len(failure_modes & SURFACE_FAILURE_MODE_HINTS)

    structural_keywords = [
        "condition",
        "integration",
        "hard sell",
        "high integration",
        "product entry",
        "speaking mode",
        "product centrality",
        "commercial salience",
        "narrative structure",
        "manipulation",
        "structural",
    ]
    surface_keywords = [
        "realism",
        "uncanny",
        "ai detect",
        "lip sync",
        "motion",
        "gesture",
        "rendering",
        "face",
        "synthetic",
        "surface",
    ]

    structural_hits += sum(1 for keyword in structural_keywords if keyword in summary_blob)
    surface_hits += sum(1 for keyword in surface_keywords if keyword in summary_blob)

    if structural_hits > surface_hits:
        return "structural_regenerate"
    return "surface_regenerate"


def build_regeneration_route_guidance(
    regeneration_route: Optional[str],
    condition: Literal["high_integration", "hard_sell"],
) -> str:
    route = str(regeneration_route or "surface_regenerate").strip().lower()
    if route == "structural_regenerate":
        return "\n".join(
            [
                "Regeneration route: structural_regenerate",
                "- Treat this as a structure-repair pass.",
                "- Repair condition validity, product-entry timing, speaking mode, product centrality, commercial salience pattern, and narrative/product integration.",
                "- Preserve the shared skeleton, but allow blueprint-level restructuring when needed to restore the intended manipulation.",
                f"- Make the {condition} condition signal clearer and more defensible.",
            ]
        ).strip()
    return "\n".join(
        [
            "Regeneration route: surface_regenerate",
            "- Treat this as a surface-quality repair pass.",
            "- Preserve the current condition logic and structure as much as possible.",
            "- Focus on realism, facial behavior, motion timing, object continuity, prop consistency, AI naturalness, and non-uncanny human-like performance.",
            "- Do not heavily rewrite product-entry or narrative structure unless absolutely required for realism.",
        ]
    ).strip()


def build_score_target_guidance(
    edited_score_targets: Optional[Dict[str, Any]],
    condition: Literal["high_integration", "hard_sell"],
) -> str:
    if not isinstance(edited_score_targets, dict) or not edited_score_targets:
        return ""

    def score(name: str, default: int = 4) -> int:
        try:
            return max(1, min(7, int(float(edited_score_targets.get(name, default)))))
        except Exception:
            return default

    realism = score("realism_score_1to7")
    platform_fit = score("platform_fit_score_1to7")
    integration = score("integration_score_1to7")
    commercial_salience = score("commercial_salience_score_1to7")
    character_consistency = score("character_consistency_score_1to7")
    ai_naturalness = score(AI_NATURALNESS_KEY)
    non_uncanny_naturalness = score(NON_UNCANNY_NATURALNESS_KEY)

    condition_note = (
        f"Keep integration near {integration}/7 while avoiding over-dominant sales cues."
        if condition == "high_integration"
        else f"Keep commercial salience near {commercial_salience}/7 while preserving explicit hard-sell signaling."
    )

    return "\n".join(
        [
            "Current regeneration quality targets (human-edited score targets):",
            f"- Realism target: {realism}/7",
            f"- Platform fit target: {platform_fit}/7",
            f"- Character consistency target: {character_consistency}/7",
            f"- AI naturalness target: {ai_naturalness}/7",
            f"- Non-uncanny naturalness target: {non_uncanny_naturalness}/7",
            f"- {condition_note}",
        ]
    ).strip()


def build_blueprint_prompt(
    config: Dict[str, Any],
    condition: Literal["high_integration", "hard_sell"],
    revision_instruction_text: Optional[str] = None,
    edited_score_targets: Optional[Dict[str, Any]] = None,
    regeneration_route: Optional[str] = None,
) -> str:
    skeleton = build_shared_stimulus_skeleton(config)
    controls = get_condition_controls(config, condition)
    condition_guidance = build_condition_delta(condition)
    score_target_guidance = build_score_target_guidance(edited_score_targets, condition)
    route_guidance = build_regeneration_route_guidance(regeneration_route, condition)
    revision_text = _clean_text(revision_instruction_text)
    revision_block = (
        f"\nTargeted regeneration notes for this candidate only:\n{revision_text}\n"
        if revision_text
        else ""
    )
    score_target_block = f"\nQuality targets for this regeneration:\n{score_target_guidance}\n" if score_target_guidance else ""
    return f"""
You are designing a controlled experimental stimulus blueprint for a short-form social video ad.

Your job:
1. Preserve the shared stimulus skeleton exactly.
2. Obey the explicit condition controls exactly.
3. Build a Blueprint V2 design specification with temporal planning, asset planning, storyboard checkpoints, and execution planning.
4. Only fill in natural execution details around those controls.
5. Return STRICT JSON only. No markdown. No commentary.

Target condition: {condition}

Shared stimulus skeleton:
{json.dumps(skeleton, ensure_ascii=False, indent=2)}

Explicit condition controls that must be preserved exactly:
{json.dumps(controls, ensure_ascii=False, indent=2)}

Condition guidance:
{condition_guidance}
Regeneration route guidance:
{route_guidance}
{score_target_block}
{revision_block}
Rules:
- Keep the same product, influencer identity, scene logic, visual style, camera style, tone, aspect ratio, and negative constraints.
- Do not redefine the condition control values. Copy them exactly into the output JSON.
- Only change how the blueprint operationalizes those controls in a coherent scene.
- Use 2 to 4 temporal beats only.
- Storyboard panels are key visual checkpoints, not separate independently generated video shots.
- The execution plan should read like a performance plan, not a generic prose summary.
- Keep the stimulus TikTok-native and realistic.
- The prompt_ready_summary must be concise but execution-ready.
- Output one JSON object with the exact schema below.

Required schema:
{{
  "blueprint_version": "v2",
  "condition": "{condition}",
  "scene_opening": "",
  "narrative_beat": "",
  "product_entry_timing_seconds": 0.0,
  "product_entry_mode": "incidental | embedded_use | direct_showcase | pitch_opening",
  "speaking_mode": "narrative | direct_pitch | mixed | no_speech",
  "camera_behavior": "",
  "product_focus_pattern": "background | intermittent | shared_focus | dominant_focus",
  "persuasive_intensity": "low | medium | high",
  "narrative_interruption": "none | mild | strong",
  "direct_address_to_camera": "no | partial | yes",
  "call_to_action_presence": "none | soft | explicit",
  "product_centrality": "contextual | supportive | central | dominant",
  "closing_beat": "",
  "condition_rationale": "",
  "prompt_ready_summary": "",
  "shared_stable_elements": {{
    "product_name": "",
    "product_category": "",
    "influencer_profile": "",
    "scene_context": "",
    "duration_hint": "",
    "aspect_ratio": "",
    "visual_style_lock": "",
    "camera_style": "",
    "tone_lock": "",
    "negative_constraints": ""
  }},
  "condition_control_layer": {{
    "condition": "{condition}",
    "product_entry_timing_seconds": 0.0,
    "product_entry_mode": "incidental | embedded_use | direct_showcase | pitch_opening",
    "speaking_mode": "narrative | direct_pitch | mixed | no_speech",
    "product_focus_pattern": "background | intermittent | shared_focus | dominant_focus",
    "persuasive_intensity": "low | medium | high",
    "narrative_interruption": "none | mild | strong",
    "direct_address_to_camera": "no | partial | yes",
    "call_to_action_presence": "none | soft | explicit",
    "product_centrality": "contextual | supportive | central | dominant"
  }},
  "temporal_plan": {{
    "beats": [
      {{
        "beat_id": "beat_01",
        "beat_label": "",
        "approximate_start_seconds": 0.0,
        "approximate_end_seconds": 0.0,
        "beat_goal": "",
        "product_visibility_role": "",
        "speaking_behavior": "",
        "camera_intent": "",
        "condition_signal_role": "",
        "notes": ""
      }}
    ]
  }},
  "asset_plan": {{
    "scene_reference_required": true,
    "product_reference_required": true,
    "character_reference_required": true,
    "style_reference_required": true,
    "optional_condition_cue_reference_required": false,
    "notes_on_asset_usage": ""
  }},
  "storyboard_plan": {{
    "panels": [
      {{
        "panel_id": "panel_01",
        "linked_beat_id": "beat_01",
        "panel_goal": "",
        "visual_focus": "",
        "required_elements": ["", ""],
        "should_directly_support_condition_signal": true
      }}
    ]
  }},
  "execution_plan": {{
    "overall_scene_intent": "",
    "beat_by_beat_action_instructions": [
      {{
        "beat_id": "beat_01",
        "action_instruction": "",
        "camera_note": ""
      }}
    ],
    "dialogue_outline": ["", ""],
    "action_outline": ["", ""],
    "product_interaction_outline": ["", ""],
    "direct_address_timing": "",
    "cta_behavior": "",
    "camera_behavior_notes": ["", ""],
    "realism_priorities": ["", ""],
    "continuity_priorities": ["", ""],
    "forbidden_drifts": ["", ""]
  }}
}}
""".strip()


def build_candidate_asset_pack(
    run_id: str,
    condition: Literal["high_integration", "hard_sell"],
    candidate_index: int,
    blueprint: Dict[str, Any],
    regeneration_route: Optional[str] = None,
) -> Dict[str, Any]:
    shared_asset_dir = get_run_asset_dir(run_id)
    candidate_asset_dir = get_candidate_asset_dir(run_id, condition, candidate_index)
    ensure_dir(shared_asset_dir)
    ensure_dir(candidate_asset_dir)

    asset_plan = blueprint.get("asset_plan", {}) if isinstance(blueprint.get("asset_plan"), dict) else {}
    scope_assets = [
        ("scene_reference", "shared", _clean_bool(asset_plan.get("scene_reference_required"), True)),
        ("product_reference", "shared", _clean_bool(asset_plan.get("product_reference_required"), True)),
        ("character_reference", "shared", _clean_bool(asset_plan.get("character_reference_required"), True)),
        ("style_reference", "shared", _clean_bool(asset_plan.get("style_reference_required"), True)),
        (
            "condition_cue_reference",
            "condition_specific",
            _clean_bool(asset_plan.get("optional_condition_cue_reference_required"), condition == "hard_sell"),
        ),
    ]

    assets: List[Dict[str, Any]] = []
    for asset_type, scope, required in scope_assets:
        directory = shared_asset_dir if scope == "shared" else candidate_asset_dir
        stem = asset_type if scope == "shared" else f"{asset_type}_{condition}"
        expected_path = _existing_reference_path(directory, stem)
        exists = Path(expected_path).exists()
        status = "provided" if exists else ("placeholder_required" if required else "optional_missing")
        assets.append(
            {
                "asset_type": asset_type,
                "scope": scope,
                "required": "yes" if required else "no",
                "status": status,
                "expected_path": expected_path,
                "uploaded_file_path": expected_path if exists else "",
                "manually_supplied_path": expected_path if exists else "",
                "asset_source_type": "provided_file" if exists else "missing",
                "in_use_for_generation": "no",
                "notes": (
                    "Manual asset injection is supported by placing a matching file at the expected path."
                    if not exists
                    else "Existing reference asset will be reused."
                ),
            }
        )

    return {
        "run_id": run_id,
        "condition": condition,
        "candidate_index": candidate_index,
        "regeneration_route": str(regeneration_route or "").strip().lower(),
        "generation_mode": "shared_asset_scaffold_with_manual_injection_hooks",
        "shared_asset_dir": str(shared_asset_dir),
        "candidate_asset_dir": str(candidate_asset_dir),
        "asset_plan": asset_plan,
        "assets": assets,
        "notes_on_asset_usage": _clean_text(asset_plan.get("notes_on_asset_usage")),
    }


def build_storyboard_panel_package(
    run_id: str,
    condition: Literal["high_integration", "hard_sell"],
    candidate_index: int,
    blueprint: Dict[str, Any],
    regeneration_route: Optional[str] = None,
) -> Dict[str, Any]:
    panel_dir = get_storyboard_panel_dir(run_id, condition, candidate_index)
    ensure_dir(panel_dir)
    storyboard_plan = blueprint.get("storyboard_plan", {}) if isinstance(blueprint.get("storyboard_plan"), dict) else {}
    panels = storyboard_plan.get("panels", []) if isinstance(storyboard_plan.get("panels"), list) else []
    temporal_beats = blueprint.get("temporal_plan", {}).get("beats", []) if isinstance(blueprint.get("temporal_plan"), dict) else []
    beat_map = {str(beat.get("beat_id")): beat for beat in temporal_beats if isinstance(beat, dict)}
    route_name = str(regeneration_route or "surface_regenerate").strip().lower()

    enriched_panels: List[Dict[str, Any]] = []
    for panel in panels:
        beat = beat_map.get(str(panel.get("linked_beat_id")), {})
        panel_id = _clean_text(panel.get("panel_id")) or f"panel_{len(enriched_panels)+1:02d}"
        image_path = panel_dir / f"{panel_id}.png"
        enriched_panels.append(
            {
                "panel_id": panel_id,
                "linked_beat_id": _clean_text(panel.get("linked_beat_id")),
                "image_path": str(image_path),
                "panel_prompt": (
                    f"Key visual checkpoint for {condition}. "
                    f"Show {panel.get('panel_goal', '')} with visual focus on {panel.get('visual_focus', '')}. "
                    f"Preserve influencer identity, product identity, and scene continuity. "
                    f"Beat context: {beat.get('beat_goal', '')}. Route emphasis: {route_name}."
                ).strip(),
                "panel_summary": _clean_text(panel.get("panel_goal")),
                "visual_focus": _clean_text(panel.get("visual_focus")),
                "condition_signal_strength_hint": _clean_text(beat.get("condition_signal_role")) or "support_condition_logic",
                "continuity_constraints": [
                    "same creator identity",
                    "same product identity",
                    "same scene world",
                    "no object switching across adjacent beats",
                ],
                "required_elements": _normalize_short_list(panel.get("required_elements"), 6),
                "should_directly_support_condition_signal": _clean_bool(panel.get("should_directly_support_condition_signal"), False),
                "generation_status": "image_exists" if image_path.exists() else "scaffold_only",
                "manually_supplied_path": str(image_path) if image_path.exists() else "",
                "panel_source_type": "provided_file" if image_path.exists() else "missing",
                "in_use_for_generation": "no",
            }
        )

    return {
        "run_id": run_id,
        "condition": condition,
        "candidate_index": candidate_index,
        "storyboard_generation_mode": "scaffold_with_optional_manual_panel_injection",
        "panel_dir": str(panel_dir),
        "panels": enriched_panels,
    }


def build_execution_script(
    run_id: str,
    condition: Literal["high_integration", "hard_sell"],
    candidate_index: int,
    blueprint: Dict[str, Any],
    regeneration_route: Optional[str] = None,
) -> Dict[str, Any]:
    execution_plan = blueprint.get("execution_plan", {}) if isinstance(blueprint.get("execution_plan"), dict) else {}
    route_name = str(regeneration_route or "surface_regenerate").strip().lower()
    route_emphasis = (
        "Prioritize condition-valid restructuring if needed."
        if route_name == "structural_regenerate"
        else "Preserve structure and prioritize surface realism, continuity, and performance polish."
    )
    return {
        "run_id": run_id,
        "condition": condition,
        "candidate_index": candidate_index,
        "execution_script_version": "v1",
        "regeneration_route": route_name,
        "overall_scene_intent": execution_plan.get("overall_scene_intent", ""),
        "beat_by_beat_action_instructions": execution_plan.get("beat_by_beat_action_instructions", []),
        "dialogue_outline": execution_plan.get("dialogue_outline", []),
        "action_outline": execution_plan.get("action_outline", []),
        "product_interaction_outline": execution_plan.get("product_interaction_outline", []),
        "direct_address_timing": execution_plan.get("direct_address_timing", ""),
        "cta_behavior": execution_plan.get("cta_behavior", ""),
        "camera_behavior_notes": execution_plan.get("camera_behavior_notes", []),
        "realism_priorities": execution_plan.get("realism_priorities", []),
        "continuity_priorities": execution_plan.get("continuity_priorities", []),
        "forbidden_drifts": execution_plan.get("forbidden_drifts", []),
        "route_emphasis": route_emphasis,
    }


def build_execution_script_text(execution_script: Dict[str, Any]) -> str:
    beat_lines = [
        f"- {item.get('beat_id', '')}: {item.get('action_instruction', '')} | camera: {item.get('camera_note', '')}"
        for item in execution_script.get("beat_by_beat_action_instructions", [])
        if isinstance(item, dict)
    ]
    sections = [
        f"Overall scene intent: {execution_script.get('overall_scene_intent', '')}",
        f"Route emphasis: {execution_script.get('route_emphasis', '')}",
        "Beat-by-beat action instructions:",
        "\n".join(beat_lines) or "-",
        "Dialogue outline:",
        "\n".join(f"- {item}" for item in execution_script.get("dialogue_outline", [])) or "-",
        "Action outline:",
        "\n".join(f"- {item}" for item in execution_script.get("action_outline", [])) or "-",
        "Product interaction outline:",
        "\n".join(f"- {item}" for item in execution_script.get("product_interaction_outline", [])) or "-",
        f"Direct address timing: {execution_script.get('direct_address_timing', '')}",
        f"CTA behavior: {execution_script.get('cta_behavior', '')}",
        "Camera behavior notes:",
        "\n".join(f"- {item}" for item in execution_script.get("camera_behavior_notes", [])) or "-",
        "Realism priorities:",
        "\n".join(f"- {item}" for item in execution_script.get("realism_priorities", [])) or "-",
        "Continuity priorities:",
        "\n".join(f"- {item}" for item in execution_script.get("continuity_priorities", [])) or "-",
        "Forbidden drifts:",
        "\n".join(f"- {item}" for item in execution_script.get("forbidden_drifts", [])) or "-",
    ]
    return "\n\n".join(sections).strip()


def build_final_generation_package(
    config: Dict[str, Any],
    blueprint: Dict[str, Any],
    asset_pack: Dict[str, Any],
    storyboard_package: Dict[str, Any],
    execution_script: Dict[str, Any],
    revision_instruction_text: Optional[str] = None,
    edited_score_targets: Optional[Dict[str, Any]] = None,
    regeneration_route: Optional[str] = None,
    multimodal_options: Optional[Dict[str, Any]] = None,
    resolved_multimodal_inputs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    condition = "hard_sell" if str(blueprint.get("condition", "")).strip() == "hard_sell" else "high_integration"
    resolved_multimodal_inputs = resolved_multimodal_inputs if isinstance(resolved_multimodal_inputs, dict) else {}
    normalized_options = normalize_multimodal_generation_options(multimodal_options, config)
    panel_map = {
        str(panel.get("panel_id")): panel
        for panel in storyboard_package.get("panels", [])
        if isinstance(panel, dict)
    } if isinstance(storyboard_package.get("panels"), list) else {}
    selected_first_frame_panel_path = ""
    selected_last_frame_panel_path = ""
    if normalized_options.get("selected_first_frame_panel_id") in panel_map:
        selected_first_frame_panel_path = str(panel_map[normalized_options["selected_first_frame_panel_id"]].get("image_path", "") or "")
    if normalized_options.get("selected_last_frame_panel_id") in panel_map:
        selected_last_frame_panel_path = str(panel_map[normalized_options["selected_last_frame_panel_id"]].get("image_path", "") or "")
    return {
        "package_version": "v1",
        "run_id": _clean_text(config.get("run_id")),
        "condition": condition,
        "candidate_index": _coerce_int(blueprint.get("candidate_index"), 0),
        "model": _clean_text(config.get("model")),
        "analysis_model": _clean_text(config.get("analysis_model")),
        "blueprint_v2": blueprint,
        "shared_stable_elements": blueprint.get("shared_stable_elements", build_shared_stimulus_skeleton(config)),
        "condition_control_layer": blueprint.get("condition_control_layer", {}),
        "asset_pack": asset_pack,
        "storyboard_package": storyboard_package,
        "execution_script": execution_script,
        "edited_score_targets": edited_score_targets or {},
        "revision_instruction_text": _clean_text(revision_instruction_text),
        "regeneration_route": str(regeneration_route or "").strip().lower(),
        "video_input_mode": normalized_options.get("generation_mode", "text_only"),
        "use_storyboard_panel_as_first_frame": normalized_options.get("use_storyboard_panel_as_first_frame", True),
        "selected_first_frame_panel_path": selected_first_frame_panel_path or _clean_text(resolved_multimodal_inputs.get("first_frame_image_path")),
        "selected_last_frame_panel_path": selected_last_frame_panel_path or _clean_text(resolved_multimodal_inputs.get("last_frame_image_path")),
        "selected_reference_image_paths": resolved_multimodal_inputs.get("reference_image_paths", []),
        "selected_asset_types": resolved_multimodal_inputs.get("selected_asset_types", []),
        "resolved_multimodal_inputs": resolved_multimodal_inputs,
        "multimodal_generation_enabled": bool(resolved_multimodal_inputs.get("multimodal_generation_enabled", False)),
        "multimodal_input_summary": (
            f"mode={normalized_options.get('generation_mode', 'text_only')} | "
            f"first_frame={_clean_text(resolved_multimodal_inputs.get('first_frame_image_path')) or 'none'} | "
            f"references={len(resolved_multimodal_inputs.get('reference_image_paths', []) or [])} | "
            f"storyboard_panels={','.join(resolved_multimodal_inputs.get('selected_storyboard_panel_ids', []) or []) or 'none'} | "
            f"asset_types={','.join(resolved_multimodal_inputs.get('selected_asset_types', []) or []) or 'none'}"
        ),
        "prompt_ready_summary": _clean_text(blueprint.get("prompt_ready_summary")),
    }


def resolve_multimodal_inputs(
    run_id: str,
    condition: Literal["high_integration", "hard_sell"],
    candidate_index: int,
    *,
    asset_pack: Optional[Dict[str, Any]] = None,
    storyboard_package: Optional[Dict[str, Any]] = None,
    options: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = config or load_run_config(run_id)
    asset_pack = asset_pack if isinstance(asset_pack, dict) else safe_read_json(get_asset_pack_path(run_id, condition, candidate_index))
    storyboard_package = storyboard_package if isinstance(storyboard_package, dict) else safe_read_json(get_storyboard_plan_path(run_id, condition, candidate_index))
    normalized_options = normalize_multimodal_generation_options(options, cfg)
    generation_mode = normalized_options["generation_mode"]
    selected_asset_types = list(normalized_options["selected_asset_types"])
    selected_panel_ids = list(normalized_options["selected_storyboard_panel_ids"])
    selected_first_frame_panel_id = normalized_options["selected_first_frame_panel_id"]
    selected_last_frame_panel_id = normalized_options["selected_last_frame_panel_id"]
    use_storyboard_panel_as_first_frame = normalized_options["use_storyboard_panel_as_first_frame"]
    use_storyboard_panel_as_last_frame = normalized_options["use_storyboard_panel_as_last_frame"]

    asset_rows = asset_pack.get("assets", []) if isinstance(asset_pack.get("assets"), list) else []
    panel_rows = storyboard_package.get("panels", []) if isinstance(storyboard_package.get("panels"), list) else []
    panel_map = {str(panel.get("panel_id")): panel for panel in panel_rows if isinstance(panel, dict)}
    notes: List[str] = []

    def existing_path(path_text: Any) -> str:
        text = str(path_text or "").strip()
        return text if text and Path(text).exists() else ""

    available_assets = {str(asset.get("asset_type")): asset for asset in asset_rows if isinstance(asset, dict) and existing_path(asset.get("expected_path"))}
    available_panels = {panel_id: panel for panel_id, panel in panel_map.items() if existing_path(panel.get("image_path"))}

    selected_reference_assets: List[tuple[str, str]] = []
    for asset_type in selected_asset_types:
        asset = available_assets.get(asset_type)
        if not asset:
            continue
        path = existing_path(asset.get("expected_path"))
        if not path:
            continue
        selected_reference_assets.append((asset_type, path))

    selected_storyboard_panel_ids: List[str] = []
    for panel_id in selected_panel_ids:
        if panel_id in available_panels:
            selected_storyboard_panel_ids.append(panel_id)
            panel_path = existing_path(available_panels[panel_id].get("image_path"))
            if panel_path and ("storyboard_panel", panel_path) not in selected_reference_assets:
                selected_reference_assets.append((f"storyboard_panel:{panel_id}", panel_path))

    first_frame_panel_path = ""
    if use_storyboard_panel_as_first_frame:
        candidate_panel_id = selected_first_frame_panel_id or (selected_storyboard_panel_ids[0] if selected_storyboard_panel_ids else "")
        if candidate_panel_id and candidate_panel_id in available_panels:
            first_frame_panel_path = existing_path(available_panels[candidate_panel_id].get("image_path"))
            if candidate_panel_id not in selected_storyboard_panel_ids:
                selected_storyboard_panel_ids.insert(0, candidate_panel_id)

    if not first_frame_panel_path and use_storyboard_panel_as_first_frame and available_panels:
        first_panel = next(iter(available_panels.values()))
        first_frame_panel_path = existing_path(first_panel.get("image_path"))
        if first_frame_panel_path:
            selected_storyboard_panel_ids.insert(0, _clean_text(first_panel.get("panel_id")))
            notes.append("No explicit first-frame panel was selected, so the first available storyboard panel was used.")

    first_frame_image_path = first_frame_panel_path
    if not first_frame_image_path and generation_mode in {"first_frame_to_video", "first_frame_plus_reference_images"}:
        fallback_asset = available_assets.get("scene_reference") or available_assets.get("character_reference") or available_assets.get("product_reference")
        if fallback_asset:
            first_frame_image_path = existing_path(fallback_asset.get("expected_path"))
            if first_frame_image_path:
                notes.append("No storyboard first frame was available, so a shared reference asset was used as the first frame.")

    last_frame_image_path = ""
    if use_storyboard_panel_as_last_frame:
        candidate_last_panel_id = selected_last_frame_panel_id or (selected_storyboard_panel_ids[-1] if selected_storyboard_panel_ids else "")
        if candidate_last_panel_id and candidate_last_panel_id in available_panels:
            last_frame_image_path = existing_path(available_panels[candidate_last_panel_id].get("image_path"))
            if candidate_last_panel_id not in selected_storyboard_panel_ids:
                selected_storyboard_panel_ids.append(candidate_last_panel_id)

    multimodal_ready = False
    if generation_mode == "text_only":
        multimodal_ready = True
    elif generation_mode == "first_frame_to_video":
        multimodal_ready = bool(first_frame_image_path)
    elif generation_mode == "reference_images_to_video":
        multimodal_ready = bool(selected_reference_assets)
    elif generation_mode == "first_frame_plus_reference_images":
        multimodal_ready = bool(first_frame_image_path and selected_reference_assets)

    effective_generation_mode_hint = generation_mode
    if generation_mode == "first_frame_plus_reference_images" and first_frame_image_path and selected_reference_assets:
        effective_generation_mode_hint = "first_frame_to_video"
        notes.append(
            "The current Veo SDK does not support image/last_frame together with reference_images in a single request, so runtime will fall back to first_frame_to_video while keeping reference cues in the prompt text."
        )

    if generation_mode != "text_only" and not multimodal_ready:
        notes.append("Requested multimodal mode does not currently have enough valid image inputs, so generation should fall back to text_only.")

    used_asset_types = [source for source, _ in selected_reference_assets[:4] if not str(source).startswith("storyboard_panel:")]

    return {
        "generation_mode": generation_mode,
        "effective_generation_mode_hint": effective_generation_mode_hint,
        "first_frame_image_path": first_frame_image_path,
        "last_frame_image_path": last_frame_image_path,
        "reference_image_paths": [path for _, path in selected_reference_assets[:4]],
        "reference_image_sources": [source for source, _ in selected_reference_assets[:4]],
        "selected_storyboard_panel_ids": selected_storyboard_panel_ids,
        "selected_asset_types": used_asset_types,
        "use_storyboard_panel_as_first_frame": use_storyboard_panel_as_first_frame,
        "use_storyboard_panel_as_last_frame": use_storyboard_panel_as_last_frame,
        "multimodal_ready": multimodal_ready,
        "multimodal_generation_enabled": generation_mode != "text_only" and multimodal_ready,
        "multimodal_notes": notes,
    }


def compose_video_prompt_from_blueprint(
    config: Dict[str, Any],
    blueprint: Dict[str, Any],
    revision_instruction_text: Optional[str] = None,
    edited_score_targets: Optional[Dict[str, Any]] = None,
    regeneration_route: Optional[str] = None,
    asset_pack: Optional[Dict[str, Any]] = None,
    storyboard_package: Optional[Dict[str, Any]] = None,
    execution_script: Optional[Dict[str, Any]] = None,
    final_generation_package: Optional[Dict[str, Any]] = None,
) -> str:
    skeleton = blueprint.get("shared_stable_elements") if isinstance(blueprint.get("shared_stable_elements"), dict) else build_shared_stimulus_skeleton(config)
    blueprint_condition = "hard_sell" if str(blueprint.get("condition", "")).strip() == "hard_sell" else "high_integration"
    controls = get_condition_controls(config, blueprint_condition)
    revision_text = _clean_text(revision_instruction_text)
    score_target_guidance = build_score_target_guidance(edited_score_targets, blueprint_condition)
    route_guidance = build_regeneration_route_guidance(regeneration_route, blueprint_condition)
    asset_pack = asset_pack if isinstance(asset_pack, dict) else {}
    storyboard_package = storyboard_package if isinstance(storyboard_package, dict) else {}
    execution_script = execution_script if isinstance(execution_script, dict) else {}
    final_generation_package = final_generation_package if isinstance(final_generation_package, dict) else {}
    resolved_multimodal_inputs = final_generation_package.get("resolved_multimodal_inputs", {}) if isinstance(final_generation_package.get("resolved_multimodal_inputs"), dict) else {}
    revision_block = (
        "\nCritical repair constraints for this regeneration only. These corrections are non-optional and must be applied while preserving the same skeleton and condition logic:\n"
        f"{revision_text}\n"
        if revision_text
        else ""
    )
    score_target_block = (
        f"\nQuality targets for this regeneration. Use these as explicit improvement targets while preserving condition validity:\n{score_target_guidance}\n"
        if score_target_guidance
        else ""
    )
    asset_lines = []
    for asset in asset_pack.get("assets", []):
        if not isinstance(asset, dict):
            continue
        asset_lines.append(
            f"- {asset.get('asset_type', '')}: status={asset.get('status', '')}, path={asset.get('expected_path', '')}, notes={asset.get('notes', '')}"
        )
    storyboard_lines = []
    for panel in storyboard_package.get("panels", [])[:4]:
        if not isinstance(panel, dict):
            continue
        storyboard_lines.append(
            f"- {panel.get('panel_id', '')} linked to {panel.get('linked_beat_id', '')}: {panel.get('panel_summary', '')} | visual focus: {panel.get('visual_focus', '')} | continuity: {', '.join(panel.get('continuity_constraints', []))}"
        )
    beat_lines = []
    for beat in blueprint.get("temporal_plan", {}).get("beats", [])[:4]:
        if not isinstance(beat, dict):
            continue
        beat_lines.append(
            f"- {beat.get('beat_id', '')} ({beat.get('approximate_start_seconds', 0.0)}s-{beat.get('approximate_end_seconds', 0.0)}s): {beat.get('beat_goal', '')} | speaking: {beat.get('speaking_behavior', '')} | condition role: {beat.get('condition_signal_role', '')}"
        )
    return f"""
Create a realistic TikTok-style short-form vertical video stimulus from the following final generation package.

Stable shared elements that must not drift:
- Product: {skeleton.get("product_name", "")} ({skeleton.get("product_category", "")})
- Influencer profile: {skeleton.get("influencer_profile", "")}
- Scene context: {skeleton.get("scene_context", "")}
- Duration hint: {skeleton.get("duration_hint", "")}
- Aspect ratio: {skeleton.get("aspect_ratio", "")}
- Visual style lock: {skeleton.get("visual_style_lock", "")}
- Camera style lock: {skeleton.get("camera_style", "")}
- Tone lock: {skeleton.get("tone_lock", "")}
- Negative constraints: {skeleton.get("negative_constraints", "")}

Explicit condition controls that must remain valid:
- Product entry timing seconds: {controls.get("product_entry_timing_seconds", "")}
- Product entry mode: {controls.get("product_entry_mode", "")}
- Speaking mode: {controls.get("speaking_mode", "")}
- Product focus pattern: {controls.get("product_focus_pattern", "")}
- Persuasive intensity: {controls.get("persuasive_intensity", "")}
- Narrative interruption: {controls.get("narrative_interruption", "")}
- Direct address to camera: {controls.get("direct_address_to_camera", "")}
- Call to action presence: {controls.get("call_to_action_presence", "")}
- Product centrality: {controls.get("product_centrality", "")}

Blueprint V2 temporal plan:
{chr(10).join(beat_lines) if beat_lines else "- No explicit beats were provided."}

Storyboard panel checkpoints:
{chr(10).join(storyboard_lines) if storyboard_lines else "- No storyboard panel scaffolds were provided."}

Asset pack expectations:
{chr(10).join(asset_lines) if asset_lines else "- No asset scaffolds were provided."}

Multimodal video input mode:
- Mode: {final_generation_package.get("video_input_mode", "text_only")}
- First-frame image: {resolved_multimodal_inputs.get("first_frame_image_path", "") or "none"}
- Last-frame image: {resolved_multimodal_inputs.get("last_frame_image_path", "") or "none"}
- Reference images: {", ".join(resolved_multimodal_inputs.get("reference_image_paths", [])) or "none"}
- Selected asset types: {", ".join(resolved_multimodal_inputs.get("selected_asset_types", [])) or "none"}
- Selected storyboard panels: {", ".join(resolved_multimodal_inputs.get("selected_storyboard_panel_ids", [])) or "none"}

Execution script:
- Overall scene intent: {execution_script.get("overall_scene_intent", "")}
- Route emphasis: {execution_script.get("route_emphasis", "")}
- Direct address timing: {execution_script.get("direct_address_timing", "")}
- CTA behavior: {execution_script.get("cta_behavior", "")}
- Camera behavior notes: {", ".join(execution_script.get("camera_behavior_notes", []))}
- Realism priorities: {", ".join(execution_script.get("realism_priorities", []))}
- Continuity priorities: {", ".join(execution_script.get("continuity_priorities", []))}
- Forbidden drifts: {", ".join(execution_script.get("forbidden_drifts", []))}

Route-aware regeneration guidance:
{route_guidance}

Blueprint execution summary:
- Scene opening: {blueprint.get("scene_opening", "")}
- Narrative beat: {blueprint.get("narrative_beat", "")}
- Closing beat: {blueprint.get("closing_beat", "")}
- Condition rationale: {blueprint.get("condition_rationale", "")}
- Prompt-ready summary: {final_generation_package.get("prompt_ready_summary", blueprint.get("prompt_ready_summary", ""))}
{score_target_block}
{revision_block}
Important:
- Preserve the same product, influencer, scene, style, and condition logic aligned with the validated blueprint v2.
- Treat storyboard panels as continuity checkpoints, not as separate video outputs.
- Preserve temporal coherence, product identity, and creator identity across the full clip.
- Do not drift into a different product demo, setting, or creator identity.
- Maintain TikTok-native realism rather than polished commercial aesthetics.
- If critical repair constraints are provided, treat them as mandatory fixes rather than optional suggestions.
""".strip()


# =========================================================
# Model clients
# =========================================================


def _get_client() -> Any:
    if genai is None:
        raise RuntimeError("google-genai is not installed. Please install the SDK first.")
    return genai.Client()


# =========================================================
# Generation and analysis engines
# =========================================================


def generate_video_from_prompt(model: str, prompt: str, output_path: Path, poll_seconds: int = 10) -> None:
    client = _get_client()
    operation = client.models.generate_videos(model=model, prompt=prompt)

    while not getattr(operation, "done", False):
        time.sleep(poll_seconds)
        operation = client.operations.get(operation)

    response = getattr(operation, "response", None)
    generated_videos = getattr(response, "generated_videos", None) if response is not None else None
    if not generated_videos:
        raise RuntimeError("Video generation completed without any generated_videos in the response.")

    generated_video = generated_videos[0]
    video_file = getattr(generated_video, "video", None)
    if video_file is None:
        raise RuntimeError("Generated video response did not include a downloadable video file.")

    client.files.download(file=video_file)
    video_file.save(str(output_path))


def _load_genai_image_from_local_path(path_text: str) -> Any:
    """Load a local image using the installed google.genai SDK's actual API shape."""
    if genai_types is None:
        raise RuntimeError("google.genai.types is not available in the current environment.")
    return genai_types.Image.from_file(location=path_text)


def generate_video_from_multimodal_package(
    model: str,
    prompt: str,
    resolved_multimodal_inputs: Dict[str, Any],
    output_path: Path,
    poll_seconds: int = 10,
) -> Dict[str, Any]:
    if genai_types is None:
        raise RuntimeError("google.genai.types is not available in the current environment, so multimodal Veo generation cannot run.")

    client = _get_client()
    generation_mode = str(resolved_multimodal_inputs.get("generation_mode", "text_only") or "text_only")
    multimodal_ready = bool(resolved_multimodal_inputs.get("multimodal_ready", False))
    effective_mode = (
        str(resolved_multimodal_inputs.get("effective_generation_mode_hint", "") or "").strip()
        or generation_mode
    )
    if effective_mode not in VIDEO_INPUT_MODES:
        effective_mode = generation_mode
    if generation_mode != "text_only" and not multimodal_ready:
        effective_mode = "text_only"

    first_frame_path = str(resolved_multimodal_inputs.get("first_frame_image_path", "") or "").strip()
    last_frame_path = str(resolved_multimodal_inputs.get("last_frame_image_path", "") or "").strip()
    reference_paths = [
        str(path).strip()
        for path in resolved_multimodal_inputs.get("reference_image_paths", []) or []
        if str(path).strip() and Path(str(path).strip()).exists()
    ][:4]
    reference_sources = list(resolved_multimodal_inputs.get("reference_image_sources", []) or [])

    config_kwargs: Dict[str, Any] = {}
    image_input = None

    if effective_mode in {"first_frame_to_video", "first_frame_plus_reference_images"} and first_frame_path:
        image_input = _load_genai_image_from_local_path(first_frame_path)
    if effective_mode in {"first_frame_to_video", "first_frame_plus_reference_images"} and last_frame_path:
        config_kwargs["last_frame"] = _load_genai_image_from_local_path(last_frame_path)
    if effective_mode in {"reference_images_to_video", "first_frame_plus_reference_images"} and reference_paths:
        reference_images = []
        for idx, path in enumerate(reference_paths):
            asset_type = reference_sources[idx] if idx < len(reference_sources) else ""
            ref_type = (
                genai_types.VideoGenerationReferenceType.STYLE
                if asset_type == "style_reference"
                else genai_types.VideoGenerationReferenceType.ASSET
            )
            reference_images.append(
                genai_types.VideoGenerationReferenceImage(
                    image=_load_genai_image_from_local_path(path),
                    reference_type=ref_type,
                )
            )
        if reference_images:
            config_kwargs["reference_images"] = reference_images

    operation = client.models.generate_videos(
        model=model,
        prompt=prompt,
        image=image_input,
        config=genai_types.GenerateVideosConfig(**config_kwargs) if config_kwargs else None,
    )

    while not getattr(operation, "done", False):
        time.sleep(poll_seconds)
        operation = client.operations.get(operation)

    response = getattr(operation, "response", None)
    generated_videos = getattr(response, "generated_videos", None) if response is not None else None
    if not generated_videos:
        raise RuntimeError("Video generation completed without any generated_videos in the response.")

    generated_video = generated_videos[0]
    video_file = getattr(generated_video, "video", None)
    if video_file is None:
        raise RuntimeError("Generated video response did not include a downloadable video file.")

    client.files.download(file=video_file)
    video_file.save(str(output_path))
    return {
        "requested_mode": generation_mode,
        "effective_mode": effective_mode,
        "used_first_frame": bool(image_input is not None),
        "used_last_frame": bool(config_kwargs.get("last_frame")),
        "used_reference_images": len(reference_paths),
    }


def _extract_json_from_text(text: str) -> Dict[str, Any]:
    text = text.strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            return {"raw_text": text}
    return {"raw_text": text}


def normalize_candidate_blueprint(
    blueprint: Dict[str, Any],
    config: Dict[str, Any],
    condition: Literal["high_integration", "hard_sell"],
    run_id: Optional[str] = None,
    candidate_index: Optional[int] = None,
) -> Dict[str, Any]:
    raw = blueprint if isinstance(blueprint, dict) else {}
    duration_seconds = max(4.0, _extract_first_number(config.get("duration_hint"), 8.0))
    controls = get_condition_controls(config, condition)
    shared_stable_elements = build_shared_stimulus_skeleton(config)
    condition_layer_raw = raw.get("condition_control_layer") if isinstance(raw.get("condition_control_layer"), dict) else {}

    product_entry_timing_seconds = round(
        _clamp_float(
            raw.get("product_entry_timing_seconds", condition_layer_raw.get("product_entry_timing_seconds")),
            0.0,
            duration_seconds,
            duration_seconds / 2.0,
        ),
        2,
    )
    product_entry_mode = _normalize_enum(
        raw.get("product_entry_mode", condition_layer_raw.get("product_entry_mode")),
        BLUEPRINT_PRODUCT_ENTRY_MODES,
        "",
    )
    speaking_mode = _normalize_enum(
        raw.get("speaking_mode", condition_layer_raw.get("speaking_mode")),
        BLUEPRINT_SPEAKING_MODES,
        "",
    )
    product_focus_pattern = _normalize_enum(
        raw.get("product_focus_pattern", condition_layer_raw.get("product_focus_pattern")),
        BLUEPRINT_PRODUCT_FOCUS_PATTERNS,
        "",
    )
    persuasive_intensity = _normalize_enum(
        raw.get("persuasive_intensity", condition_layer_raw.get("persuasive_intensity")),
        BLUEPRINT_PERSUASIVE_INTENSITIES,
        "",
    )
    narrative_interruption = _normalize_enum(
        raw.get("narrative_interruption", condition_layer_raw.get("narrative_interruption")),
        BLUEPRINT_NARRATIVE_INTERRUPTION_LEVELS,
        "",
    )
    direct_address_to_camera = _normalize_enum(
        raw.get("direct_address_to_camera", condition_layer_raw.get("direct_address_to_camera")),
        BLUEPRINT_DIRECT_ADDRESS_LEVELS,
        "",
    )
    call_to_action_presence = _normalize_enum(
        raw.get("call_to_action_presence", condition_layer_raw.get("call_to_action_presence")),
        BLUEPRINT_CALL_TO_ACTION_LEVELS,
        "",
    )
    product_centrality = _normalize_enum(
        raw.get("product_centrality", condition_layer_raw.get("product_centrality")),
        BLUEPRINT_PRODUCT_CENTRALITY_LEVELS,
        "",
    )
    temporal_raw = raw.get("temporal_plan") if isinstance(raw.get("temporal_plan"), dict) else {}
    beats_raw = temporal_raw.get("beats") if isinstance(temporal_raw.get("beats"), list) else raw.get("beats")
    temporal_beats = _normalize_temporal_beats(beats_raw, condition, controls, duration_seconds)
    asset_plan = _normalize_asset_plan(raw.get("asset_plan"), condition)
    storyboard_plan = _normalize_storyboard_plan(raw.get("storyboard_plan"), temporal_beats, condition)
    execution_plan = _normalize_execution_plan(raw.get("execution_plan"), temporal_beats, condition, controls)
    normalized = {
        "blueprint_version": "v2",
        "run_id": _clean_text(run_id or raw.get("run_id") or config.get("run_id")),
        "condition": condition,
        "candidate_index": _coerce_int(candidate_index if candidate_index is not None else raw.get("candidate_index"), 0),
        "shared_skeleton": shared_stable_elements,
        "shared_stable_elements": shared_stable_elements,
        "scene_opening": _clean_text(raw.get("scene_opening")),
        "narrative_beat": _clean_text(raw.get("narrative_beat")),
        "product_entry_timing_seconds": product_entry_timing_seconds,
        "product_entry_mode": product_entry_mode,
        "speaking_mode": speaking_mode,
        "camera_behavior": _clean_text(raw.get("camera_behavior")),
        "product_focus_pattern": product_focus_pattern,
        "persuasive_intensity": persuasive_intensity,
        "narrative_interruption": narrative_interruption,
        "direct_address_to_camera": direct_address_to_camera,
        "call_to_action_presence": call_to_action_presence,
        "product_centrality": product_centrality,
        "closing_beat": _clean_text(raw.get("closing_beat")),
        "condition_rationale": _clean_text(raw.get("condition_rationale")),
        "prompt_ready_summary": _clean_text(raw.get("prompt_ready_summary")),
        "condition_controls": controls,
        "condition_control_layer": {
            "condition": condition,
            "product_entry_timing_seconds": product_entry_timing_seconds,
            "product_entry_mode": product_entry_mode,
            "speaking_mode": speaking_mode,
            "product_focus_pattern": product_focus_pattern,
            "persuasive_intensity": persuasive_intensity,
            "narrative_interruption": narrative_interruption,
            "direct_address_to_camera": direct_address_to_camera,
            "call_to_action_presence": call_to_action_presence,
            "product_centrality": product_centrality,
        },
        "temporal_plan": {
            "beats": temporal_beats,
        },
        "asset_plan": asset_plan,
        "storyboard_plan": storyboard_plan,
        "execution_plan": execution_plan,
    }
    return normalized


def validate_candidate_blueprint(
    blueprint: Dict[str, Any],
    config: Dict[str, Any],
    condition: Literal["high_integration", "hard_sell"],
) -> Dict[str, Any]:
    normalized = normalize_candidate_blueprint(
        blueprint=blueprint,
        config=config,
        condition=condition,
        run_id=config.get("run_id"),
        candidate_index=blueprint.get("candidate_index") if isinstance(blueprint, dict) else None,
    )

    errors: List[str] = []
    warnings: List[str] = []
    control_mismatches: List[str] = []
    controls = get_condition_controls(config, condition)

    for key in STABLE_SKELETON_FIELDS:
        if not _clean_text(normalized["shared_skeleton"].get(key)):
            errors.append(f"Shared skeleton field is missing: {key}")

    for key in STABLE_SKELETON_FIELDS:
        if not _clean_text(normalized["shared_stable_elements"].get(key)):
            errors.append(f"Blueprint V2 shared_stable_elements is missing: {key}")

    for field in BLUEPRINT_REQUIRED_FIELDS:
        value = normalized.get(field)
        if isinstance(value, str) and not value.strip():
            errors.append(f"Blueprint field is required: {field}")

    condition_layer = normalized.get("condition_control_layer", {}) if isinstance(normalized.get("condition_control_layer"), dict) else {}
    for field in [
        "condition",
        "product_entry_timing_seconds",
        "product_entry_mode",
        "speaking_mode",
        "product_focus_pattern",
        "persuasive_intensity",
        "narrative_interruption",
        "direct_address_to_camera",
        "call_to_action_presence",
        "product_centrality",
    ]:
        if field not in condition_layer or _clean_text(condition_layer.get(field)) == "":
            errors.append(f"Blueprint V2 condition_control_layer is missing: {field}")

    if normalized.get("condition") != condition:
        errors.append("Blueprint condition does not match the requested condition.")

    if normalized["speaking_mode"] not in BLUEPRINT_SPEAKING_MODES:
        errors.append("speaking_mode is invalid.")
    if normalized["persuasive_intensity"] not in BLUEPRINT_PERSUASIVE_INTENSITIES:
        errors.append("persuasive_intensity is invalid.")
    if normalized["product_entry_mode"] not in BLUEPRINT_PRODUCT_ENTRY_MODES:
        errors.append("product_entry_mode is invalid.")
    if normalized["product_focus_pattern"] not in BLUEPRINT_PRODUCT_FOCUS_PATTERNS:
        errors.append("product_focus_pattern is invalid.")
    if normalized["narrative_interruption"] not in BLUEPRINT_NARRATIVE_INTERRUPTION_LEVELS:
        errors.append("narrative_interruption is invalid.")
    if normalized["direct_address_to_camera"] not in BLUEPRINT_DIRECT_ADDRESS_LEVELS:
        errors.append("direct_address_to_camera is invalid.")
    if normalized["call_to_action_presence"] not in BLUEPRINT_CALL_TO_ACTION_LEVELS:
        errors.append("call_to_action_presence is invalid.")
    if normalized["product_centrality"] not in BLUEPRINT_PRODUCT_CENTRALITY_LEVELS:
        errors.append("product_centrality is invalid.")
    if not normalized["prompt_ready_summary"]:
        errors.append("prompt_ready_summary cannot be empty.")

    timing = numeric_score(normalized.get("product_entry_timing_seconds"), 0.0)
    duration_seconds = max(4.0, _extract_first_number(config.get("duration_hint"), 8.0))
    if timing < 0 or timing > duration_seconds:
        errors.append("product_entry_timing_seconds is outside the allowed duration range.")

    beats = normalized.get("temporal_plan", {}).get("beats", []) if isinstance(normalized.get("temporal_plan"), dict) else []
    if len(beats) < 2 or len(beats) > 4:
        errors.append("Blueprint V2 temporal_plan must contain 2 to 4 beats.")
    previous_end = -0.01
    beat_ids = set()
    for beat in beats:
        beat_id = _clean_text(beat.get("beat_id"))
        if not beat_id:
            errors.append("Each temporal beat must include beat_id.")
        if beat_id in beat_ids:
            errors.append(f"Duplicate temporal beat_id detected: {beat_id}")
        beat_ids.add(beat_id)
        start = numeric_score(beat.get("approximate_start_seconds"), 0.0)
        end = numeric_score(beat.get("approximate_end_seconds"), 0.0)
        if end <= start:
            errors.append(f"Temporal beat {beat_id or '(unknown)'} has non-positive duration.")
        if start < previous_end - 0.25:
            warnings.append(f"Temporal beat {beat_id or '(unknown)'} overlaps earlier beats in a non-sequential way.")
        previous_end = max(previous_end, end)
        for field in ["beat_label", "beat_goal", "product_visibility_role", "speaking_behavior", "camera_intent", "condition_signal_role"]:
            if not _clean_text(beat.get(field)):
                warnings.append(f"Temporal beat {beat_id or '(unknown)'} is missing detail for {field}.")

    asset_plan = normalized.get("asset_plan", {}) if isinstance(normalized.get("asset_plan"), dict) else {}
    for field in [
        "scene_reference_required",
        "product_reference_required",
        "character_reference_required",
        "style_reference_required",
        "optional_condition_cue_reference_required",
        "notes_on_asset_usage",
    ]:
        if field not in asset_plan:
            errors.append(f"Blueprint V2 asset_plan is missing: {field}")

    storyboard_plan = normalized.get("storyboard_plan", {}) if isinstance(normalized.get("storyboard_plan"), dict) else {}
    panels = storyboard_plan.get("panels", []) if isinstance(storyboard_plan.get("panels"), list) else []
    if len(panels) < 2 or len(panels) > 4:
        errors.append("Blueprint V2 storyboard_plan must contain 2 to 4 panels.")
    for panel in panels:
        panel_id = _clean_text(panel.get("panel_id"))
        linked_beat_id = _clean_text(panel.get("linked_beat_id"))
        if not panel_id:
            errors.append("Each storyboard panel must include panel_id.")
        if linked_beat_id and linked_beat_id not in beat_ids:
            warnings.append(f"Storyboard panel {panel_id or '(unknown)'} references missing beat_id {linked_beat_id}.")
        for field in ["panel_goal", "visual_focus"]:
            if not _clean_text(panel.get(field)):
                warnings.append(f"Storyboard panel {panel_id or '(unknown)'} is missing {field}.")

    execution_plan = normalized.get("execution_plan", {}) if isinstance(normalized.get("execution_plan"), dict) else {}
    for field in [
        "overall_scene_intent",
        "beat_by_beat_action_instructions",
        "dialogue_outline",
        "action_outline",
        "product_interaction_outline",
        "direct_address_timing",
        "cta_behavior",
        "camera_behavior_notes",
        "realism_priorities",
        "continuity_priorities",
        "forbidden_drifts",
    ]:
        if field not in execution_plan:
            errors.append(f"Blueprint V2 execution_plan is missing: {field}")

    for field in [
        "product_entry_mode",
        "speaking_mode",
        "product_focus_pattern",
        "persuasive_intensity",
        "narrative_interruption",
        "direct_address_to_camera",
        "call_to_action_presence",
        "product_centrality",
    ]:
        if normalized.get(field) != controls.get(field):
            control_mismatches.append(
                f"{field} must be '{controls.get(field)}' but blueprint returned '{normalized.get(field)}'."
            )
    if abs(timing - numeric_score(controls.get("product_entry_timing_seconds"), timing)) > 0.5:
        control_mismatches.append(
            "product_entry_timing_seconds must stay within ±0.5 seconds of the condition control value "
            f"({controls.get('product_entry_timing_seconds')})."
        )

    if control_mismatches:
        errors.extend(control_mismatches)

    rationale = normalized.get("condition_rationale", "")
    speaking_mode = normalized["speaking_mode"]
    focus_pattern = normalized["product_focus_pattern"]
    entry_mode = normalized["product_entry_mode"]
    intensity = normalized["persuasive_intensity"]

    if condition == "high_integration":
        if timing <= 1.0 and focus_pattern == "dominant_focus":
            errors.append("high_integration should not open with immediate dominant product focus.")
        if speaking_mode == "direct_pitch":
            errors.append("high_integration should not use fully direct_pitch speaking mode.")
        if intensity == "high":
            errors.append("high_integration persuasive_intensity should not be high.")
        if focus_pattern == "dominant_focus":
            warnings.append("high_integration usually works better without dominant product focus.")
        if entry_mode in {"direct_showcase", "pitch_opening"}:
            warnings.append("high_integration usually benefits from less direct product entry.")
        if not _contains_any(rationale, ["story", "narrative", "organic", "natural", "lived", "moment", "integrat"]):
            warnings.append("high_integration rationale should describe story-led or organic integration.")
    else:
        if speaking_mode not in {"direct_pitch", "mixed"}:
            errors.append("hard_sell should use direct_pitch or mixed speaking mode.")
        if focus_pattern not in {"shared_focus", "dominant_focus"}:
            errors.append("hard_sell should keep the product as shared_focus or dominant_focus.")
        if intensity != "high":
            errors.append("hard_sell persuasive_intensity should be high.")
        if timing > max(2.5, duration_seconds * 0.45):
            warnings.append("hard_sell product entry is later than expected for a direct pitch stimulus.")
        if entry_mode not in {"direct_showcase", "pitch_opening", "embedded_use"}:
            warnings.append("hard_sell usually benefits from direct_showcase or pitch_opening.")
        if not _contains_any(rationale, ["promotion", "promotional", "direct", "recommend", "sell", "pitch", "explicit"]):
            warnings.append("hard_sell rationale should make explicit promotional intent clear.")

    condition_signals = {
        "condition": condition,
        "condition_control_summary": controls,
        "control_compliance_ok": len(control_mismatches) == 0,
        "story_led": speaking_mode in {"narrative", "mixed"} and focus_pattern in {"background", "intermittent", "shared_focus"},
        "direct_pitch_present": speaking_mode in {"direct_pitch", "mixed"},
        "product_focality": focus_pattern,
        "persuasive_intensity": intensity,
        "product_entry_seconds": timing,
        "product_entry_mode": entry_mode,
        "summary": (
            f"{condition}: speaking={speaking_mode}, focus={focus_pattern}, intensity={intensity}, "
            f"entry={timing:.2f}s via {entry_mode}"
        ),
    }

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "control_mismatches": control_mismatches,
        "control_compliance_ok": len(control_mismatches) == 0,
        "normalized_blueprint": normalized,
        "condition_signals": condition_signals,
        "condition_signal_summary": condition_signals.get("summary", ""),
    }


def generate_candidate_blueprint(
    run_id: str,
    condition: Literal["high_integration", "hard_sell"],
    candidate_index: int,
    revision_instruction_text: Optional[str] = None,
    blueprint_model: Optional[str] = None,
    edited_score_targets: Optional[Dict[str, Any]] = None,
    regeneration_route: Optional[str] = None,
) -> Dict[str, Any]:
    config = load_run_config(run_id)
    model_name = str(blueprint_model or config.get("blueprint_model") or DEFAULT_BLUEPRINT_MODEL)
    max_attempts = max(1, _coerce_int(config.get("blueprint_max_attempts"), DEFAULT_BLUEPRINT_MAX_ATTEMPTS))
    prompt = build_blueprint_prompt(
        config=config,
        condition=condition,
        revision_instruction_text=revision_instruction_text,
        edited_score_targets=edited_score_targets,
        regeneration_route=regeneration_route,
    )
    client = _get_client()

    last_validation: Dict[str, Any] = {}
    last_blueprint: Dict[str, Any] = {}
    last_raw_text = ""
    errors: List[str] = []

    for attempt in range(1, max_attempts + 1):
        try:
            response = client.models.generate_content(model=model_name, contents=prompt)
            last_raw_text = getattr(response, "text", "") or ""
            parsed = _extract_json_from_text(last_raw_text)
            last_blueprint = normalize_candidate_blueprint(
                parsed,
                config=config,
                condition=condition,
                run_id=run_id,
                candidate_index=candidate_index,
            )
            last_validation = validate_candidate_blueprint(last_blueprint, config, condition)
            if last_validation.get("ok"):
                return {
                    "ok": True,
                    "blueprint_status": "BLUEPRINT_READY",
                    "blueprint_attempts": attempt,
                    "blueprint_model": model_name,
                    "blueprint": last_validation.get("normalized_blueprint", last_blueprint),
                    "validation": last_validation,
                    "raw_text": last_raw_text,
                    "generation_prompt": prompt,
                }
            validation_errors = last_validation.get("errors", [])
            errors.append(f"Attempt {attempt}: {'; '.join(validation_errors) or 'Blueprint validation failed.'}")
            if attempt < max_attempts:
                prompt = (
                    f"{build_blueprint_prompt(config=config, condition=condition, revision_instruction_text=revision_instruction_text, edited_score_targets=edited_score_targets, regeneration_route=regeneration_route)}\n\n"
                    "Previous output failed validation. Fix these issues and return strict JSON only:\n"
                    f"{json.dumps(validation_errors, ensure_ascii=False)}"
                )
        except Exception as exc:
            errors.append(f"Attempt {attempt}: {exc}")

    return {
        "ok": False,
        "blueprint_status": "BLUEPRINT_FAILED",
        "blueprint_attempts": max_attempts,
        "blueprint_model": model_name,
        "blueprint": last_validation.get("normalized_blueprint", last_blueprint),
        "validation": last_validation
        or {
            "ok": False,
            "errors": errors or ["Blueprint generation failed before a valid JSON response was produced."],
            "warnings": [],
            "normalized_blueprint": last_blueprint,
            "condition_signals": {},
        },
        "raw_text": last_raw_text,
        "generation_prompt": prompt,
        "error_message": " | ".join(errors),
    }


def build_video_analysis_prompt(condition: str, generation_prompt: str) -> str:
    return f"""
You are evaluating an AI-generated short-form social media advertising video for experimental stimulus development.

Target condition: {condition}
Original generation prompt:
{generation_prompt}

Your task is to produce a research-oriented diagnostic for stimulus screening.
Return STRICT JSON only.

IMPORTANT PRIORITY RULES:
1. Your summary must be driven by the scoring dimensions, not by whichever flaw is easiest to describe.
2. If any score is weak (1-3), you must explicitly mention that weakness in the summary or main_issue.
3. If AI naturalness or non-uncanny naturalness is weak (1-3), you must explicitly discuss synthetic-looking cues or unnaturalness.
4. If realism is weak (1-3), you must explicitly discuss why the video may fail as believable creator content.
5. If object continuity, prop identity stability, hand-object consistency, or temporal coherence breaks down, treat that as a serious validity threat.
6. The main_issue must be anchored in one of the weakest scoring dimensions or in a severe temporal/object continuity failure.
7. Prompt revision suggestions must map to weak dimensions. Do not give generic advice.
8. You must identify concrete repairable failure modes rather than vague high-level complaints.
9. Connect weak scores -> failure modes -> repair targets -> short repair instructions.
10. Prefer repair-relevant labels like lip_sync_mismatch, stiff_body_motion, object_continuity_failure, prop_switching_artifact, duplicate_object_artifact, hand_object_inconsistency, weak_condition_signal.

LAYER 1: OBJECTIVE VIDEO UNDERSTANDING
Describe observable facts only. Do not jump to conclusions.
You must explicitly inspect object continuity, prop consistency, hand-object interaction consistency, and temporal coherence across the full clip.
If a prop duplicates, switches identity, appears/disappears without scene logic, or becomes inconsistent across adjacent moments, record that directly rather than collapsing it into generic uncanniness.
Capture:
- scene_summary
- narrative_flow_summary
- product_first_appearance_seconds
- product_visible_most_of_video
- product_is_visual_focal_point
- speaking_style (choose one: narrative, direct_pitch, mixed, no_speech)
- unnatural_motion_present
- character_identity_consistent
- object_continuity_ok
- hand_object_interaction_consistent
- unexpected_extra_objects_present
- prop_identity_stable
- temporal_coherence_ok
- key_observations (list)

LAYER 2: RAISE-STYLE SCORING SUGGESTIONS
Return integer scores from 1 to 7 for:
- realism_score_1to7
- platform_fit_score_1to7
- integration_score_1to7
- commercial_salience_score_1to7
- character_consistency_score_1to7
- ai_naturalness_score_1to7
- non_uncanny_naturalness_score_1to7

Rubric guidance:
- realism: plausible real creator content rather than obviously synthetic content
- platform_fit: native TikTok/Reels feel rather than polished commercial aesthetics
- integration: natural embedding of the product into story/context
- commercial_salience: strength and obviousness of persuasive ad intent
- character_consistency: stable influencer identity, behavior, and continuity
- ai_naturalness: how successfully the video avoids looking AI-generated
- non_uncanny_naturalness: how natural, human-like, and non-uncanny the performance feels

Condition-specific expectations:
- If target condition is high_integration, integration should be high and commercial salience should not dominate the video.
- If target condition is hard_sell, commercial salience should be high and direct promotional framing should be obvious.

LAYER 3: SCORE-ANCHORED DIAGNOSTIC
You must identify which dimensions are strongest and weakest.

Rules for interpretation:
- strongest_dimensions: choose the 2 most successful scoring dimensions
- weakest_dimensions: choose the 2 weakest scoring dimensions
- condition_success_dimensions: include the dimension(s) that most directly show whether the target condition was successfully realized
- validity_risks: include any dimensions that threaten experimental usefulness, especially realism, ai_naturalness, non_uncanny_naturalness, object continuity, hand-object consistency, prop stability, and temporal coherence

Rules for summary:
- Write 2-4 sentences.
- Sentence 1 should say whether the target condition was successfully realized.
- Sentence 2 should mention the most important validity risk(s).
- If ai_naturalness_score_1to7 <= 3, explicitly mention synthetic/AI-generated appearance concerns.
- If non_uncanny_naturalness_score_1to7 <= 3, explicitly mention uncanny or unnatural motion/expression/timing concerns.
- If realism_score_1to7 <= 3, explicitly mention low believability.
- If object_continuity_ok, prop_identity_stable, hand_object_interaction_consistent, or temporal_coherence_ok are false, explicitly mention the continuity failure in summary or main_issue.

Rules for main_issue:
- Must be the single most important weakness for stimulus quality.
- Must usually come from weakest_dimensions.
- Must prioritize validity threats over cosmetic issues.

LAYER 4: FAILURE DIAGNOSIS AND REPAIR PLANNING
You must explicitly classify repair-relevant failure modes.

Rules for failure_modes:
- Return 2 to 5 short standardized labels.
- Use concrete repair categories, not vague abstractions.
- Prefer labels like:
  lip_sync_mismatch
  unnatural_expression_timing
  stiff_body_motion
  synthetic_face_cues
  scripted_delivery
  overpolished_ad_tone
  weak_condition_signal
  unstable_character_identity
  robotic_gesture_timing
  unrealistic_text_reflection
  object_continuity_failure
  prop_switching_artifact
  extra_object_artifact
  hand_object_inconsistency
  temporal_coherence_failure
  duplicate_object_artifact
- Choose labels that would directly guide regeneration edits.

Rules for primary_repair_targets:
- Return the 2 to 4 most important repair targets.
- Phrase them as short concrete weaknesses to repair.
- Focus on issues that are realistically fixable in one more generation pass.

Rules for non_negotiable_fixes:
- Return must-fix issues only.
- Include items that would make the stimulus unusable if not corrected.
- Keep each item short and direct.

Rules for regeneration_likelihood_of_success:
- Return one of: high, medium, low.
- Base it on whether the failure modes seem repairable in another targeted generation attempt.

Rules for prompt_revision_suggestions:
- Return 3 to 5 suggestions only.
- Each suggestion must target exactly one failure mode.
- Each suggestion must be short, direct, and operational.
- Use imperative phrasing.
- Keep each suggestion ideally under 25 words.
- Do not combine multiple unrelated fixes into one sentence.
- Do not use vague language like "improve realism" unless tied to a concrete failure mode.
- Write them like repair instructions, not like broad advice.
- If continuity failures are detected, suggestions should explicitly mention object continuity, prop identity stability, hand-object consistency, duplicate objects, or temporal coherence.

Return JSON with this exact schema:
{{
  "condition": "{condition}",
  "objective_observations": {{
    "scene_summary": "",
    "narrative_flow_summary": "",
    "product_first_appearance_seconds": 0.0,
    "product_visible_most_of_video": false,
    "product_is_visual_focal_point": false,
    "speaking_style": "narrative",
    "unnatural_motion_present": false,
    "character_identity_consistent": true,
    "object_continuity_ok": true,
    "hand_object_interaction_consistent": true,
    "unexpected_extra_objects_present": false,
    "prop_identity_stable": true,
    "temporal_coherence_ok": true,
    "key_observations": ["", ""]
  }},
  "raise_scoring": {{
    "realism_score_1to7": 0,
    "platform_fit_score_1to7": 0,
    "integration_score_1to7": 0,
    "commercial_salience_score_1to7": 0,
    "character_consistency_score_1to7": 0,
    "ai_naturalness_score_1to7": 0,
    "non_uncanny_naturalness_score_1to7": 0
  }},
  "score_justifications": {{
    "realism_score_1to7": "",
    "platform_fit_score_1to7": "",
    "integration_score_1to7": "",
    "commercial_salience_score_1to7": "",
    "character_consistency_score_1to7": "",
    "ai_naturalness_score_1to7": "",
    "non_uncanny_naturalness_score_1to7": ""
  }},
  "diagnostic_focus": {{
    "strongest_dimensions": ["", ""],
    "weakest_dimensions": ["", ""],
    "condition_success_dimensions": ["", ""],
    "validity_risks": ["", ""]
  }},
  "summary": "",
  "strengths": ["", ""],
  "major_issues": ["", ""],
  "main_issue": "",
  "keep_recommendation": "yes",
  "revision_priority": "",
  "failure_modes": ["", ""],
  "primary_repair_targets": ["", ""],
  "non_negotiable_fixes": ["", ""],
  "regeneration_likelihood_of_success": "medium",
  "prompt_revision_suggestions": ["", "", ""]
}}
""".strip()


def normalize_analysis_result(parsed: Dict[str, Any], condition: str) -> Dict[str, Any]:
    result = parsed if isinstance(parsed, dict) else {}

    observations = result.get("objective_observations")
    if not isinstance(observations, dict):
        observations = {}

    raw_scores = result.get("raise_scoring")
    if not isinstance(raw_scores, dict):
        raw_scores = {}
    raw_scores = remap_score_keys(raw_scores)

    raw_score_justifications = result.get("score_justifications")
    if not isinstance(raw_score_justifications, dict):
        raw_score_justifications = {}
    raw_score_justifications = remap_score_keys(raw_score_justifications)

    raw_diagnostic_focus = result.get("diagnostic_focus")
    if not isinstance(raw_diagnostic_focus, dict):
        raw_diagnostic_focus = {}

    normalized_scores = {
        "realism_score_1to7": _clamp_score(raw_scores.get("realism_score_1to7"), 4),
        "platform_fit_score_1to7": _clamp_score(raw_scores.get("platform_fit_score_1to7"), 4),
        "integration_score_1to7": _clamp_score(raw_scores.get("integration_score_1to7"), 4),
        "commercial_salience_score_1to7": _clamp_score(raw_scores.get("commercial_salience_score_1to7"), 4),
        "character_consistency_score_1to7": _clamp_score(raw_scores.get("character_consistency_score_1to7"), 4),
        AI_NATURALNESS_KEY: _clamp_score(get_score_with_backward_compat(raw_scores, AI_NATURALNESS_KEY, 4), 4),
        NON_UNCANNY_NATURALNESS_KEY: _clamp_score(get_score_with_backward_compat(raw_scores, NON_UNCANNY_NATURALNESS_KEY, 4), 4),
    }

    likelihood = _clean_text(result.get("regeneration_likelihood_of_success")).lower()
    if likelihood not in {"high", "medium", "low"}:
        likelihood = ""

    normalized = {
        "condition": _clean_text(result.get("condition")) or condition,
        "objective_observations": {
            "scene_summary": _clean_text(observations.get("scene_summary")),
            "narrative_flow_summary": _clean_text(observations.get("narrative_flow_summary")),
            "product_first_appearance_seconds": observations.get("product_first_appearance_seconds", 0.0),
            "product_visible_most_of_video": bool(observations.get("product_visible_most_of_video", False)),
            "product_is_visual_focal_point": bool(observations.get("product_is_visual_focal_point", False)),
            "speaking_style": _clean_text(observations.get("speaking_style")) or "",
            "unnatural_motion_present": bool(observations.get("unnatural_motion_present", False)),
            "character_identity_consistent": bool(observations.get("character_identity_consistent", False)),
            "object_continuity_ok": bool(observations.get("object_continuity_ok", True)),
            "hand_object_interaction_consistent": bool(observations.get("hand_object_interaction_consistent", True)),
            "unexpected_extra_objects_present": bool(observations.get("unexpected_extra_objects_present", False)),
            "prop_identity_stable": bool(observations.get("prop_identity_stable", True)),
            "temporal_coherence_ok": bool(observations.get("temporal_coherence_ok", True)),
            "key_observations": _normalize_short_list(observations.get("key_observations"), 8),
        },
        "raise_scoring": normalized_scores,
        "scores": normalized_scores,
        "score_justifications": {
            "realism_score_1to7": _clean_text(raw_score_justifications.get("realism_score_1to7")),
            "platform_fit_score_1to7": _clean_text(raw_score_justifications.get("platform_fit_score_1to7")),
            "integration_score_1to7": _clean_text(raw_score_justifications.get("integration_score_1to7")),
            "commercial_salience_score_1to7": _clean_text(raw_score_justifications.get("commercial_salience_score_1to7")),
            "character_consistency_score_1to7": _clean_text(raw_score_justifications.get("character_consistency_score_1to7")),
            AI_NATURALNESS_KEY: _clean_text(get_score_with_backward_compat(raw_score_justifications, AI_NATURALNESS_KEY, "")),
            NON_UNCANNY_NATURALNESS_KEY: _clean_text(get_score_with_backward_compat(raw_score_justifications, NON_UNCANNY_NATURALNESS_KEY, "")),
        },
        "diagnostic_focus": {
            "strongest_dimensions": _normalize_short_list(raw_diagnostic_focus.get("strongest_dimensions"), 4),
            "weakest_dimensions": _normalize_short_list(raw_diagnostic_focus.get("weakest_dimensions"), 4),
            "condition_success_dimensions": _normalize_short_list(raw_diagnostic_focus.get("condition_success_dimensions"), 4),
            "validity_risks": _normalize_short_list(raw_diagnostic_focus.get("validity_risks"), 6),
        },
        "summary": _clean_text(result.get("summary")),
        "strengths": _normalize_short_list(result.get("strengths"), 6),
        "major_issues": _normalize_short_list(result.get("major_issues"), 6),
        "main_issue": _clean_text(result.get("main_issue")),
        "keep_recommendation": _clean_keep(result.get("keep_recommendation")),
        "revision_priority": _clean_text(result.get("revision_priority")),
        "failure_modes": _normalize_short_list(result.get("failure_modes"), 5),
        "primary_repair_targets": _normalize_short_list(result.get("primary_repair_targets"), 4),
        "non_negotiable_fixes": _normalize_short_list(result.get("non_negotiable_fixes"), 4),
        "regeneration_likelihood_of_success": likelihood,
        "prompt_revision_suggestions": _normalize_short_list(result.get("prompt_revision_suggestions"), 5),
    }

    return normalized


def build_default_repair_patch(analysis_result: Dict[str, Any]) -> str:
    if not isinstance(analysis_result, dict):
        return ""
    lines: List[str] = []
    must_fix = _as_list(analysis_result.get("non_negotiable_fixes"))
    repair_suggestions = _as_list(analysis_result.get("prompt_revision_suggestions"))
    if must_fix:
        lines.append("Critical repair constraints:")
        lines.extend(f"- {item}" for item in must_fix)
    if repair_suggestions:
        if lines:
            lines.append("")
        lines.append("Targeted repair instructions:")
        lines.extend(f"- {item}" for item in repair_suggestions)
    return "\n".join(lines).strip()


def analyze_video_candidate(
    run_id: str,
    condition: Literal["high_integration", "hard_sell"],
    candidate_index: int,
    analysis_model: Optional[str] = None,
) -> Dict[str, Any]:
    config = load_run_config(run_id)
    manifest_rows = load_manifest_rows(run_id)

    target = None
    for row in manifest_rows:
        if str(row.get("condition")) == str(condition) and _coerce_int(row.get("candidate_index", 0)) == int(candidate_index):
            target = row
            break

    if target is None:
        raise ValueError("Candidate not found in manifest.")

    video_path = Path(str(target.get("output_video_path", "")))
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    analysis_model = analysis_model or str(config.get("analysis_model") or DEFAULT_ANALYSIS_MODEL)
    generation_prompt = str(target.get("final_generation_prompt") or target.get("condition_prompt", ""))
    prompt = build_video_analysis_prompt(str(condition), generation_prompt)
    analysis_path = get_analysis_path(run_id, condition, int(candidate_index))

    client = _get_client()

    try:
        uploaded = client.files.upload(file=str(video_path))

        uploaded_name = getattr(uploaded, "name", None)
        if not uploaded_name:
            raise RuntimeError("Uploaded file has no name; cannot poll file state.")

        active_file = None
        max_polls = 60
        poll_seconds = 2

        for _ in range(max_polls):
            current_file = client.files.get(name=uploaded_name)
            state = getattr(current_file, "state", None)
            state_name = getattr(state, "name", str(state)) if state is not None else ""

            if state_name == "ACTIVE":
                active_file = current_file
                break

            if state_name in {"FAILED", "ERROR"}:
                raise RuntimeError(f"Uploaded file processing failed with state={state_name}")

            time.sleep(poll_seconds)

        if active_file is None:
            raise RuntimeError("Uploaded file did not become ACTIVE in time.")

        response = client.models.generate_content(
            model=analysis_model,
            contents=[active_file, prompt],
        )

        text = getattr(response, "text", "") or ""
        parsed = _extract_json_from_text(text)
        normalized_result = normalize_analysis_result(parsed, str(condition))

        payload = {
            "run_id": run_id,
            "condition": condition,
            "candidate_index": int(candidate_index),
            "analysis_model": analysis_model,
            "video_path": str(video_path),
            "analysis_prompt": prompt,
            "analysis_result": normalized_result,
            "raw_text": text,
            "analysis_status": "ANALYSIS_READY",
            "analyzed_at_epoch": int(time.time()),
        }

    except Exception as exc:
        payload = {
            "run_id": run_id,
            "condition": condition,
            "candidate_index": int(candidate_index),
            "analysis_model": analysis_model,
            "video_path": str(video_path),
            "analysis_prompt": prompt,
            "analysis_result": {},
            "raw_text": "",
            "analysis_status": "ANALYSIS_FAILED",
            "error_message": str(exc),
            "analyzed_at_epoch": int(time.time()),
        }

    write_json(analysis_path, payload)

    upsert_manifest_row(
        run_id,
        {
            "condition": condition,
            "candidate_index": int(candidate_index),
            "analysis_json_path": str(analysis_path),
            "analysis_status": payload.get("analysis_status", ""),
            "analysis_model": analysis_model,
            "updated_at_epoch": int(time.time()),
        },
    )

    return payload


def load_analysis(run_id: str, condition: str, candidate_index: int) -> Dict[str, Any]:
    return safe_read_json(get_analysis_path(run_id, condition, int(candidate_index)))


def load_reusable_validated_blueprint(run_id: str, condition: str, candidate_index: int) -> Optional[Dict[str, Any]]:
    payload = safe_read_json(get_blueprint_validation_path(run_id, condition, int(candidate_index)))
    normalized_blueprint = payload.get("normalized_blueprint", {})
    if not isinstance(normalized_blueprint, dict) or not normalized_blueprint:
        return None
    if payload.get("ok") is False:
        return None
    if payload.get("control_compliance_ok") is False:
        return None
    blueprint_status = str(payload.get("blueprint_status", "") or "").strip().upper()
    if blueprint_status and blueprint_status == "BLUEPRINT_FAILED":
        return None
    if str(normalized_blueprint.get("condition", "")).strip() != str(condition):
        return None
    return payload


# =========================================================
# Manifest / screening / final helpers
# =========================================================


def load_manifest_rows(run_id: str) -> List[Dict[str, Any]]:
    rows = safe_read_csv(get_manifest_path(run_id))
    for row in rows:
        row["candidate_index"] = _coerce_int(row.get("candidate_index", 0))
    return rows


def load_screening_rows(run_id: str) -> List[Dict[str, Any]]:
    rows = safe_read_csv(get_screening_path(run_id))
    for row in rows:
        row.update(remap_score_keys(row))
        row["candidate_index"] = _coerce_int(row.get("candidate_index", 0))
    return rows


def load_final_selection_rows(run_id: str) -> List[Dict[str, Any]]:
    rows = safe_read_csv(get_final_selection_path(run_id))
    for row in rows:
        row["candidate_index"] = _coerce_int(row.get("candidate_index", 0))
    return rows


def upsert_manifest_row(run_id: str, row_payload: Dict[str, Any]) -> None:
    path = get_manifest_path(run_id)
    rows = load_manifest_rows(run_id)
    target_key = (str(row_payload.get("condition", "")), _coerce_int(row_payload.get("candidate_index", 0)))
    replaced = False

    for i, row in enumerate(rows):
        key = (str(row.get("condition", "")), _coerce_int(row.get("candidate_index", 0)))
        if key == target_key:
            merged = {**row, **row_payload}
            merged["run_id"] = run_id
            merged["candidate_index"] = _coerce_int(merged.get("candidate_index", 0))
            rows[i] = merged
            replaced = True
            break

    if not replaced:
        new_row = {k: "" for k in MANIFEST_COLUMNS}
        new_row.update(row_payload)
        new_row["run_id"] = run_id
        new_row["candidate_index"] = _coerce_int(new_row.get("candidate_index", 0))
        rows.append(new_row)

    rows = sorted(rows, key=lambda x: (str(x.get("condition", "")), _coerce_int(x.get("candidate_index", 0))))
    write_csv_rows(path, rows, MANIFEST_COLUMNS)


def upsert_screening_row(run_id: str, row_payload: Dict[str, Any]) -> None:
    path = get_screening_path(run_id)
    rows = load_screening_rows(run_id)
    row_payload = remap_score_keys(row_payload)
    target_key = (str(row_payload.get("condition", "")), _coerce_int(row_payload.get("candidate_index", 0)))
    replaced = False

    for i, row in enumerate(rows):
        key = (str(row.get("condition", "")), _coerce_int(row.get("candidate_index", 0)))
        if key == target_key:
            merged = {**row, **row_payload}
            merged["run_id"] = run_id
            merged["candidate_index"] = _coerce_int(merged.get("candidate_index", 0))
            rows[i] = merged
            replaced = True
            break

    if not replaced:
        new_row = {k: "" for k in SCREENING_COLUMNS}
        new_row.update(row_payload)
        new_row["run_id"] = run_id
        new_row["candidate_index"] = _coerce_int(new_row.get("candidate_index", 0))
        rows.append(new_row)

    rows = sorted(rows, key=lambda x: (str(x.get("condition", "")), _coerce_int(x.get("candidate_index", 0))))
    write_csv_rows(path, rows, SCREENING_COLUMNS)


def upsert_final_selection_row(run_id: str, row_payload: Dict[str, Any]) -> None:
    path = get_final_selection_path(run_id)
    rows = load_final_selection_rows(run_id)
    target_key = (str(row_payload.get("condition", "")), _coerce_int(row_payload.get("candidate_index", 0)))
    replaced = False

    for i, row in enumerate(rows):
        key = (str(row.get("condition", "")), _coerce_int(row.get("candidate_index", 0)))
        if key == target_key:
            merged = {**row, **row_payload}
            merged["run_id"] = run_id
            merged["candidate_index"] = _coerce_int(merged.get("candidate_index", 0))
            rows[i] = merged
            replaced = True
            break

    if not replaced:
        new_row = {k: "" for k in FINAL_SELECTION_COLUMNS}
        new_row.update(row_payload)
        new_row["run_id"] = run_id
        new_row["candidate_index"] = _coerce_int(new_row.get("candidate_index", 0))
        rows.append(new_row)

    rows = sorted(rows, key=lambda x: (str(x.get("condition", "")), _coerce_int(x.get("candidate_index", 0))))
    write_csv_rows(path, rows, FINAL_SELECTION_COLUMNS)


# =========================================================
# Orchestration
# =========================================================


def next_candidate_index(run_id: str, condition: str) -> int:
    existing = [r["candidate_index"] for r in load_manifest_rows(run_id) if str(r.get("condition")) == condition]
    return max(existing, default=0) + 1


def generate_single_candidate(
    run_id: str,
    condition: Literal["high_integration", "hard_sell"],
    candidate_index: Optional[int] = None,
    revision_instruction_text: Optional[str] = None,
    edited_score_targets: Optional[Dict[str, Any]] = None,
    regeneration_route: Optional[str] = None,
    multimodal_options: Optional[Dict[str, Any]] = None,
    model_override: Optional[str] = None,
    auto_analyze: Optional[bool] = None,
) -> Dict[str, Any]:
    initialize_run_storage(run_id)
    config = load_run_config(run_id)
    model = str(model_override or config.get("model") or DEFAULT_MODEL)
    blueprint_model = str(config.get("blueprint_model") or DEFAULT_BLUEPRINT_MODEL)
    condition_controls = get_condition_controls(config, condition)
    normalized_multimodal_options = normalize_multimodal_generation_options(multimodal_options, config)
    candidate_index = int(candidate_index or next_candidate_index(run_id, condition))
    should_auto_analyze = bool(config.get("auto_analyze_after_generation", True)) if auto_analyze is None else bool(auto_analyze)

    candidate_dir = get_candidate_dir(run_id, condition, candidate_index)
    metadata_path = get_metadata_path(run_id, condition, candidate_index)
    analysis_path = get_analysis_path(run_id, condition, candidate_index)
    blueprint_path = get_blueprint_path(run_id, condition, candidate_index)
    blueprint_validation_path = get_blueprint_validation_path(run_id, condition, candidate_index)
    asset_pack_path = get_asset_pack_path(run_id, condition, candidate_index)
    asset_manifest_path = get_asset_manifest_path(run_id, condition, candidate_index)
    storyboard_plan_path = get_storyboard_plan_path(run_id, condition, candidate_index)
    storyboard_panel_dir = get_storyboard_panel_dir(run_id, condition, candidate_index)
    execution_script_path = get_execution_script_path(run_id, condition, candidate_index)
    execution_script_text_path = get_execution_script_text_path(run_id, condition, candidate_index)
    final_generation_package_path = get_final_generation_package_path(run_id, condition, candidate_index)
    output_video_path = candidate_dir / f"{condition}_cand_{candidate_index:02d}.mp4"
    route_name = str(regeneration_route or "structural_regenerate").strip().lower()
    reusable_validation = (
        load_reusable_validated_blueprint(run_id, condition, candidate_index)
        if route_name == "surface_regenerate"
        else None
    )
    reuse_existing_blueprint = reusable_validation is not None

    base_prompt = build_base_stimulus_brief(config)
    canonical_condition_prompt = build_condition_prompt(config, condition)

    row = {
        "run_id": run_id,
        "candidate_index": candidate_index,
        "condition": condition,
        "model": model,
        "analysis_model": str(config.get("analysis_model") or DEFAULT_ANALYSIS_MODEL),
        "blueprint_model": blueprint_model,
        "product_name": config.get("product_name", ""),
        "product_category": config.get("product_category", ""),
        "influencer_profile": config.get("influencer_profile", ""),
        "scene_context": config.get("scene_context", ""),
        "disclosure_text": "",
        "duration_hint": config.get("duration_hint", ""),
        "aspect_ratio": config.get("aspect_ratio", ""),
        "visual_style_lock": config.get("visual_style_lock", ""),
        "camera_style": config.get("camera_style", ""),
        "tone_lock": config.get("tone_lock", ""),
        "negative_constraints": config.get("negative_constraints", ""),
        "base_stimulus_brief": base_prompt,
        "base_prompt": base_prompt,
        "condition_prompt": canonical_condition_prompt,
        "regeneration_instruction_text": _clean_text(revision_instruction_text),
        "regeneration_route": route_name,
        "edited_score_targets_json_compact": _compact_json(edited_score_targets or {}),
        "video_input_mode": normalized_multimodal_options.get("generation_mode", "text_only"),
        "multimodal_generation_enabled": "no",
        "resolved_multimodal_inputs_json_compact": "",
        "first_frame_image_path_used": "",
        "last_frame_image_path_used": "",
        "reference_image_paths_used_json_compact": "",
        "storyboard_panel_used_for_generation": "",
        "asset_types_used_for_generation": "",
        "multimodal_input_summary": "",
        "blueprint_json_path": str(blueprint_path),
        "blueprint_validation_json_path": str(blueprint_validation_path),
        "blueprint_status": "BLUEPRINT_PENDING",
        "blueprint_attempts": 0,
        "blueprint_summary": "",
        "blueprint_reused_for_surface_regenerate": "yes" if reuse_existing_blueprint and route_name == "surface_regenerate" else "no",
        "reused_blueprint_source": "validated_blueprint" if reuse_existing_blueprint and route_name == "surface_regenerate" else "",
        "condition_controls_json_compact": _compact_json(condition_controls),
        "control_compliance_ok": "",
        "control_mismatch_summary": "",
        "validated_condition_signal_summary": "",
        "validated_blueprint_json_compact": "",
        "planning_stack_version": "v2",
        "asset_pack_json_path": str(asset_pack_path),
        "asset_manifest_csv_path": str(asset_manifest_path),
        "storyboard_plan_json_path": str(storyboard_plan_path),
        "storyboard_panel_dir": str(storyboard_panel_dir),
        "execution_script_json_path": str(execution_script_path),
        "execution_script_text_path": str(execution_script_text_path),
        "final_generation_package_json_path": str(final_generation_package_path),
        "final_generation_prompt": "",
        "output_dir": str(candidate_dir),
        "output_video_path": str(output_video_path),
        "metadata_json_path": str(metadata_path),
        "analysis_json_path": str(analysis_path),
        "status": "BLUEPRINT_PENDING",
        "analysis_status": "",
        "error_message": "",
        "created_at_epoch": int(time.time()),
        "updated_at_epoch": int(time.time()),
    }

    stale_paths = [
        analysis_path,
        output_video_path,
        asset_pack_path,
        asset_manifest_path,
        storyboard_plan_path,
        execution_script_path,
        execution_script_text_path,
        final_generation_package_path,
    ]
    if not reuse_existing_blueprint:
        stale_paths.extend([blueprint_path, blueprint_validation_path])
    for stale_path in stale_paths:
        if stale_path.exists():
            try:
                stale_path.unlink()
            except Exception:
                pass

    if reuse_existing_blueprint:
        blueprint_payload = {
            "ok": True,
            "blueprint_status": "BLUEPRINT_READY",
            "blueprint_attempts": 0,
            "blueprint_model": blueprint_model,
            "blueprint": reusable_validation.get("normalized_blueprint", {}),
            "validation": reusable_validation,
            "raw_text": "",
            "generation_prompt": "",
        }
        blueprint_data = blueprint_payload["blueprint"]
        validation_data = reusable_validation
    else:
        blueprint_payload = generate_candidate_blueprint(
            run_id=run_id,
            condition=condition,
            candidate_index=candidate_index,
            revision_instruction_text=revision_instruction_text,
            blueprint_model=blueprint_model,
            edited_score_targets=edited_score_targets,
            regeneration_route=route_name,
        )
        blueprint_data = blueprint_payload.get("blueprint", {}) if isinstance(blueprint_payload.get("blueprint"), dict) else {}
        validation_data = blueprint_payload.get("validation", {}) if isinstance(blueprint_payload.get("validation"), dict) else {}

        write_json(
            blueprint_path,
            {
                "run_id": run_id,
                "condition": condition,
                "candidate_index": candidate_index,
                "blueprint_model": blueprint_model,
                "blueprint_status": blueprint_payload.get("blueprint_status", ""),
                "raw_text": blueprint_payload.get("raw_text", ""),
                "blueprint": blueprint_data,
                "generation_prompt": blueprint_payload.get("generation_prompt", ""),
            },
        )
        write_json(
            blueprint_validation_path,
            {
                "run_id": run_id,
                "condition": condition,
                "candidate_index": candidate_index,
                "blueprint_status": blueprint_payload.get("blueprint_status", ""),
                **validation_data,
            },
        )

    row["blueprint_status"] = blueprint_payload.get("blueprint_status", "BLUEPRINT_FAILED")
    row["blueprint_attempts"] = _coerce_int(blueprint_payload.get("blueprint_attempts"), 0)
    row["blueprint_summary"] = _clean_text(blueprint_data.get("prompt_ready_summary"))
    row["blueprint_reused_for_surface_regenerate"] = "yes" if reuse_existing_blueprint and route_name == "surface_regenerate" else "no"
    row["reused_blueprint_source"] = "validated_blueprint" if reuse_existing_blueprint and route_name == "surface_regenerate" else ""
    row["condition_controls_json_compact"] = _compact_json(condition_controls)
    row["control_compliance_ok"] = "yes" if validation_data.get("control_compliance_ok") else "no"
    row["control_mismatch_summary"] = " | ".join(validation_data.get("control_mismatches", []))
    row["validated_condition_signal_summary"] = _compact_json(validation_data.get("condition_signals", {}))
    row["validated_blueprint_json_compact"] = _compact_json(validation_data.get("normalized_blueprint", blueprint_data))

    if not blueprint_payload.get("ok"):
        row["status"] = "BLUEPRINT_FAILED"
        row["error_message"] = " | ".join(validation_data.get("errors", [])) or _clean_text(blueprint_payload.get("error_message"))
        row["updated_at_epoch"] = int(time.time())
        write_json(metadata_path, row)
        upsert_manifest_row(run_id, row)
        upsert_screening_row(
            run_id,
            {
                "run_id": run_id,
                "candidate_index": candidate_index,
                "condition": condition,
                "output_video_path": str(output_video_path),
            },
        )
        return row

    row["status"] = "BLUEPRINT_READY"
    asset_pack = build_candidate_asset_pack(
        run_id=run_id,
        condition=condition,
        candidate_index=candidate_index,
        blueprint=validation_data.get("normalized_blueprint", blueprint_data),
        regeneration_route=route_name,
    )
    write_json(asset_pack_path, asset_pack)
    write_csv_rows(asset_manifest_path, asset_pack.get("assets", []), ASSET_MANIFEST_COLUMNS)

    storyboard_package = build_storyboard_panel_package(
        run_id=run_id,
        condition=condition,
        candidate_index=candidate_index,
        blueprint=validation_data.get("normalized_blueprint", blueprint_data),
        regeneration_route=route_name,
    )
    write_json(storyboard_plan_path, storyboard_package)

    execution_script = build_execution_script(
        run_id=run_id,
        condition=condition,
        candidate_index=candidate_index,
        blueprint=validation_data.get("normalized_blueprint", blueprint_data),
        regeneration_route=route_name,
    )
    write_json(execution_script_path, execution_script)
    execution_script_text_path.write_text(build_execution_script_text(execution_script), encoding="utf-8")

    final_generation_package = build_final_generation_package(
        config=config,
        blueprint=validation_data.get("normalized_blueprint", blueprint_data),
        asset_pack=asset_pack,
        storyboard_package=storyboard_package,
        execution_script=execution_script,
        revision_instruction_text=revision_instruction_text,
        edited_score_targets=edited_score_targets,
        regeneration_route=route_name,
        multimodal_options=normalized_multimodal_options,
        resolved_multimodal_inputs=resolve_multimodal_inputs(
            run_id=run_id,
            condition=condition,
            candidate_index=candidate_index,
            asset_pack=asset_pack,
            storyboard_package=storyboard_package,
            options=normalized_multimodal_options,
            config=config,
        ),
    )
    resolved_multimodal_inputs = final_generation_package.get("resolved_multimodal_inputs", {}) if isinstance(final_generation_package.get("resolved_multimodal_inputs"), dict) else {}
    selected_asset_types = set(resolved_multimodal_inputs.get("selected_asset_types", []) or [])
    selected_panel_ids = set(resolved_multimodal_inputs.get("selected_storyboard_panel_ids", []) or [])
    for asset in asset_pack.get("assets", []) if isinstance(asset_pack.get("assets"), list) else []:
        if isinstance(asset, dict):
            asset["in_use_for_generation"] = "yes" if str(asset.get("asset_type")) in selected_asset_types else "no"
    for panel in storyboard_package.get("panels", []) if isinstance(storyboard_package.get("panels"), list) else []:
        if isinstance(panel, dict):
            panel["in_use_for_generation"] = "yes" if str(panel.get("panel_id")) in selected_panel_ids else "no"
    write_json(asset_pack_path, asset_pack)
    write_json(storyboard_plan_path, storyboard_package)
    final_generation_package["asset_pack"] = asset_pack
    final_generation_package["storyboard_package"] = storyboard_package
    write_json(final_generation_package_path, final_generation_package)
    row["video_input_mode"] = final_generation_package.get("video_input_mode", normalized_multimodal_options.get("generation_mode", "text_only"))
    row["multimodal_generation_enabled"] = "yes" if final_generation_package.get("multimodal_generation_enabled") else "no"
    row["resolved_multimodal_inputs_json_compact"] = _compact_json(resolved_multimodal_inputs)
    row["first_frame_image_path_used"] = _clean_text(resolved_multimodal_inputs.get("first_frame_image_path"))
    row["last_frame_image_path_used"] = _clean_text(resolved_multimodal_inputs.get("last_frame_image_path"))
    row["reference_image_paths_used_json_compact"] = _compact_json(resolved_multimodal_inputs.get("reference_image_paths", []))
    row["storyboard_panel_used_for_generation"] = ", ".join(resolved_multimodal_inputs.get("selected_storyboard_panel_ids", []) or [])
    row["asset_types_used_for_generation"] = ", ".join(resolved_multimodal_inputs.get("selected_asset_types", []) or [])
    row["multimodal_input_summary"] = _clean_text(final_generation_package.get("multimodal_input_summary"))
    final_generation_prompt = compose_video_prompt_from_blueprint(
        config=config,
        blueprint=validation_data.get("normalized_blueprint", blueprint_data),
        revision_instruction_text=revision_instruction_text,
        edited_score_targets=edited_score_targets,
        regeneration_route=route_name,
        asset_pack=asset_pack,
        storyboard_package=storyboard_package,
        execution_script=execution_script,
        final_generation_package=final_generation_package,
    )
    row["final_generation_prompt"] = final_generation_prompt

    try:
        multimodal_result = generate_video_from_multimodal_package(
            model=model,
            prompt=final_generation_prompt,
            resolved_multimodal_inputs=resolved_multimodal_inputs,
            output_path=output_video_path,
        )
        row["video_input_mode"] = multimodal_result.get("effective_mode", row["video_input_mode"])
        row["status"] = "VIDEO_READY"
    except Exception as exc:
        row["status"] = "VIDEO_FAILED"
        row["error_message"] = str(exc)

    row["updated_at_epoch"] = int(time.time())
    write_json(metadata_path, row)
    upsert_manifest_row(run_id, row)
    upsert_screening_row(
        run_id,
        {
            "run_id": run_id,
            "candidate_index": candidate_index,
            "condition": condition,
            "output_video_path": str(output_video_path),
        },
    )

    if row["status"] == "VIDEO_READY" and should_auto_analyze:
        analysis_payload = analyze_video_candidate(
            run_id=run_id,
            condition=condition,
            candidate_index=candidate_index,
            analysis_model=str(config.get("analysis_model") or DEFAULT_ANALYSIS_MODEL),
        )
        if analysis_payload.get("analysis_status") == "ANALYSIS_READY":
            scoring = analysis_payload.get("analysis_result", {}).get("raise_scoring", {}) or {}
            upsert_screening_row(
                run_id,
                {
                    "run_id": run_id,
                    "candidate_index": candidate_index,
                    "condition": condition,
                    "output_video_path": str(output_video_path),
                    **{k: v for k, v in scoring.items() if k in RATING_COLUMNS},
                    "overall_keep_yes_no": analysis_payload.get("analysis_result", {}).get("keep_recommendation", ""),
                    "notes": analysis_payload.get("analysis_result", {}).get("summary", ""),
                },
            )
            row["analysis_status"] = "ANALYSIS_READY"
        else:
            row["analysis_status"] = analysis_payload.get("analysis_status", "ANALYSIS_FAILED")
        row["updated_at_epoch"] = int(time.time())
        write_json(metadata_path, row)
        upsert_manifest_row(run_id, row)

    return row


def generate_candidates_for_condition(
    run_id: str,
    condition: Literal["high_integration", "hard_sell"],
    n_candidates: int,
    auto_analyze: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    start_index = next_candidate_index(run_id, condition)
    for offset in range(n_candidates):
        results.append(generate_single_candidate(run_id, condition, start_index + offset, auto_analyze=auto_analyze))
    return results


def regenerate_candidate(
    run_id: str,
    condition: Literal["high_integration", "hard_sell"],
    candidate_index: int,
    revision_instruction_text: Optional[str] = None,
    edited_score_targets: Optional[Dict[str, Any]] = None,
    regeneration_route: Optional[str] = None,
    multimodal_options: Optional[Dict[str, Any]] = None,
    model_override: Optional[str] = None,
    auto_analyze: Optional[bool] = None,
) -> Dict[str, Any]:
    return generate_single_candidate(
        run_id=run_id,
        condition=condition,
        candidate_index=int(candidate_index),
        revision_instruction_text=revision_instruction_text,
        edited_score_targets=edited_score_targets,
        regeneration_route=regeneration_route,
        multimodal_options=multimodal_options,
        model_override=model_override,
        auto_analyze=auto_analyze,
    )


def list_runs() -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    for path in sorted(BASE_OUTPUT_DIR.iterdir() if BASE_OUTPUT_DIR.exists() else [], reverse=True):
        if not path.is_dir():
            continue
        run_id = path.name
        manifest_rows = load_manifest_rows(run_id)
        screening_rows = load_screening_rows(run_id)
        keep_yes = sum(1 for r in screening_rows if str(r.get("overall_keep_yes_no", "")).lower() == "yes")
        blueprint_failures = sum(1 for r in manifest_rows if str(r.get("blueprint_status", "")).upper() == "BLUEPRINT_FAILED")
        runs.append(
            {
                "run_id": run_id,
                "n_manifest": len(manifest_rows),
                "n_screening": len(screening_rows),
                "n_keep_yes": keep_yes,
                "n_blueprint_failed": blueprint_failures,
                "config_exists": get_run_config_path(run_id).exists(),
            }
        )
    return runs


def get_run_table(run_id: str) -> List[Dict[str, Any]]:
    manifest = load_manifest_rows(run_id)
    screening = {(str(r.get("condition")), _coerce_int(r.get("candidate_index", 0))): remap_score_keys(r) for r in load_screening_rows(run_id)}
    final_map = {(str(r.get("condition")), _coerce_int(r.get("candidate_index", 0))): r for r in load_final_selection_rows(run_id)}

    merged: List[Dict[str, Any]] = []
    for row in manifest:
        key = (str(row.get("condition")), _coerce_int(row.get("candidate_index", 0)))
        out = dict(row)
        out.update(screening.get(key, {}))
        out.update(final_map.get(key, {}))
        analysis = load_analysis(run_id, str(row.get("condition", "")), _coerce_int(row.get("candidate_index", 0)))
        blueprint = safe_read_json(Path(str(out.get("blueprint_json_path", "")))) if str(out.get("blueprint_json_path", "")) else {}
        blueprint_validation = (
            safe_read_json(Path(str(out.get("blueprint_validation_json_path", ""))))
            if str(out.get("blueprint_validation_json_path", ""))
            else {}
        )
        out["analysis_available"] = bool(analysis)
        out["analysis_status"] = analysis.get("analysis_status", out.get("analysis_status", ""))
        result = analysis.get("analysis_result", {}) if analysis else {}
        scoring = remap_score_keys(result.get("raise_scoring", {})) if isinstance(result.get("raise_scoring"), dict) else {}
        out["analysis_summary"] = result.get("summary", "") if analysis else ""
        out["analysis_keep_recommendation"] = result.get("keep_recommendation", "") if analysis else ""
        for k, v in scoring.items():
            out[f"analysis_{k}"] = v
        if blueprint:
            out["blueprint_summary"] = out.get("blueprint_summary") or blueprint.get("blueprint", {}).get("prompt_ready_summary", "")
        if blueprint_validation:
            out["blueprint_validation_errors"] = blueprint_validation.get("errors", [])
            out["blueprint_validation_warnings"] = blueprint_validation.get("warnings", [])
            out["control_mismatches"] = blueprint_validation.get("control_mismatches", [])
            out["control_compliance_ok"] = blueprint_validation.get("control_compliance_ok", out.get("control_compliance_ok", ""))
        merged.append(out)

    merged.sort(key=lambda x: (str(x.get("condition", "")), _coerce_int(x.get("candidate_index", 0))))
    return merged


def export_selected_manifest(run_id: str) -> Path:
    rows = [r for r in get_run_table(run_id) if str(r.get("final_selected_yes_no", "")).lower() == "yes"]
    output_path = get_run_dir(run_id) / "final_selected_export.csv"
    fieldnames = sorted({k for row in rows for k in row.keys()}) if rows else FINAL_SELECTION_COLUMNS
    write_csv_rows(output_path, rows, fieldnames)
    return output_path


if __name__ == "__main__":
    config = save_run_config(
        {
            "product_name": "LumaSkin",
            "product_category": "skincare serum",
            "influencer_profile": "female lifestyle creator in her 20s, casual, relatable, warm, energetic",
            "scene_context": "she is rushing to get ready in the morning before leaving for the airport",
            "n_candidates_high_integration": 2,
            "n_candidates_hard_sell": 2,
            "auto_analyze_after_generation": True,
        }
    )
    print(f"Created run: {config['run_id']}")
    print("Use the Streamlit app to generate, analyze, screen, and select stimuli.")
