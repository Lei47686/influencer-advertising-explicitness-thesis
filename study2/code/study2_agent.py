from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from study2_code import (
    BASE_OUTPUT_DIR,
    DEFAULT_ANALYSIS_MODEL,
    DEFAULT_BLUEPRINT_MODEL,
    DEFAULT_MODEL,
    analyze_video_candidate,
    build_run_id,
    export_selected_manifest,
    generate_candidates_for_condition,
    get_run_dir,
    get_run_table,
    list_runs,
    load_run_config,
    regenerate_candidate,
    save_run_config,
    upsert_final_selection_row,
    upsert_screening_row,
    validate_condition_controls,
)


ConditionName = Literal["high_integration", "hard_sell"]


class Study2Agent:
    """
    Controller for Study 2 stimulus development and curation.

    Role:
    - validate experiment-level run settings
    - create/update a run config
    - generate candidate pools for both conditions
    - trigger machine analysis for a selected candidate
    - save screening/final-selection decisions
    - export the final stimulus manifest

    It intentionally preserves the existing generation and analysis
    logic implemented in study2_code.py.
    """

    REQUIRED_CONFIG_FIELDS = [
        "product_name",
        "product_category",
        "influencer_profile",
        "scene_context",
    ]

    def __init__(self, results_base_dir: str | Path = BASE_OUTPUT_DIR):
        self.results_base_dir = Path(results_base_dir).expanduser().resolve()
        self.results_base_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _clean_text(value: Any) -> str:
        return "" if value is None else str(value).strip()

    @staticmethod
    def _normalize_condition(condition: str) -> str:
        value = str(condition).strip().lower()
        if value not in {"high_integration", "hard_sell"}:
            raise ValueError("condition must be 'high_integration' or 'hard_sell'")
        return value

    def validate_run_config(self, run_config: Dict[str, Any]) -> Dict[str, Any]:
        missing_fields = [
            field
            for field in self.REQUIRED_CONFIG_FIELDS
            if not self._clean_text(run_config.get(field))
        ]

        n_hi = int(run_config.get("n_candidates_high_integration", 0) or 0)
        n_hs = int(run_config.get("n_candidates_hard_sell", 0) or 0)
        hi_controls_validation = validate_condition_controls(run_config.get("high_integration_controls"), "high_integration", run_config)
        hs_controls_validation = validate_condition_controls(run_config.get("hard_sell_controls"), "hard_sell", run_config)

        if missing_fields:
            return {
                "ok": False,
                "error_type": "missing_required_fields",
                "message": f"Run config is missing required fields: {missing_fields}",
                "missing_fields": missing_fields,
            }

        if n_hi < 0 or n_hs < 0:
            return {
                "ok": False,
                "error_type": "invalid_candidate_counts",
                "message": "Candidate counts must be zero or positive integers.",
                "missing_fields": [],
            }

        if n_hi == 0 and n_hs == 0:
            return {
                "ok": False,
                "error_type": "empty_candidate_pool",
                "message": "At least one condition must request at least one candidate.",
                "missing_fields": [],
            }

        control_errors = hi_controls_validation.get("errors", []) + hs_controls_validation.get("errors", [])
        if control_errors:
            return {
                "ok": False,
                "error_type": "invalid_condition_controls",
                "message": f"Condition controls are invalid: {control_errors}",
                "missing_fields": [],
            }

        return {
            "ok": True,
            "error_type": "",
            "message": "",
            "missing_fields": [],
        }

    def create_or_update_run(self, run_config: Dict[str, Any]) -> Dict[str, Any]:
        validation = self.validate_run_config(run_config)
        if not validation["ok"]:
            return {
                "ok": False,
                "agent_stage": "validation",
                "agent_message": validation["message"],
                "validation": validation,
                "run_config": None,
            }

        payload = dict(run_config)
        payload.setdefault("run_id", build_run_id("study2"))
        payload.setdefault("model", DEFAULT_MODEL)
        payload.setdefault("analysis_model", DEFAULT_ANALYSIS_MODEL)
        payload.setdefault("blueprint_model", DEFAULT_BLUEPRINT_MODEL)

        saved = save_run_config(payload)
        return {
            "ok": True,
            "agent_stage": "run_config_saved",
            "agent_message": "Study 2 run configuration saved.",
            "validation": validation,
            "run_config": saved,
            "run_dir": str(get_run_dir(saved["run_id"])),
        }

    def generate_candidate_pool(
        self,
        *,
        run_id: str,
        auto_analyze: Optional[bool] = None,
    ) -> Dict[str, Any]:
        run_config = load_run_config(run_id)
        validation = self.validate_run_config(run_config)
        if not validation["ok"]:
            return {
                "ok": False,
                "agent_stage": "validation",
                "agent_message": validation["message"],
                "validation": validation,
                "generated_rows": [],
                "run_summary": {},
            }

        n_hi = int(run_config.get("n_candidates_high_integration", 0) or 0)
        n_hs = int(run_config.get("n_candidates_hard_sell", 0) or 0)

        generated_rows: List[Dict[str, Any]] = []
        if n_hi > 0:
            generated_rows.extend(
                generate_candidates_for_condition(
                    run_id=run_id,
                    condition="high_integration",
                    n_candidates=n_hi,
                    auto_analyze=auto_analyze,
                )
            )
        if n_hs > 0:
            generated_rows.extend(
                generate_candidates_for_condition(
                    run_id=run_id,
                    condition="hard_sell",
                    n_candidates=n_hs,
                    auto_analyze=auto_analyze,
                )
            )

        return {
            "ok": True,
            "agent_stage": "candidate_pool_generated",
            "agent_message": "Study 2 candidate pool generated.",
            "validation": validation,
            "generated_rows": generated_rows,
            "run_summary": self.get_run_summary(run_id),
        }

    def regenerate_one_candidate(
        self,
        *,
        run_id: str,
        condition: ConditionName,
        candidate_index: int,
        revision_instruction_text: Optional[str] = None,
        edited_score_targets: Optional[Dict[str, Any]] = None,
        regeneration_route: Optional[str] = None,
        multimodal_options: Optional[Dict[str, Any]] = None,
        model_override: Optional[str] = None,
        auto_analyze: Optional[bool] = None,
    ) -> Dict[str, Any]:
        result = regenerate_candidate(
            run_id=run_id,
            condition=self._normalize_condition(condition),
            candidate_index=int(candidate_index),
            revision_instruction_text=revision_instruction_text,
            edited_score_targets=edited_score_targets,
            regeneration_route=regeneration_route,
            multimodal_options=multimodal_options,
            model_override=model_override,
            auto_analyze=auto_analyze,
        )
        return {
            "ok": True,
            "agent_stage": "candidate_regenerated",
            "agent_message": "Candidate regenerated.",
            "candidate": result,
        }

    def analyze_candidate(
        self,
        *,
        run_id: str,
        condition: ConditionName,
        candidate_index: int,
        analysis_model: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload = analyze_video_candidate(
            run_id=run_id,
            condition=self._normalize_condition(condition),
            candidate_index=int(candidate_index),
            analysis_model=analysis_model,
        )
        return {
            "ok": payload.get("analysis_status") == "ANALYSIS_READY",
            "agent_stage": "candidate_analyzed",
            "agent_message": "Candidate analysis finished.",
            "analysis": payload,
        }

    def save_screening_decision(
        self,
        *,
        run_id: str,
        condition: ConditionName,
        candidate_index: int,
        output_video_path: str,
        ratings: Optional[Dict[str, Any]] = None,
        overall_keep_yes_no: str = "",
        notes: str = "",
    ) -> Dict[str, Any]:
        payload = {
            "run_id": run_id,
            "condition": self._normalize_condition(condition),
            "candidate_index": int(candidate_index),
            "output_video_path": output_video_path,
            "overall_keep_yes_no": self._clean_text(overall_keep_yes_no).lower(),
            "notes": notes,
        }
        if ratings:
            payload.update(ratings)
        upsert_screening_row(run_id, payload)
        return {
            "ok": True,
            "agent_stage": "screening_saved",
            "agent_message": "Screening decision saved.",
            "screening_row": payload,
        }

    def save_final_selection(
        self,
        *,
        run_id: str,
        condition: ConditionName,
        candidate_index: int,
        output_video_path: str,
        final_selected_yes_no: str,
        experiment_label: str = "",
        selection_notes: str = "",
    ) -> Dict[str, Any]:
        payload = {
            "run_id": run_id,
            "condition": self._normalize_condition(condition),
            "candidate_index": int(candidate_index),
            "output_video_path": output_video_path,
            "final_selected_yes_no": self._clean_text(final_selected_yes_no).lower(),
            "experiment_label": experiment_label,
            "selection_notes": selection_notes,
        }
        upsert_final_selection_row(run_id, payload)
        return {
            "ok": True,
            "agent_stage": "final_selection_saved",
            "agent_message": "Final selection saved.",
            "final_selection_row": payload,
        }

    def export_final_manifest(self, *, run_id: str) -> Dict[str, Any]:
        export_path = export_selected_manifest(run_id)
        return {
            "ok": True,
            "agent_stage": "manifest_exported",
            "agent_message": "Final selected stimulus manifest exported.",
            "export_path": str(export_path),
        }

    def get_run_summary(self, run_id: str) -> Dict[str, Any]:
        rows = get_run_table(run_id)
        total = len(rows)
        hi_rows = [r for r in rows if str(r.get("condition")) == "high_integration"]
        hs_rows = [r for r in rows if str(r.get("condition")) == "hard_sell"]
        video_ready = sum(1 for r in rows if str(r.get("status")) == "VIDEO_READY")
        blueprint_ready = sum(1 for r in rows if str(r.get("blueprint_status")) == "BLUEPRINT_READY")
        blueprint_failed = sum(1 for r in rows if str(r.get("blueprint_status")) == "BLUEPRINT_FAILED")
        analysis_ready = sum(1 for r in rows if str(r.get("analysis_status")) == "ANALYSIS_READY")
        keep_yes = sum(1 for r in rows if str(r.get("overall_keep_yes_no", "")).lower() == "yes")
        final_yes = sum(1 for r in rows if str(r.get("final_selected_yes_no", "")).lower() == "yes")

        return {
            "run_id": run_id,
            "run_dir": str(get_run_dir(run_id)),
            "total_candidates": total,
            "high_integration_candidates": len(hi_rows),
            "hard_sell_candidates": len(hs_rows),
            "blueprint_ready_count": blueprint_ready,
            "blueprint_failed_count": blueprint_failed,
            "video_ready_count": video_ready,
            "analysis_ready_count": analysis_ready,
            "keep_yes_count": keep_yes,
            "final_selected_count": final_yes,
        }

    def list_runs(self) -> List[Dict[str, Any]]:
        return list_runs()
