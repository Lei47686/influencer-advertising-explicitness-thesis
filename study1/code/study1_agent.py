from __future__ import annotations

import json
import math
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from study1_code import (
    analyze_single_video_with_metadata,
    batch_analyze_from_csv,
    flatten_study1_record,
    get_batch_required_columns,
)


class Study1Agent:
    """
    Minimal Study 1 agent controller.

    Role:
    - validate single-run inputs coming from the app
    - invoke the existing AES pipeline in study1_code.py
    - save standardized outputs
    - return one unified result payload to the UI

    It intentionally does NOT change the AES computation logic.
    """

    def __init__(self, results_base_dir: str | Path):
        self.results_base_dir = Path(results_base_dir).expanduser().resolve()
        self.results_base_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def build_run_id(prefix: str = "ui") -> str:
        return f"{prefix}_{uuid.uuid4().hex[:12]}"

    @staticmethod
    def _file_exists(path_str: str) -> bool:
        return bool(path_str) and Path(path_str).exists()

    @staticmethod
    def _make_json_safe(obj: Any):
        if isinstance(obj, dict):
            return {str(k): Study1Agent._make_json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [Study1Agent._make_json_safe(v) for v in obj]
        if isinstance(obj, tuple):
            return [Study1Agent._make_json_safe(v) for v in obj]
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        return obj

    @staticmethod
    def _is_success_status(status: Any) -> bool:
        return str(status).strip() in {"AES_READY", "PRECHECK_WARNING"}

    @staticmethod
    def _classify_aes_band(aes_score: Any) -> str:
        try:
            score = float(aes_score)
        except Exception:
            return "unknown"
        if score >= 0.55:
            return "high_explicitness"
        if score <= 0.40:
            return "low_explicitness"
        return "medium_explicitness"

    @staticmethod
    def _value_counts_dict(series: pd.Series) -> Dict[str, int]:
        if series.empty:
            return {}
        cleaned = series.fillna("").astype(str).str.strip()
        cleaned = cleaned[cleaned != ""]
        if cleaned.empty:
            return {}
        counts = cleaned.value_counts(dropna=False)
        return {str(k): int(v) for k, v in counts.items()}

    @staticmethod
    def _normalize_post_date_series(series: pd.Series) -> pd.Series:
        parsed = pd.to_datetime(series, errors="coerce")
        return parsed.dt.strftime("%Y-%m-%d").fillna("")

    @staticmethod
    def _safe_numeric_series(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
        if column not in df.columns:
            return pd.Series([default] * len(df), index=df.index, dtype=float)
        return pd.to_numeric(df[column], errors="coerce").fillna(default)

    @staticmethod
    def _safe_text_series(df: pd.DataFrame, column: str, default: str = "") -> pd.Series:
        if column not in df.columns:
            return pd.Series([default] * len(df), index=df.index, dtype=object)
        return df[column].fillna(default).astype(str).str.strip()

    @staticmethod
    def _safe_bool_series(df: pd.DataFrame, column: str) -> pd.Series:
        if column not in df.columns:
            return pd.Series([False] * len(df), index=df.index, dtype=bool)
        series = df[column]
        if pd.api.types.is_bool_dtype(series):
            return series.fillna(False)
        text = series.fillna("").astype(str).str.strip().str.lower()
        return text.isin({"true", "1", "yes"})

    # Summary builder for research-level batch diagnostics.
    # It derives interpretable batch statistics from the already written batch_results.csv
    # without re-running any AES computation.
    def _build_batch_summary(
        self,
        *,
        out_df: pd.DataFrame,
        failed_df: pd.DataFrame,
        batch_output_dir: Path,
        output_csv_path: Path,
        failed_rows_csv_path: Path,
    ) -> Dict[str, Any]:
        total_rows = int(len(out_df))
        failed_rows = int(len(failed_df))
        success_rows = int(total_rows - failed_rows)
        success_rate = round((success_rows / total_rows), 4) if total_rows > 0 else 0.0

        if out_df.empty:
            success_df = pd.DataFrame()
        else:
            status_series = out_df["status"] if "status" in out_df.columns else pd.Series([""] * len(out_df))
            success_df = out_df.loc[status_series.apply(self._is_success_status)].copy()

        if "aes_score" in success_df.columns and not success_df.empty:
            aes_series = pd.to_numeric(success_df["aes_score"], errors="coerce").dropna()
        else:
            aes_series = pd.Series(dtype=float)

        if not aes_series.empty:
            aes_band_series = aes_series.apply(self._classify_aes_band)
            mean_aes = round(float(aes_series.mean()), 6)
            median_aes = round(float(aes_series.median()), 6)
            min_aes = round(float(aes_series.min()), 6)
            max_aes = round(float(aes_series.max()), 6)
            aes_band_counts = {
                "high_explicitness": int((aes_band_series == "high_explicitness").sum()),
                "medium_explicitness": int((aes_band_series == "medium_explicitness").sum()),
                "low_explicitness": int((aes_band_series == "low_explicitness").sum()),
            }
        else:
            mean_aes = None
            median_aes = None
            min_aes = None
            max_aes = None
            aes_band_counts = {
                "high_explicitness": 0,
                "medium_explicitness": 0,
                "low_explicitness": 0,
            }

        video_mode_counts = self._value_counts_dict(success_df["video_mode"]) if "video_mode" in success_df.columns else {}
        transcript_quality_counts = (
            self._value_counts_dict(success_df["transcript_quality"])
            if "transcript_quality" in success_df.columns
            else {}
        )

        product_focused_review_count = 0
        if "is_product_focused_review" in success_df.columns and not success_df.empty:
            product_focused_review_count = int(
                pd.Series(success_df["is_product_focused_review"]).fillna(False).astype(bool).sum()
            )

        local_promo_segment_count = 0
        if "has_local_promo_segment" in success_df.columns and not success_df.empty:
            local_promo_segment_count = int(
                pd.Series(success_df["has_local_promo_segment"]).fillna(False).astype(bool).sum()
            )

        if failed_df.empty:
            failure_status_counts = {}
            failure_reason_counts = {}
            top_failure_examples = []
        else:
            failure_status_counts = (
                self._value_counts_dict(failed_df["status"])
                if "status" in failed_df.columns
                else {}
            )

            def _failure_reason(row: pd.Series) -> str:
                status = str(row.get("status", "") or "").strip()
                error_stage = str(row.get("error_stage", "") or "").strip()
                if status and status != "PIPELINE_EXCEPTION":
                    return status
                if status and error_stage:
                    return f"{status} | {error_stage}"
                if status:
                    return status
                if error_stage:
                    return error_stage
                return "UNKNOWN_FAILURE"

            reason_series = failed_df.apply(_failure_reason, axis=1)
            failure_reason_counts = self._value_counts_dict(reason_series)

            top_cols = [c for c in ["video_id", "status", "error_stage", "error_message"] if c in failed_df.columns]
            top_failure_examples = failed_df[top_cols].head(5).to_dict(orient="records")

        return {
            "total_rows": total_rows,
            "success_rows": success_rows,
            "failed_rows": failed_rows,
            "success_rate": success_rate,
            "batch_output_dir": str(batch_output_dir),
            "output_csv_path": str(output_csv_path),
            "failed_rows_csv_path": str(failed_rows_csv_path),
            "video_mode_counts": video_mode_counts,
            "transcript_quality_counts": transcript_quality_counts,
            "mean_aes": mean_aes,
            "median_aes": median_aes,
            "min_aes": min_aes,
            "max_aes": max_aes,
            "aes_band_counts": aes_band_counts,
            "product_focused_review_count": product_focused_review_count,
            "local_promo_segment_count": local_promo_segment_count,
            "failure_reason_counts": failure_reason_counts,
            "failure_status_counts": failure_status_counts,
            "top_failure_examples": self._make_json_safe(top_failure_examples),
        }

    # Modeling dataset builder for formal research analysis.
    def _build_video_modeling_dataset(self, out_df: pd.DataFrame) -> pd.DataFrame:
        if out_df.empty:
            return pd.DataFrame(
                columns=[
                    "video_id", "product_id", "brand_id", "post_date", "creator_id", "aes_score", "aes_reason",
                    "likes", "comments", "shares", "views", "engagement_rate", "video_mode",
                    "transcript_quality", "video_duration_seconds", "frame_count",
                    "is_product_focused_review", "has_local_promo_segment", "product_category",
                    "product_price", "price_tier", "hedonic_utilitarian", "status", "error_stage", "error_message",
                ]
            )

        status_series = self._safe_text_series(out_df, "status")
        success_df = out_df.loc[status_series.apply(self._is_success_status)].copy()
        if success_df.empty:
            return pd.DataFrame()

        success_df["post_date"] = self._normalize_post_date_series(success_df["post_date"] if "post_date" in success_df.columns else pd.Series(dtype=object))

        keep_cols = [
            "video_id",
            "product_id",
            "brand_id",
            "post_date",
            "creator_id",
            "aes_score",
            "aes_reason",
            "likes",
            "comments",
            "shares",
            "views",
            "engagement_rate",
            "video_mode",
            "transcript_quality",
            "video_duration_seconds",
            "frame_count",
            "is_product_focused_review",
            "has_local_promo_segment",
            "product_category",
            "product_price",
            "price_tier",
            "hedonic_utilitarian",
            "status",
            "error_stage",
            "error_message",
        ]
        for col in keep_cols:
            if col not in success_df.columns:
                success_df[col] = ""
        return success_df[keep_cols].copy()

    def _build_panel_from_group(
        self,
        df: pd.DataFrame,
        group_keys: list[str],
        carry_fields: list[str],
    ) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()

        work = df.copy()
        work["aes_score"] = self._safe_numeric_series(work, "aes_score")
        work["engagement_rate"] = self._safe_numeric_series(work, "engagement_rate")
        work["likes"] = self._safe_numeric_series(work, "likes")
        work["comments"] = self._safe_numeric_series(work, "comments")
        work["shares"] = self._safe_numeric_series(work, "shares")
        work["views"] = self._safe_numeric_series(work, "views")
        work["video_duration_seconds"] = self._safe_numeric_series(work, "video_duration_seconds")
        work["is_product_focused_review"] = self._safe_bool_series(work, "is_product_focused_review").astype(int)
        work["has_local_promo_segment"] = self._safe_bool_series(work, "has_local_promo_segment").astype(int)
        work["video_mode"] = self._safe_text_series(work, "video_mode")
        work["aes_band"] = work["aes_score"].apply(self._classify_aes_band)

        groupby_obj = work.groupby(group_keys, dropna=False)
        rows = []
        for keys, g in groupby_obj:
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = {k: v for k, v in zip(group_keys, keys)}
            row["n_videos"] = int(len(g))
            row["mean_aes"] = round(float(g["aes_score"].mean()), 6) if len(g) else None
            row["median_aes"] = round(float(g["aes_score"].median()), 6) if len(g) else None
            row["std_aes"] = round(float(g["aes_score"].std(ddof=0)), 6) if len(g) else 0.0
            row["share_low_aes"] = round(float((g["aes_band"] == "low_explicitness").mean()), 6)
            row["share_medium_aes"] = round(float((g["aes_band"] == "medium_explicitness").mean()), 6)
            row["share_high_aes"] = round(float((g["aes_band"] == "high_explicitness").mean()), 6)
            row["mean_engagement_rate"] = round(float(g["engagement_rate"].mean()), 6) if len(g) else None
            row["median_engagement_rate"] = round(float(g["engagement_rate"].median()), 6) if len(g) else None
            row["total_likes"] = int(g["likes"].sum())
            row["total_comments"] = int(g["comments"].sum())
            row["total_shares"] = int(g["shares"].sum())
            row["total_views"] = int(g["views"].sum())
            total_engagement = float(g["likes"].sum() + g["comments"].sum() + g["shares"].sum())
            row["log_total_engagement"] = round(math.log(total_engagement + 1.0), 6)
            row["share_visual_led"] = round(float((g["video_mode"] == "visual_led").mean()), 6)
            row["share_speech_rich"] = round(float((g["video_mode"] == "speech_rich").mean()), 6)
            row["share_speech_poor"] = round(float((g["video_mode"] == "speech_poor").mean()), 6)
            row["share_product_focused_review"] = round(float(g["is_product_focused_review"].mean()), 6)
            row["share_local_promo_segment"] = round(float(g["has_local_promo_segment"].mean()), 6)
            row["mean_video_duration_seconds"] = round(float(g["video_duration_seconds"].mean()), 6) if len(g) else None
            for field in carry_fields:
                row[field] = g[field].dropna().iloc[0] if field in g.columns and not g[field].dropna().empty else ""
            rows.append(row)

        return pd.DataFrame(rows)

    # Panel-ready dataset builders.
    def _build_product_day_panel(self, video_df: pd.DataFrame) -> pd.DataFrame:
        return self._build_panel_from_group(
            video_df,
            group_keys=["product_id", "post_date"],
            carry_fields=["brand_id", "product_category", "product_price", "price_tier", "hedonic_utilitarian"],
        )

    def _build_brand_day_panel(self, video_df: pd.DataFrame) -> pd.DataFrame:
        return self._build_panel_from_group(
            video_df,
            group_keys=["brand_id", "post_date"],
            carry_fields=[],
        )

    # Optional formal panel regression module.
    def run_panel_analysis(
        self,
        *,
        panel_unit: str = "video",
        outcome: str = "engagement_rate",
        fe_level: str | None = "product",
        dataset_path: str,
        output_dir: str | Path,
    ) -> Dict[str, Any]:
        try:
            import statsmodels.formula.api as smf
        except Exception as exc:
            return {
                "ok": False,
                "message": f"statsmodels is not available: {exc}",
                "panel_regression_summary_path": None,
                "panel_regression_coefficients_path": None,
            }

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / "panel_regression_summary.json"
        coef_path = output_dir / "panel_regression_coefficients.csv"

        if panel_unit != "video":
            return {
                "ok": False,
                "message": "Only video-level regression is supported in the current version.",
                "panel_regression_summary_path": None,
                "panel_regression_coefficients_path": None,
            }

        df = pd.read_csv(dataset_path)
        required = {"aes_score", outcome, "video_duration_seconds", "video_mode", "transcript_quality", "is_product_focused_review", "has_local_promo_segment", "post_date"}
        if fe_level == "product":
            required.add("product_id")
        if fe_level == "brand":
            required.add("brand_id")
        missing = [c for c in required if c not in df.columns]
        if missing:
            return {
                "ok": False,
                "message": f"Panel regression is missing required columns: {missing}",
                "panel_regression_summary_path": None,
                "panel_regression_coefficients_path": None,
            }

        model_df = df.copy()
        model_df = model_df.dropna(subset=["aes_score", outcome])
        model_df["post_date"] = self._normalize_post_date_series(model_df["post_date"])
        if (model_df["post_date"] == "").all():
            return {
                "ok": False,
                "message": "Panel regression requires valid post_date values.",
                "panel_regression_summary_path": None,
                "panel_regression_coefficients_path": None,
            }

        fe_col = None
        if fe_level == "product":
            fe_col = "product_id"
        elif fe_level == "brand":
            fe_col = "brand_id"

        if fe_col is not None:
            non_empty_fe = model_df[fe_col].fillna("").astype(str).str.strip()
            if (non_empty_fe == "").any():
                return {
                    "ok": False,
                    "message": f"Panel regression requires non-empty {fe_col} values.",
                    "panel_regression_summary_path": None,
                    "panel_regression_coefficients_path": None,
                }
            if non_empty_fe.nunique() < 2:
                return {
                    "ok": False,
                    "message": f"Panel regression requires at least two {fe_col} groups.",
                    "panel_regression_summary_path": None,
                    "panel_regression_coefficients_path": None,
                }

        formula = (
            f"{outcome} ~ aes_score + video_duration_seconds + "
            "C(video_mode) + C(transcript_quality) + "
            "is_product_focused_review + has_local_promo_segment + "
            "C(post_date)"
        )
        if fe_col is not None:
            formula += f" + C({fe_col})"

        model = smf.ols(formula=formula, data=model_df).fit()
        coef_df = pd.DataFrame({
            "term": model.params.index,
            "coefficient": model.params.values,
            "std_error": model.bse.values,
            "t_value": model.tvalues.values,
            "p_value": model.pvalues.values,
        })
        coef_df.to_csv(coef_path, index=False, encoding="utf-8-sig")

        aes_row = coef_df.loc[coef_df["term"].astype(str) == "aes_score"].copy()
        if aes_row.empty:
            panel_regression_key_result = {
                "aes_coefficient": None,
                "aes_std_error": None,
                "aes_t_value": None,
                "aes_p_value": None,
                "r_squared": float(model.rsquared),
                "adj_r_squared": float(model.rsquared_adj),
                "n_obs": int(model.nobs),
            }
        else:
            first = aes_row.iloc[0]
            panel_regression_key_result = {
                "aes_coefficient": float(first["coefficient"]),
                "aes_std_error": float(first["std_error"]),
                "aes_t_value": float(first["t_value"]),
                "aes_p_value": float(first["p_value"]),
                "r_squared": float(model.rsquared),
                "adj_r_squared": float(model.rsquared_adj),
                "n_obs": int(model.nobs),
            }

        summary_payload = {
            "ok": True,
            "panel_unit": panel_unit,
            "outcome": outcome,
            "fe_level": fe_level,
            "n_obs": int(model.nobs),
            "r_squared": float(model.rsquared),
            "adj_r_squared": float(model.rsquared_adj),
            "formula": formula,
            "dataset_path": str(dataset_path),
            "panel_regression_key_result": panel_regression_key_result,
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(self._make_json_safe(summary_payload), f, ensure_ascii=False, indent=2)

        return {
            "ok": True,
            "message": "Panel regression completed.",
            "panel_regression_summary_path": str(summary_path),
            "panel_regression_coefficients_path": str(coef_path),
            "panel_regression_key_result": panel_regression_key_result,
        }

    def validate_single_inputs(
        self,
        *,
        video_path: str,
        product_info_path: str,
        product_image_path: str,
        metadata: Optional[dict],
    ) -> Dict[str, Any]:
        missing_files = [
            p
            for p in [video_path, product_info_path, product_image_path]
            if not self._file_exists(p)
        ]

        if metadata is None:
            return {
                "ok": False,
                "error_type": "missing_metadata",
                "message": "Metadata is not ready yet.",
                "missing_files": [],
            }

        if missing_files:
            return {
                "ok": False,
                "error_type": "missing_files",
                "message": "Some required files are missing or have not been uploaded yet.",
                "missing_files": missing_files,
            }

        return {
            "ok": True,
            "error_type": "",
            "message": "",
            "missing_files": [],
        }

    def save_analysis_outputs(self, result: dict) -> dict:
        run_id = result.get("run_id", "") or self.build_run_id("saved")
        output_dir = self.results_base_dir / run_id
        output_dir.mkdir(parents=True, exist_ok=True)

        result_json_path = output_dir / "result.json"
        flat_record_json_path = output_dir / "flat_record.json"
        flat_record_csv_path = output_dir / "flat_record.csv"

        safe_result = self._make_json_safe(result)
        flat_record = flatten_study1_record(result)
        safe_flat_record = self._make_json_safe(flat_record)

        with open(result_json_path, "w", encoding="utf-8") as f:
            json.dump(safe_result, f, ensure_ascii=False, indent=2)

        with open(flat_record_json_path, "w", encoding="utf-8") as f:
            json.dump(safe_flat_record, f, ensure_ascii=False, indent=2)

        pd.DataFrame([safe_flat_record]).to_csv(
            flat_record_csv_path,
            index=False,
            encoding="utf-8-sig",
        )

        return {
            "output_dir": str(output_dir),
            "result_json_path": str(result_json_path),
            "flat_record_json_path": str(flat_record_json_path),
            "flat_record_csv_path": str(flat_record_csv_path),
        }

    def validate_batch_csv(
        self,
        *,
        csv_path: str,
    ) -> Dict[str, Any]:
        csv_path_obj = Path(csv_path)
        if not csv_path_obj.exists():
            return {"ok": False, "message": f"Batch CSV not found: {csv_path}", "missing_columns": [], "row_count": 0}
        df = pd.read_csv(csv_path_obj)
        missing_columns = [c for c in get_batch_required_columns() if c not in df.columns]
        if missing_columns:
            return {"ok": False, "message": f"Batch CSV is missing required columns: {missing_columns}", "missing_columns": missing_columns, "row_count": int(len(df))}
        if df.empty:
            return {"ok": False, "message": "Batch CSV is empty.", "missing_columns": [], "row_count": 0}
        return {"ok": True, "message": "", "missing_columns": [], "row_count": int(len(df))}

    def run_single(
        self,
        *,
        video_path: str,
        product_info_path: str,
        product_image_path: str,
        metadata: dict,
        run_id: Optional[str] = None,
    ) -> dict:
        validation = self.validate_single_inputs(
            video_path=video_path,
            product_info_path=product_info_path,
            product_image_path=product_image_path,
            metadata=metadata,
        )

        if not validation["ok"]:
            return {
                "ok": False,
                "agent_stage": "validation",
                "agent_message": validation["message"],
                "validation": validation,
                "result": None,
                "saved_paths": None,
            }

        actual_run_id = run_id or self.build_run_id("ui")
        result = analyze_single_video_with_metadata(
            video_path=video_path,
            product_info_path=product_info_path,
            product_image_path=product_image_path,
            metadata=metadata,
            run_id=actual_run_id,
        )
        saved_paths = self.save_analysis_outputs(result)

        return {
            "ok": True,
            "agent_stage": "completed",
            "agent_message": "Analysis completed.",
            "validation": validation,
            "result": result,
            "saved_paths": saved_paths,
        }

    # Batch controller:
    # - validate CSV first
    # - create a dedicated batch output directory
    # - call the existing study1_code batch function
    # - export failed_rows.csv and return a unified batch response
    def run_batch(
        self,
        *,
        csv_path: str,
        run_id: Optional[str] = None,
    ) -> dict:
        validation = self.validate_batch_csv(csv_path=csv_path)

        if not validation["ok"]:
            return {
                "ok": False,
                "agent_stage": "validation",
                "agent_message": validation["message"],
                "validation": validation,
                "output_csv_path": None,
                "failed_rows_csv_path": None,
                "batch_summary": {
                    "total_rows": int(validation.get("row_count", 0) or 0),
                    "success_rows": 0,
                    "failed_rows": 0,
                    "batch_output_dir": "",
                },
            }

        try:
            actual_run_id = run_id or self.build_run_id("batch")
            batch_output_dir = self.results_base_dir / actual_run_id
            batch_output_dir.mkdir(parents=True, exist_ok=True)

            output_csv_path = batch_output_dir / "batch_results.csv"
            failed_rows_csv_path = batch_output_dir / "failed_rows.csv"
            batch_summary_json_path = batch_output_dir / "batch_summary.json"
            video_modeling_dataset_path = batch_output_dir / "video_modeling_dataset.csv"
            product_day_panel_path = batch_output_dir / "product_day_panel.csv"
            brand_day_panel_path = batch_output_dir / "brand_day_panel.csv"

            batch_analyze_from_csv(
                csv_path=csv_path,
                output_csv_path=str(output_csv_path),
            )

            if output_csv_path.exists():
                out_df = pd.read_csv(output_csv_path)
            else:
                out_df = pd.DataFrame()

            total_rows = int(len(out_df))
            if out_df.empty:
                failed_df = pd.DataFrame(
                    columns=["video_id", "video_path", "status", "error_stage", "error_message"]
                )
                success_rows = 0
                failed_rows = 0
            else:
                status_series = out_df["status"] if "status" in out_df.columns else pd.Series([""] * len(out_df))
                failed_mask = ~status_series.apply(self._is_success_status)
                failed_df = out_df.loc[failed_mask].copy()
                failed_rows = int(len(failed_df))
                success_rows = int(total_rows - failed_rows)

            failed_df.to_csv(
                failed_rows_csv_path,
                index=False,
                encoding="utf-8-sig",
            )

            # Build and persist research-level batch summary.
            batch_summary = self._build_batch_summary(
                out_df=out_df,
                failed_df=failed_df,
                batch_output_dir=batch_output_dir,
                output_csv_path=output_csv_path,
                failed_rows_csv_path=failed_rows_csv_path,
            )

            # Modeling / panel dataset exports.
            video_modeling_df = self._build_video_modeling_dataset(out_df)
            product_day_panel_df = self._build_product_day_panel(video_modeling_df)
            brand_day_panel_df = self._build_brand_day_panel(video_modeling_df)

            video_modeling_df.to_csv(video_modeling_dataset_path, index=False, encoding="utf-8-sig")
            product_day_panel_df.to_csv(product_day_panel_path, index=False, encoding="utf-8-sig")
            brand_day_panel_df.to_csv(brand_day_panel_path, index=False, encoding="utf-8-sig")

            batch_summary["video_modeling_dataset_path"] = str(video_modeling_dataset_path)
            batch_summary["product_day_panel_path"] = str(product_day_panel_path)
            batch_summary["brand_day_panel_path"] = str(brand_day_panel_path)

            panel_result = {
                "ok": False,
                "message": "Panel regression was not run.",
                "panel_regression_summary_path": None,
                "panel_regression_coefficients_path": None,
            }
            if not video_modeling_df.empty:
                panel_result = self.run_panel_analysis(
                    panel_unit="video",
                    outcome="engagement_rate",
                    fe_level="product" if "product_id" in video_modeling_df.columns else None,
                    dataset_path=str(video_modeling_dataset_path),
                    output_dir=batch_output_dir,
                )
            if panel_result.get("ok"):
                batch_summary["panel_regression_summary_path"] = panel_result.get("panel_regression_summary_path")
                batch_summary["panel_regression_coefficients_path"] = panel_result.get("panel_regression_coefficients_path")
                batch_summary["panel_regression_key_result"] = panel_result.get("panel_regression_key_result", {})

            with open(batch_summary_json_path, "w", encoding="utf-8") as f:
                json.dump(self._make_json_safe(batch_summary), f, ensure_ascii=False, indent=2)

            return {
                "ok": True,
                "agent_stage": "completed",
                "agent_message": "Batch analysis completed.",
                "validation": validation,
                "output_csv_path": str(output_csv_path),
                "failed_rows_csv_path": str(failed_rows_csv_path),
                "batch_summary_json_path": str(batch_summary_json_path),
                "video_modeling_dataset_path": str(video_modeling_dataset_path),
                "product_day_panel_path": str(product_day_panel_path),
                "brand_day_panel_path": str(brand_day_panel_path),
                "panel_regression_summary_path": panel_result.get("panel_regression_summary_path"),
                "panel_regression_coefficients_path": panel_result.get("panel_regression_coefficients_path"),
                "panel_regression_key_result": panel_result.get("panel_regression_key_result", {}),
                "batch_summary": batch_summary,
            }

        except Exception as exc:
            return {
                "ok": False,
                "agent_stage": "batch_execution_failed",
                "agent_message": f"Batch analysis failed: {exc}",
                "validation": validation,
                "output_csv_path": None,
                "failed_rows_csv_path": None,
                "batch_summary_json_path": None,
                "video_modeling_dataset_path": None,
                "product_day_panel_path": None,
                "brand_day_panel_path": None,
                "panel_regression_summary_path": None,
                "panel_regression_coefficients_path": None,
                "panel_regression_key_result": {},
                "batch_summary": {
                    "total_rows": int(validation.get("row_count", 0) or 0),
                    "success_rows": 0,
                    "failed_rows": 0,
                    "success_rate": 0.0,
                    "batch_output_dir": "",
                    "output_csv_path": "",
                    "failed_rows_csv_path": "",
                    "video_modeling_dataset_path": "",
                    "product_day_panel_path": "",
                    "brand_day_panel_path": "",
                    "video_mode_counts": {},
                    "transcript_quality_counts": {},
                    "mean_aes": None,
                    "median_aes": None,
                    "min_aes": None,
                    "max_aes": None,
                    "aes_band_counts": {
                        "high_explicitness": 0,
                        "medium_explicitness": 0,
                        "low_explicitness": 0,
                    },
                    "product_focused_review_count": 0,
                    "local_promo_segment_count": 0,
                    "failure_reason_counts": {},
                    "failure_status_counts": {},
                    "top_failure_examples": [],
                    "panel_regression_summary_path": "",
                    "panel_regression_coefficients_path": "",
                    "panel_regression_key_result": {},
                },
            }
