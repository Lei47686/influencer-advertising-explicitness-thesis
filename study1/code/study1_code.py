from __future__ import annotations

from typing import TypedDict, List, Dict, Any, Tuple
from pathlib import Path
import subprocess
import re
import shutil
import csv
import uuid
import hashlib
import tempfile

import numpy as np
from PIL import Image
import cv2

import torch
import torch.nn as nn
from torchvision import models

from sentence_transformers import SentenceTransformer
from faster_whisper import WhisperModel
from huggingface_hub import snapshot_download

from langgraph.graph import StateGraph, START, END


# =========================================================
# Config
# =========================================================

RUNS_BASE_DIR = Path(tempfile.gettempdir()) / "aes_v3_runs"

CONTROLLER_THRESHOLDS = {
    "low_coverage": 0.20,
    "medium_coverage": 0.50,
    "low_sentence_count": 2,
    "medium_sentence_count": 4,
    "low_text_unit_count": 12,
    "medium_text_unit_count": 30,
    "low_avg_text_units": 4.0,
    "medium_avg_text_units": 8.0,
}

AES_CONFIG = {
    "visual_score_weights": {
        "avg": 0.30,
        "top10": 0.70,
    },
    "semantic_score_weights": {
        "global": 0.35,
        "peak": 0.15,
        "top5": 0.50,
    },
    "fusion_weights": {
        "speech_rich": {"visual": 0.45, "semantic": 0.55},
        "speech_poor": {"visual": 0.60, "semantic": 0.40},
        "visual_led": {"visual": 0.85, "semantic": 0.15},
    },
    "thresholds": {
        "high_aes": 0.55,
        "low_aes": 0.40,
        "product_focused_semantic_global": 0.55,
        "product_focused_semantic_top5": 0.52,
        "product_focused_visual_top10": 0.50,
        "local_promo_high_sentence_ratio": 0.08,
        "local_promo_semantic_peak": 0.60,
        "local_promo_visual_top10": 0.32,
    },
}

WHISPER_MODEL_NAME = "small"
WHISPER_COMPUTE_TYPE = "int8"
TRANSCRIPTION_LANGUAGE = "zh"
SEMANTIC_SCORING_VERSION = "bge-zh-v1"
THRESHOLD_CALIBRATION_STATUS = "final_v3_frozen"
TEXT_MODEL_NAME = "BAAI/bge-base-zh-v1.5"

BATCH_REQUIRED_COLUMNS = [
    "video_id",
    "video_path",
    "product_info_path",
    "product_image_path",
    "likes",
    "comments",
    "shares",
    "views",
    "post_date",
    "creator_id",
    "product_id",
    "brand_id",
]

def get_batch_required_columns() -> list[str]:
    return list(BATCH_REQUIRED_COLUMNS)


# =========================================================
# State
# =========================================================

class AESState(TypedDict, total=False):
    # Input paths
    video_path: str
    product_info_path: str
    product_image_path: str

    # Run-level paths
    run_id: str
    run_dir: str
    frames_dir: str
    audio_dir: str

    # Loaded content
    product_description: str
    product_embedding_text: str
    transcript: str
    transcript_segments: List[Dict[str, Any]]
    sentences: List[Dict[str, Any]]

    # Status
    status: str
    status_reason: str
    is_commercial_candidate: bool
    error_stage: str
    error_message: str

    # Reproducibility metadata
    text_model_name: str
    transcription_language: str
    semantic_scoring_version: str
    threshold_calibration_status: str

    # Video / audio metadata
    video_duration_seconds: float
    frame_count: int
    audio_path: str
    audio_extraction_success: bool
    audio_error: str

    # Frame extraction
    extracted_frame_items: List[Dict[str, Any]]

    # Cached embeddings / features
    product_text_embedding: List[float]
    product_image_feature: List[float]
    product_text_embedding_cached: bool
    product_text_embedding_model: str
    product_image_feature_cached: bool

    # Visual features
    frame_similarity_items: List[Dict[str, Any]]
    frame_similarities: List[float]
    visual_similarity_score: float
    visual_peak_similarity: float
    high_visual_similarity_ratio: float
    visual_explicitness_score: float
    top_10_frame_mean: float

    # Semantic features
    global_semantic_similarity: float
    sentence_similarities: List[float]
    sentence_peak_similarity: float
    high_similarity_sentence_ratio: float
    semantic_explicitness_score: float
    top_5_sentence_mean: float

    # Controller
    transcript_token_count: int
    transcript_text_unit_count: int
    transcript_cjk_character_count: int
    effective_sentence_count: int
    avg_words_per_sentence: float
    avg_text_units_per_sentence: float
    transcript_coverage_ratio: float
    transcript_quality: str
    video_mode: str
    controller_reason: str

    # Metadata / engagement
    video_id: str
    likes: int
    comments: int
    shares: int
    views: int
    engagement_rate: float
    post_date: str
    creator_id: str

    # AES
    aes_components: Dict[str, Any]
    aes_score: float

    # Explanations
    visual_reason: str
    semantic_reason: str
    aes_reason: str


# =========================================================
# Utilities
# =========================================================

def read_text_file(path_str: str) -> str:
    path = Path(path_str)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    denom = float(np.linalg.norm(vec1) * np.linalg.norm(vec2))
    if denom == 0.0:
        return 0.0
    return float(np.dot(vec1, vec2) / denom)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def split_mixed_text_to_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    # Chinese transcripts normally do not contain whitespace after punctuation.
    # Split immediately after terminal punctuation while retaining it.
    parts = re.split(r"(?<=[。！？!?；;\.])\s*", text)
    cleaned = [p.strip() for p in parts if p.strip()]

    if len(cleaned) <= 1 and len(text) > 120:
        chunks = []
        step = 80
        start = 0
        while start < len(text):
            chunks.append(text[start:start + step].strip())
            start += step
        cleaned = [c for c in chunks if c]

    return cleaned


def count_cjk_characters(text: str) -> int:
    return len(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", text or ""))


def count_text_units(text: str) -> int:
    """Count language-neutral text units for Chinese and space-delimited text.

    Each CJK character counts as one unit. Consecutive Latin letters/numbers count
    as one token, so Chinese transcripts are not incorrectly treated as one word.
    """
    if not text:
        return 0
    cjk_count = count_cjk_characters(text)
    non_cjk_text = re.sub(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", " ", text)
    latin_numeric_tokens = re.findall(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*", non_cjk_text)
    return cjk_count + len(latin_numeric_tokens)


_PRODUCT_INFO_EXCLUDED_LABELS = (
    "来源", "链接", "网址", "核验", "确认日期", "抓取日期", "排除",
    "研究备注", "样本备注", "视频备注", "飞瓜", "数据来源",
)


def build_product_embedding_text(product_description: str) -> str:
    """Create neutral product reference text for semantic comparison.

    Research/source metadata and URLs are excluded, while product identity,
    features, aliases and ordinary descriptive prose are retained.
    """
    kept_lines: List[str] = []
    for raw_line in (product_description or "").splitlines():
        line = raw_line.strip().lstrip("-•* ")
        if not line:
            continue
        label = re.split(r"[:：]", line, maxsplit=1)[0].strip()
        if any(label.startswith(prefix) for prefix in _PRODUCT_INFO_EXCLUDED_LABELS):
            continue
        line = re.sub(r"https?://\S+|www\.\S+", "", line).strip(" ，,;；")
        if line:
            kept_lines.append(line)

    cleaned = "。".join(kept_lines)
    cleaned = re.sub(r"。+", "。", cleaned).strip("。 ")
    return cleaned or re.sub(r"https?://\S+|www\.\S+", "", product_description or "").strip()


def build_transcription_prompt(product_description: str) -> str:
    """Build a conservative ASR prompt from identity fields only.

    Functional claims are intentionally excluded to avoid steering Whisper into
    hallucinating promotional language that was not actually spoken.
    """
    identity_labels = (
        "品牌", "产品名称", "商品名称", "产品名", "商品名", "常见别名", "别名",
        "brand", "product name", "product", "aliases", "alias",
    )
    identity_items: List[str] = []
    for raw_line in (product_description or "").splitlines():
        line = raw_line.strip().lstrip("-•* ")
        if not line or not re.search(r"[:：]", line):
            continue
        label, value = re.split(r"[:：]", line, maxsplit=1)
        if label.strip().lower() in identity_labels and value.strip():
            identity_items.append(value.strip())

    base = "这是一段中文抖音商品相关视频。"
    if identity_items:
        return f"{base}可能出现的品牌或产品名称：{'、'.join(identity_items)}。"
    return base


def distribute_segment_time(segment_start: float, segment_end: float, chunks: List[str]) -> List[Dict[str, Any]]:
    if not chunks:
        return []

    total_duration = max(0.0, float(segment_end) - float(segment_start))
    if total_duration == 0.0:
        return [
            {"text": chunk, "start": float(segment_start), "end": float(segment_end)}
            for chunk in chunks
        ]

    unit = total_duration / len(chunks)
    items = []
    for i, chunk in enumerate(chunks):
        s = float(segment_start) + i * unit
        e = float(segment_start) + (i + 1) * unit
        items.append({
            "text": chunk,
            "start": round(s, 2),
            "end": round(e, 2),
        })
    return items


def compute_engagement_rate(likes: int, comments: int, shares: int, views: int) -> float:
    if not views or views <= 0:
        return 0.0
    return round((likes + comments + shares) / views, 6)


def parse_supplied_interaction_rate(value: Any) -> float | None:
    """Parse an externally supplied interaction rate without inventing views.

    Accepts either a decimal rate (0.0046) or a percentage string ("0.46%").
    Missing and non-finite values return None so callers can fall back to a
    views-based calculation when genuine view counts are available.
    """
    if value is None:
        return None
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "-", "--"}:
        return None
    is_percent = text.endswith("%")
    if is_percent:
        text = text[:-1].strip()
    try:
        rate = float(text)
    except (TypeError, ValueError):
        return None
    if is_percent:
        rate /= 100.0
    if not np.isfinite(rate) or rate < 0:
        return None
    return round(rate, 6)


def fetch_video_metadata_from_dict(metadata: dict) -> dict:
    return {
        "video_id": metadata.get("video_id", ""),
        "likes": int(metadata.get("likes", 0) or 0),
        "comments": int(metadata.get("comments", 0) or 0),
        "shares": int(metadata.get("shares", 0) or 0),
        "views": int(metadata.get("views", 0) or 0),
        "interaction_rate": parse_supplied_interaction_rate(metadata.get("interaction_rate")),
        "post_date": metadata.get("post_date", ""),
        "creator_id": metadata.get("creator_id", ""),
        "product_id": metadata.get("product_id", ""),
        "brand_id": metadata.get("brand_id", ""),
        "product_category": metadata.get("product_category", ""),
        "product_price": metadata.get("product_price", ""),
        "price_tier": metadata.get("price_tier", ""),
        "hedonic_utilitarian": metadata.get("hedonic_utilitarian", ""),
    }


def make_run_id(video_path: str, product_info_path: str, product_image_path: str) -> str:
    seed = f"{video_path}|{product_info_path}|{product_image_path}|{uuid.uuid4().hex}"
    digest = hashlib.md5(seed.encode("utf-8")).hexdigest()[:12]
    return f"run_{digest}"


def initialize_run_paths(run_id: str) -> Tuple[Path, Path, Path]:
    run_dir = RUNS_BASE_DIR / run_id
    frames_dir = run_dir / "frames"
    audio_dir = run_dir / "audio"
    ensure_clean_dir(run_dir)
    ensure_dir(frames_dir)
    ensure_dir(audio_dir)
    return run_dir, frames_dir, audio_dir


def update_error_state(state: AESState, stage: str, message: str, status: str) -> AESState:
    return {
        **state,
        "status": status,
        "status_reason": message,
        "error_stage": stage,
        "error_message": message,
        "is_commercial_candidate": False,
    }


def finalize_error_if_unset(state: AESState) -> AESState:
    if state.get("status") in {"AES_READY", "MISSING_PRODUCT_REFERENCE", "MISSING_PRODUCT_IMAGE", "NOT_APPLICABLE"}:
        return state
    if state.get("error_stage") and state.get("status_reason"):
        return state
    return state


# =========================================================
# Models
# =========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weights = models.ResNet50_Weights.DEFAULT
resnet = models.resnet50(weights=weights)
feature_extractor = nn.Sequential(*list(resnet.children())[:-1]).to(device)
feature_extractor.eval()
image_transform = weights.transforms()

_text_model = None
_whisper_model = None

def get_text_model():
    global _text_model
    if _text_model is None:
        # Prefer an existing local snapshot. This avoids unnecessary Hub HEAD
        # requests (and long retries) when the research app is run offline.
        try:
            local_model_path = snapshot_download(
                repo_id=TEXT_MODEL_NAME,
                local_files_only=True,
            )
            _text_model = SentenceTransformer(
                local_model_path,
                device="cpu",
            )
        except (OSError, ValueError, FileNotFoundError):
            # First use still downloads the model through the normal
            # SentenceTransformers/Hugging Face mechanism.
            _text_model = SentenceTransformer(TEXT_MODEL_NAME, device="cpu")
    return _text_model

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = WhisperModel(WHISPER_MODEL_NAME, compute_type=WHISPER_COMPUTE_TYPE)
    return _whisper_model


# =========================================================
# Feature extractors
# =========================================================

def extract_image_feature(image_path: str) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    tensor = image_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = feature_extractor(tensor)

    return features.squeeze().detach().cpu().numpy().astype(np.float32)


def transcribe_audio_segments(audio_path: str, initial_prompt: str = "") -> List[Dict[str, Any]]:
    transcribe_options: Dict[str, Any] = {
        "language": TRANSCRIPTION_LANGUAGE,
        "task": "transcribe",
        "vad_filter": True,
    }
    if initial_prompt.strip():
        transcribe_options["initial_prompt"] = initial_prompt.strip()[:240]
    segments, info = get_whisper_model().transcribe(audio_path, **transcribe_options)
    items: List[Dict[str, Any]] = []

    for seg in segments:
        text = (seg.text or "").strip()
        if not text:
            continue
        items.append({
            "text": text,
            "start": round(float(seg.start), 2),
            "end": round(float(seg.end), 2),
        })

    return items


def get_or_compute_product_text_embedding(state: AESState) -> Tuple[np.ndarray, AESState]:
    cached = state.get("product_text_embedding", [])
    cached_model = state.get("product_text_embedding_model", "")
    if cached and cached_model == TEXT_MODEL_NAME:
        return np.array(cached, dtype=np.float32), {
            **state,
            "product_text_embedding_cached": True,
        }

    product_description = state.get("product_embedding_text", "").strip()
    emb = get_text_model().encode(
        product_description,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)

    return emb, {
        **state,
        "product_text_embedding": emb.tolist(),
        "product_text_embedding_model": TEXT_MODEL_NAME,
        "product_text_embedding_cached": True,
    }


def get_or_compute_product_image_feature(state: AESState) -> Tuple[np.ndarray, AESState]:
    cached = state.get("product_image_feature", [])
    if cached:
        return np.array(cached, dtype=np.float32), {
            **state,
            "product_image_feature_cached": True,
        }

    product_image_path = state.get("product_image_path", "").strip()
    feature = extract_image_feature(product_image_path).astype(np.float32)

    return feature, {
        **state,
        "product_image_feature": feature.tolist(),
        "product_image_feature_cached": True,
    }


# =========================================================
# Nodes
# =========================================================

def preprocess_node(state: AESState) -> AESState:
    video_path = state.get("video_path", "").strip()
    product_info_path = state.get("product_info_path", "").strip()
    product_image_path = state.get("product_image_path", "").strip()
    run_id = state.get("run_id", "").strip() or make_run_id(video_path, product_info_path, product_image_path)

    if not video_path:
        raise ValueError("video_path is required.")
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    run_dir, frames_dir, audio_dir = initialize_run_paths(run_id)
    product_description = read_text_file(product_info_path) if product_info_path else ""
    product_embedding_text = build_product_embedding_text(product_description)
    cached_embedding = state.get("product_text_embedding", [])
    cached_embedding_model = state.get("product_text_embedding_model", "")
    if cached_embedding_model != TEXT_MODEL_NAME:
        cached_embedding = []

    return {
        **state,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "frames_dir": str(frames_dir),
        "audio_dir": str(audio_dir),
        "video_path": video_path,
        "product_info_path": product_info_path,
        "product_image_path": product_image_path,
        "product_description": product_description,
        "product_embedding_text": product_embedding_text,
        "text_model_name": TEXT_MODEL_NAME,
        "transcription_language": TRANSCRIPTION_LANGUAGE,
        "semantic_scoring_version": SEMANTIC_SCORING_VERSION,
        "threshold_calibration_status": THRESHOLD_CALIBRATION_STATUS,
        "status": "PREPROCESS_READY",
        "status_reason": "Input paths and run directories are initialized.",
        "error_stage": "",
        "error_message": "",
        "audio_extraction_success": False,
        "audio_error": "",
        "product_text_embedding": cached_embedding,
        "product_text_embedding_model": TEXT_MODEL_NAME if cached_embedding else "",
        "product_image_feature": state.get("product_image_feature", []),
        "product_text_embedding_cached": bool(cached_embedding),
        "product_image_feature_cached": bool(state.get("product_image_feature", [])),
    }


def frame_extraction_node(state: AESState) -> AESState:
    video_path = state["video_path"]
    output_dir = Path(state["frames_dir"])
    ensure_clean_dir(output_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return update_error_state(
            state,
            stage="frame_extraction",
            message=f"Could not open video: {video_path}",
            status="FRAME_EXTRACTION_FAILED",
        )

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        fps = 30.0

    duration_seconds = total_frames / fps if total_frames > 0 else 0.0
    interval_sec = 1.0
    timestamps = np.arange(0, duration_seconds, interval_sec)

    extracted: List[Dict[str, Any]] = []

    try:
        for idx, ts in enumerate(timestamps):
            frame_idx = int(ts * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, frame = cap.read()

            if success and frame is not None:
                frame_path = output_dir / f"frame_{idx:04d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                extracted.append({
                    "path": str(frame_path),
                    "timestamp": round(float(ts), 2),
                })
    finally:
        cap.release()

    if len(extracted) == 0:
        return update_error_state(
            {
                **state,
                "frame_count": 0,
                "video_duration_seconds": round(duration_seconds, 2),
                "extracted_frame_items": [],
            },
            stage="frame_extraction",
            message="No frames could be extracted from the video.",
            status="NOT_APPLICABLE",
        )

    return {
        **state,
        "extracted_frame_items": extracted,
        "frame_count": len(extracted),
        "video_duration_seconds": round(duration_seconds, 2),
    }


def audio_extraction_node(state: AESState) -> AESState:
    if state.get("status") not in {"PREPROCESS_READY", "AES_READY", "PRECHECK_READY", "PRECHECK_WARNING", ""} and state.get("frame_count", 0) == 0:
        return state

    if state.get("status") in {"FRAME_EXTRACTION_FAILED", "NOT_APPLICABLE"}:
        return state

    video_path = state["video_path"]
    audio_dir = Path(state["audio_dir"])
    ensure_clean_dir(audio_dir)

    audio_path = audio_dir / "temp_audio.wav"

    cmd = [
        "/opt/homebrew/bin/ffmpeg",
        "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(audio_path)
    ]

    try:
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return {
            **state,
            "audio_path": str(audio_path),
            "audio_extraction_success": True,
            "audio_error": "",
        }
    except FileNotFoundError as e:
        return update_error_state(
            {
                **state,
                "audio_path": "",
                "audio_extraction_success": False,
                "audio_error": f"ffmpeg not found: {e}",
            },
            stage="audio_extraction",
            message="Audio extraction failed because ffmpeg is not installed or not found in PATH.",
            status="AUDIO_EXTRACTION_FAILED",
        )
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or "").strip()
        msg = stderr if stderr else str(e)
        return update_error_state(
            {
                **state,
                "audio_path": "",
                "audio_extraction_success": False,
                "audio_error": msg,
            },
            stage="audio_extraction",
            message=f"Audio extraction failed: {msg}",
            status="AUDIO_EXTRACTION_FAILED",
        )
    except Exception as e:
        return update_error_state(
            {
                **state,
                "audio_path": "",
                "audio_extraction_success": False,
                "audio_error": str(e),
            },
            stage="audio_extraction",
            message=f"Audio extraction failed: {e}",
            status="AUDIO_EXTRACTION_FAILED",
        )


def transcription_node(state: AESState) -> AESState:
    if state.get("status") == "AUDIO_EXTRACTION_FAILED":
        return {
            **state,
            "transcript": "",
            "transcript_segments": [],
        }

    audio_path = state.get("audio_path", "")
    if not audio_path or not Path(audio_path).exists():
        return {
            **state,
            "transcript": "",
            "transcript_segments": [],
        }

    try:
        prompt = build_transcription_prompt(state.get("product_description", ""))
        segments = transcribe_audio_segments(audio_path, initial_prompt=prompt)
        transcript = " ".join([x["text"] for x in segments]).strip()
        return {
            **state,
            "transcript": transcript,
            "transcript_segments": segments,
        }
    except Exception as e:
        return update_error_state(
            {
                **state,
                "transcript": "",
                "transcript_segments": [],
            },
            stage="transcription",
            message=f"Transcription failed: {e}",
            status="TRANSCRIPTION_FAILED",
        )


def sentence_segmentation_node(state: AESState) -> AESState:
    segments = state.get("transcript_segments", [])
    sentences: List[Dict[str, Any]] = []

    sent_id = 0
    for seg in segments:
        seg_text = seg["text"].strip()
        if not seg_text:
            continue

        chunks = split_mixed_text_to_sentences(seg_text)
        if not chunks:
            continue

        timed_chunks = distribute_segment_time(seg["start"], seg["end"], chunks)
        for item in timed_chunks:
            sentences.append({
                "sentence_id": sent_id,
                "text": item["text"],
                "start": item["start"],
                "end": item["end"],
            })
            sent_id += 1

    return {
        **state,
        "sentences": sentences,
    }


def applicability_check_node(state: AESState) -> AESState:
    frame_count = state.get("frame_count", 0)
    product_description = state.get("product_description", "").strip()
    product_embedding_text = state.get("product_embedding_text", "").strip()
    product_image_path = state.get("product_image_path", "").strip()
    transcript = state.get("transcript", "").strip()
    audio_extraction_success = state.get("audio_extraction_success", False)
    audio_error = state.get("audio_error", "")

    if state.get("status") == "FRAME_EXTRACTION_FAILED":
        return state

    if frame_count == 0:
        return update_error_state(
            state,
            stage="applicability_check",
            message="No frames could be extracted from the video.",
            status="NOT_APPLICABLE",
        )

    if not product_description or not product_embedding_text:
        return {
            **state,
            "status": "MISSING_PRODUCT_REFERENCE",
            "status_reason": "Usable product description is missing.",
            "error_stage": "applicability_check",
            "error_message": "Usable product description is missing.",
            "is_commercial_candidate": False,
        }

    if not product_image_path or not Path(product_image_path).exists():
        return {
            **state,
            "status": "MISSING_PRODUCT_IMAGE",
            "status_reason": "Product image is missing.",
            "error_stage": "applicability_check",
            "error_message": "Product image is missing.",
            "is_commercial_candidate": False,
        }

    if not audio_extraction_success and audio_error:
        return {
            **state,
            "status": "PRECHECK_WARNING",
            "status_reason": "Audio extraction failed; analysis can continue using visual evidence, but transcript-derived evidence is unavailable.",
            "error_stage": state.get("error_stage", "audio_extraction"),
            "error_message": state.get("error_message", audio_error),
            "is_commercial_candidate": True,
        }

    if audio_extraction_success and not transcript:
        return {
            **state,
            "status": "PRECHECK_WARNING",
            "status_reason": "Audio was extracted but transcript is empty; analysis can continue using visual evidence.",
            "error_stage": state.get("error_stage", ""),
            "error_message": state.get("error_message", ""),
            "is_commercial_candidate": True,
        }

    return {
        **state,
        "status": "AES_READY",
        "status_reason": "Video, product description, and product image are available for AES computation.",
        "is_commercial_candidate": True,
    }


def visual_similarity_node(state: AESState) -> AESState:
    if state.get("status") not in {"AES_READY", "PRECHECK_WARNING"}:
        return state

    try:
        product_feature, updated_state = get_or_compute_product_image_feature(state)
        frame_items = updated_state.get("extracted_frame_items", [])

        results: List[Dict[str, Any]] = []
        sims: List[float] = []

        for item in frame_items:
            frame_feature = extract_image_feature(item["path"])
            sim = clamp(cosine_similarity(frame_feature, product_feature))
            sims.append(sim)
            results.append({
                "timestamp": item["timestamp"],
                "path": item["path"],
                "similarity": round(sim, 4),
            })

        avg_sim = float(np.mean(sims)) if sims else 0.0
        peak_sim = float(np.max(sims)) if sims else 0.0
        high_ratio = float(np.mean([s >= 0.3 for s in sims])) if sims else 0.0

        top_k = min(10, len(sims))
        top_10_mean = float(np.mean(sorted(sims, reverse=True)[:top_k])) if sims else 0.0

        return {
            **updated_state,
            "frame_similarity_items": results,
            "frame_similarities": [round(x, 4) for x in sims],
            "visual_similarity_score": round(avg_sim, 4),
            "visual_peak_similarity": round(peak_sim, 4),
            "high_visual_similarity_ratio": round(high_ratio, 4),
            "top_10_frame_mean": round(top_10_mean, 4),
            "visual_explicitness_score": round(avg_sim, 4),
            "visual_reason": (
                f"Visual similarity was computed between the product image and "
                f"{len(frame_items)} sampled frames."
            ),
        }
    except Exception as e:
        return update_error_state(
            state,
            stage="visual_similarity",
            message=f"Visual similarity computation failed: {e}",
            status="VISUAL_SIMILARITY_FAILED",
        )


def semantic_explicitness_node(state: AESState) -> AESState:
    if state.get("status") not in {"AES_READY", "PRECHECK_WARNING"}:
        return state

    transcript = state.get("transcript", "").strip()
    sentences = state.get("sentences", [])

    if not transcript:
        base_status = state.get("status", "PRECHECK_WARNING")
        if base_status == "AES_READY":
            base_status = "PRECHECK_WARNING"
        return {
            **state,
            "status": base_status,
            "global_semantic_similarity": 0.0,
            "sentence_similarities": [],
            "sentence_peak_similarity": 0.0,
            "high_similarity_sentence_ratio": 0.0,
            "top_5_sentence_mean": 0.0,
            "semantic_explicitness_score": 0.0,
            "semantic_reason": "No transcript available; textual explicitness is unsupported.",
        }

    try:
        debug_info = {
            "substage": "start",
            "text_model_name": TEXT_MODEL_NAME,
            "semantic_scoring_version": SEMANTIC_SCORING_VERSION,
            "sentence_count": len(sentences),
            "transcript_len": len(transcript),
            "product_description_len": len(state.get("product_description", "").strip()),
            "product_embedding_text_len": len(state.get("product_embedding_text", "").strip()),
        }

        # 1) product embedding
        debug_info["substage"] = "product_embedding"
        product_emb, updated_state = get_or_compute_product_text_embedding(state)
        debug_info["product_embedding_dim"] = int(len(product_emb)) if product_emb is not None else 0

        # 2) transcript embedding
        debug_info["substage"] = "transcript_embedding"
        transcript_emb = get_text_model().encode(
            transcript,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)
        debug_info["transcript_embedding_dim"] = int(len(transcript_emb)) if transcript_emb is not None else 0
        global_sim = clamp(cosine_similarity(transcript_emb, product_emb))

        # 3) sentence embeddings
        debug_info["substage"] = "sentence_embedding_batch"
        sent_scores: List[float] = []
        enriched_sentences: List[Dict[str, Any]] = []

        if sentences:
            model = get_text_model()
            sentence_texts = [s.get("text", "") for s in sentences]
            sentence_embeddings = model.encode(
                sentence_texts,
                batch_size=32,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).astype(np.float32)
            for s, sent_emb in zip(sentences, sentence_embeddings):
                sim = clamp(cosine_similarity(sent_emb, product_emb))
                sent_scores.append(sim)
                enriched_sentences.append({
                    **s,
                    "similarity": round(sim, 4),
                })

        debug_info["substage"] = "score_aggregation"
        peak_sim = float(np.max(sent_scores)) if sent_scores else 0.0
        high_ratio = float(np.mean([x >= 0.5 for x in sent_scores])) if sent_scores else 0.0
        top_k = min(5, len(sent_scores))
        top_5_mean = float(np.mean(sorted(sent_scores, reverse=True)[:top_k])) if sent_scores else 0.0

        sw = AES_CONFIG["semantic_score_weights"]
        semantic_score = (
            sw["global"] * global_sim
            + sw["peak"] * peak_sim
            + sw["top5"] * top_5_mean
        )

        debug_info["substage"] = "completed"

        return {
            **updated_state,
            "sentences": enriched_sentences,
            "global_semantic_similarity": round(global_sim, 4),
            "sentence_similarities": [round(x, 4) for x in sent_scores],
            "sentence_peak_similarity": round(peak_sim, 4),
            "high_similarity_sentence_ratio": round(high_ratio, 4),
            "top_5_sentence_mean": round(top_5_mean, 4),
            "semantic_explicitness_score": round(semantic_score, 4),
            "semantic_reason": (
                f"Global similarity={global_sim:.4f}; "
                f"sentence peak={peak_sim:.4f}; "
                f"high-sim sentence ratio={high_ratio:.4f}."
            ),
            "semantic_debug": debug_info,
        }
    except Exception as e:
        error_message = (
            f"Semantic explicitness failed at {debug_info.get('substage', 'unknown')}: {e} | "
            f"debug={debug_info}"
        )
        return update_error_state(
            {
                **state,
                "semantic_debug": debug_info,
            },
            stage="semantic_explicitness",
            message=error_message,
            status="SEMANTIC_EXPLICITNESS_FAILED",
        )


def align_sentence_frame_node(state: AESState) -> AESState:
    if state.get("status") not in {"AES_READY", "PRECHECK_WARNING"}:
        return state

    sentences = state.get("sentences", [])
    frame_items = state.get("frame_similarity_items", [])

    aligned: List[Dict[str, Any]] = []

    for s in sentences:
        start, end = float(s["start"]), float(s["end"])
        matched = [f for f in frame_items if start <= float(f["timestamp"]) <= end]

        if matched:
            visual_mean = float(np.mean([m["similarity"] for m in matched]))
            visual_peak = float(np.max([m["similarity"] for m in matched]))
        else:
            visual_mean = 0.0
            visual_peak = 0.0

        aligned.append({
            **s,
            "window_visual_mean": round(visual_mean, 4),
            "window_visual_peak": round(visual_peak, 4),
        })

    return {
        **state,
        "sentences": aligned,
    }


def controller_node(state: AESState) -> AESState:
    transcript = state.get("transcript", "").strip()
    sentences = state.get("sentences", [])

    # Keep the legacy token field for compatibility, but base quality control on
    # language-neutral text units so unspaced Chinese is evaluated correctly.
    transcript_token_count = len(transcript.split()) if transcript else 0
    transcript_text_unit_count = count_text_units(transcript)
    transcript_cjk_character_count = count_cjk_characters(transcript)
    sentence_count = len(sentences)

    sentence_text_unit_counts = [
        count_text_units(s.get("text", "").strip())
        for s in sentences
        if s.get("text", "").strip()
    ]
    avg_text_units_per_sentence = (
        float(np.mean(sentence_text_unit_counts)) if sentence_text_unit_counts else 0.0
    )

    video_duration = state.get("video_duration_seconds", 0.0)
    transcript_segments = state.get("transcript_segments", [])

    covered_duration = 0.0
    for seg in transcript_segments:
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        covered_duration += max(0.0, end - start)

    transcript_coverage_ratio = (
        covered_duration / video_duration
        if video_duration and video_duration > 0
        else 0.0
    )

    th = CONTROLLER_THRESHOLDS

    if state.get("audio_extraction_success") is False and state.get("audio_error"):
        transcript_quality = "low"
        video_mode = "visual_led"
        reason = "Audio extraction failed, so the analysis relies mainly on visual evidence."
    elif not transcript or sentence_count == 0:
        transcript_quality = "low"
        video_mode = "visual_led"
        reason = "No usable transcript was extracted, so the analysis relies mainly on visual evidence."
    elif (
        transcript_coverage_ratio < th["low_coverage"]
        or sentence_count < th["low_sentence_count"]
        or transcript_text_unit_count < th["low_text_unit_count"]
        or avg_text_units_per_sentence < th["low_avg_text_units"]
    ):
        transcript_quality = "low"
        video_mode = "visual_led"
        reason = "Transcript coverage is too sparse or fragmented, so the analysis relies mainly on visual evidence."
    elif (
        transcript_coverage_ratio < th["medium_coverage"]
        or sentence_count < th["medium_sentence_count"]
        or transcript_text_unit_count < th["medium_text_unit_count"]
        or avg_text_units_per_sentence < th["medium_avg_text_units"]
    ):
        transcript_quality = "medium"
        video_mode = "speech_poor"
        reason = "Transcript is partially usable, so semantic evidence is down-weighted."
    else:
        transcript_quality = "high"
        video_mode = "speech_rich"
        reason = "Transcript quality is sufficient for normal multimodal AES aggregation."

    return {
        **state,
        "transcript_token_count": transcript_token_count,
        "transcript_text_unit_count": transcript_text_unit_count,
        "transcript_cjk_character_count": transcript_cjk_character_count,
        "effective_sentence_count": sentence_count,
        # Backwards-compatible alias. New analyses should use
        # avg_text_units_per_sentence instead.
        "avg_words_per_sentence": round(avg_text_units_per_sentence, 4),
        "avg_text_units_per_sentence": round(avg_text_units_per_sentence, 4),
        "transcript_coverage_ratio": round(transcript_coverage_ratio, 4),
        "transcript_quality": transcript_quality,
        "video_mode": video_mode,
        "controller_reason": reason,
    }


def aes_score_node(state: AESState) -> AESState:
    if state.get("status") not in {"AES_READY", "PRECHECK_WARNING"}:
        return state

    try:
        visual_avg = state.get("visual_similarity_score", 0.0)
        visual_peak = state.get("visual_peak_similarity", 0.0)
        visual_top10 = state.get("top_10_frame_mean", 0.0)

        semantic_global = state.get("global_semantic_similarity", 0.0)
        semantic_peak = state.get("sentence_peak_similarity", 0.0)
        semantic_top5 = state.get("top_5_sentence_mean", 0.0)

        video_mode = state.get("video_mode", "speech_rich")

        vw = AES_CONFIG["visual_score_weights"]
        sw = AES_CONFIG["semantic_score_weights"]
        fw = AES_CONFIG["fusion_weights"].get(video_mode, AES_CONFIG["fusion_weights"]["speech_rich"])
        th = AES_CONFIG["thresholds"]

        visual_score = (
            vw["avg"] * visual_avg
            + vw["top10"] * visual_top10
        )

        semantic_score = (
            sw["global"] * semantic_global
            + sw["peak"] * semantic_peak
            + sw["top5"] * semantic_top5
        )

        transcript = str(state.get("transcript") or "").strip()
        effective_sentence_count = int(state.get("effective_sentence_count", 0) or 0)
        no_usable_semantic_evidence = not transcript or effective_sentence_count == 0

        if no_usable_semantic_evidence:
            final_visual_weight = 1.0
            final_semantic_weight = 0.0
        else:
            final_visual_weight = fw["visual"]
            final_semantic_weight = fw["semantic"]

        aes_score = round(
            final_visual_weight * visual_score
            + final_semantic_weight * semantic_score,
            4
        )

        high_sentence_ratio = state.get("high_similarity_sentence_ratio", 0.0)

        is_product_focused_review = (
            semantic_global >= th["product_focused_semantic_global"]
            and semantic_top5 >= th["product_focused_semantic_top5"]
            and visual_top10 >= th["product_focused_visual_top10"]
        )

        has_local_promo_segment = (
            not is_product_focused_review
            and (
                high_sentence_ratio >= th["local_promo_high_sentence_ratio"]
                or semantic_peak >= th["local_promo_semantic_peak"]
                or visual_top10 >= th["local_promo_visual_top10"]
            )
        )

        if aes_score >= th["high_aes"]:
            if is_product_focused_review:
                reason = "High AES: the video is strongly product-focused and highly commercially explicit."
            else:
                reason = "High AES: the video is highly commercially explicit."
        elif aes_score <= th["low_aes"]:
            if has_local_promo_segment:
                reason = (
                    "Low AES: the video shows low overall product/promo explicitness, "
                    "but contains a local promo-like segment."
                )
            else:
                reason = (
                    "Low AES: the video shows low overall product/promo explicitness "
                    "throughout the video."
                )
        else:
            if is_product_focused_review:
                reason = "Medium AES: the video is broadly product-focused, with sustained product-centered discussion."
            elif has_local_promo_segment:
                reason = (
                    "Medium AES: the video shows mixed levels of product/promo explicitness "
                    "with a noticeable local promo-like segment."
                )
            else:
                reason = "Medium AES: the video shows mixed levels of product/promo explicitness."

        final_status = state.get("status", "AES_READY")
        if final_status not in {"AES_READY", "PRECHECK_WARNING"}:
            final_status = "AES_READY"

        return {
            **state,
            "status": final_status,
            "aes_score": aes_score,
            "aes_components": {
                "video_mode": video_mode,
                "visual_avg": round(visual_avg, 4),
                "visual_peak": round(visual_peak, 4),
                "visual_top10_mean": round(visual_top10, 4),
                "visual_final": round(visual_score, 4),
                "semantic_global": round(semantic_global, 4),
                "semantic_peak": round(semantic_peak, 4),
                "semantic_top5_mean": round(semantic_top5, 4),
                "semantic_final": round(state.get("semantic_explicitness_score", 0.0), 4),
                "semantic_final_for_aes": round(semantic_score, 4),
                "final_visual_weight": round(final_visual_weight, 4),
                "final_semantic_weight": round(final_semantic_weight, 4),
                "is_product_focused_review": is_product_focused_review,
                "has_local_promo_segment": has_local_promo_segment,
                "controller_thresholds": CONTROLLER_THRESHOLDS,
                "aes_config": AES_CONFIG,
            },
            "aes_reason": reason,
        }
    except Exception as e:
        return update_error_state(
            state,
            stage="aes_score",
            message=f"AES score computation failed: {e}",
            status="AES_SCORING_FAILED",
        )


# =========================================================
# Build graph
# =========================================================

graph = StateGraph(AESState)

graph.add_node("preprocess", preprocess_node)
graph.add_node("frame_extraction", frame_extraction_node)
graph.add_node("audio_extraction", audio_extraction_node)
graph.add_node("transcription", transcription_node)
graph.add_node("sentence_segmentation", sentence_segmentation_node)
graph.add_node("applicability_check", applicability_check_node)
graph.add_node("visual_similarity", visual_similarity_node)
graph.add_node("semantic_explicitness", semantic_explicitness_node)
graph.add_node("align_sentence_frame", align_sentence_frame_node)
graph.add_node("controller", controller_node)
graph.add_node("aes_score", aes_score_node)

graph.add_edge(START, "preprocess")
graph.add_edge("preprocess", "frame_extraction")
graph.add_edge("frame_extraction", "audio_extraction")
graph.add_edge("audio_extraction", "transcription")
graph.add_edge("transcription", "sentence_segmentation")
graph.add_edge("sentence_segmentation", "applicability_check")
graph.add_edge("applicability_check", "visual_similarity")
graph.add_edge("visual_similarity", "semantic_explicitness")
graph.add_edge("semantic_explicitness", "align_sentence_frame")
graph.add_edge("align_sentence_frame", "controller")
graph.add_edge("controller", "aes_score")
graph.add_edge("aes_score", END)

app = graph.compile()


# =========================================================
# Outer functions for metadata + Study 1
# =========================================================

def analyze_single_video(
    video_path: str,
    product_info_path: str,
    product_image_path: str,
    run_id: str | None = None,
) -> dict:
    actual_run_id = run_id or make_run_id(video_path, product_info_path, product_image_path)
    return app.invoke({
        "video_path": video_path,
        "product_info_path": product_info_path,
        "product_image_path": product_image_path,
        "run_id": actual_run_id,
    })


def analyze_single_video_with_metadata(
    video_path: str,
    product_info_path: str,
    product_image_path: str,
    metadata: dict,
    run_id: str | None = None,
) -> dict:
    meta = fetch_video_metadata_from_dict(metadata)

    result = analyze_single_video(
        video_path=video_path,
        product_info_path=product_info_path,
        product_image_path=product_image_path,
        run_id=run_id,
    )

    engagement_rate = meta["interaction_rate"]
    if engagement_rate is None:
        engagement_rate = compute_engagement_rate(
            likes=meta["likes"],
            comments=meta["comments"],
            shares=meta["shares"],
            views=meta["views"],
        )

    merged = {
        **result,
        "video_id": meta["video_id"],
        "likes": meta["likes"],
        "comments": meta["comments"],
        "shares": meta["shares"],
        "views": meta["views"],
        "engagement_rate": engagement_rate,
        "post_date": meta["post_date"],
        "creator_id": meta["creator_id"],
        "product_id": meta["product_id"],
        "brand_id": meta["brand_id"],
        "product_category": meta["product_category"],
        "product_price": meta["product_price"],
        "price_tier": meta["price_tier"],
        "hedonic_utilitarian": meta["hedonic_utilitarian"],
    }

    return merged


def flatten_study1_record(result: dict) -> dict:
    components = result.get("aes_components", {})

    return {
        "video_id": result.get("video_id", ""),
        "video_path": result.get("video_path", ""),
        "run_id": result.get("run_id", ""),
        "run_dir": result.get("run_dir", ""),
        "post_date": result.get("post_date", ""),
        "creator_id": result.get("creator_id", ""),
        "product_id": result.get("product_id", ""),
        "brand_id": result.get("brand_id", ""),
        "product_category": result.get("product_category", ""),
        "product_price": result.get("product_price", ""),
        "price_tier": result.get("price_tier", ""),
        "hedonic_utilitarian": result.get("hedonic_utilitarian", ""),

        "status": result.get("status", ""),
        "status_reason": result.get("status_reason", ""),
        "error_stage": result.get("error_stage", ""),
        "error_message": result.get("error_message", ""),

        "likes": result.get("likes", 0),
        "comments": result.get("comments", 0),
        "shares": result.get("shares", 0),
        "views": result.get("views", 0),
        "engagement_rate": result.get("engagement_rate", 0.0),

        "frame_count": result.get("frame_count", 0),
        "video_duration_seconds": result.get("video_duration_seconds", 0.0),
        "audio_extraction_success": result.get("audio_extraction_success", False),
        "audio_error": result.get("audio_error", ""),

        "text_model_name": result.get("text_model_name", TEXT_MODEL_NAME),
        "embedding_dimension": len(result.get("product_text_embedding", [])),
        "transcription_language": result.get("transcription_language", TRANSCRIPTION_LANGUAGE),
        "semantic_scoring_version": result.get("semantic_scoring_version", SEMANTIC_SCORING_VERSION),
        "threshold_calibration_status": result.get(
            "threshold_calibration_status", THRESHOLD_CALIBRATION_STATUS
        ),

        "aes_score": result.get("aes_score", 0.0),
        "aes_reason": result.get("aes_reason", ""),
        "video_mode": result.get("video_mode", ""),
        "transcript_quality": result.get("transcript_quality", ""),
        "transcript_token_count": result.get("transcript_token_count", 0),
        "transcript_text_unit_count": result.get("transcript_text_unit_count", 0),
        "transcript_cjk_character_count": result.get("transcript_cjk_character_count", 0),
        "effective_sentence_count": result.get("effective_sentence_count", 0),
        "avg_words_per_sentence": result.get("avg_words_per_sentence", 0.0),
        "avg_text_units_per_sentence": result.get("avg_text_units_per_sentence", 0.0),
        "transcript_coverage_ratio": result.get("transcript_coverage_ratio", 0.0),

        "visual_avg": components.get("visual_avg", 0.0),
        "visual_peak": components.get("visual_peak", 0.0),
        "visual_top10_mean": components.get("visual_top10_mean", 0.0),
        "visual_final": components.get("visual_final", 0.0),

        "semantic_global": components.get("semantic_global", 0.0),
        "semantic_peak": components.get("semantic_peak", 0.0),
        "semantic_top5_mean": components.get("semantic_top5_mean", 0.0),
        "semantic_final_for_aes": components.get("semantic_final_for_aes", 0.0),

        "is_product_focused_review": components.get("is_product_focused_review", False),
        "has_local_promo_segment": components.get("has_local_promo_segment", False),
    }


def batch_analyze_from_csv(csv_path: str, output_csv_path: str) -> None:
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    output_rows = []

    for idx, row in enumerate(rows):
        try:
            run_id = row.get("run_id", "").strip() or f"batch_{idx:05d}_{uuid.uuid4().hex[:8]}"
            result = analyze_single_video_with_metadata(
                video_path=row["video_path"],
                product_info_path=row["product_info_path"],
                product_image_path=row["product_image_path"],
                metadata=row,
                run_id=run_id,
            )
            flat = flatten_study1_record(result)
            output_rows.append(flat)

        except Exception as e:
            output_rows.append({
                "video_id": row.get("video_id", ""),
                "video_path": row.get("video_path", ""),
                "status": "PIPELINE_EXCEPTION",
                "error_stage": "batch_analyze_from_csv",
                "error_message": str(e),
            })

    if not output_rows:
        return

    fieldnames = sorted({k for r in output_rows for k in r.keys()})

    with open(output_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":
    sample_input = {
        "video_path": "sample_data/ad_video.mp4",
        "product_info_path": "sample_data/product_info.txt",
        "product_image_path": "sample_data/product.jpg",
    }

    sample_metadata = {
        "video_id": "demo_001",
        "likes": 1200,
        "comments": 86,
        "shares": 43,
        "views": 23000,
        "post_date": "2026-04-01",
        "creator_id": "creator_demo",
    }

    result = analyze_single_video_with_metadata(
        video_path=sample_input["video_path"],
        product_info_path=sample_input["product_info_path"],
        product_image_path=sample_input["product_image_path"],
        metadata=sample_metadata,
    )

    print("\n=== SINGLE VIDEO AES V3 + METADATA RESULT ===")
    for k, v in result.items():
        print(f"{k}: {v}")

    print("\n=== STATUS SUMMARY ===")
    print(f"status: {result.get('status', '')}")
    print(f"status_reason: {result.get('status_reason', '')}")
    print(f"error_stage: {result.get('error_stage', '')}")
    print(f"error_message: {result.get('error_message', '')}")

    print("\n=== CONTROLLER SUMMARY ===")
    print(f"video_mode: {result.get('video_mode', '')}")
    print(f"transcript_quality: {result.get('transcript_quality', '')}")
    print(f"effective_sentence_count: {result.get('effective_sentence_count', 0)}")
    print(f"transcript_text_unit_count: {result.get('transcript_text_unit_count', 0)}")
    print(f"transcript_cjk_character_count: {result.get('transcript_cjk_character_count', 0)}")
    print(f"avg_text_units_per_sentence: {result.get('avg_text_units_per_sentence', 0.0)}")
    print(f"transcript_coverage_ratio: {result.get('transcript_coverage_ratio', 0.0)}")
    print(f"controller_reason: {result.get('controller_reason', '')}")

    sentences = result.get("sentences", [])
    top_sentences = sorted(
        sentences,
        key=lambda x: x.get("similarity", 0.0),
        reverse=True
    )[:5]

    print("\n=== TOP 5 PRODUCT-RELEVANT SENTENCES ===")
    for i, s in enumerate(top_sentences, start=1):
        print(
            f"{i}. "
            f"[{s.get('start', 0.0):.2f}s - {s.get('end', 0.0):.2f}s] "
            f"sim={s.get('similarity', 0.0):.4f} | "
            f"text={s.get('text', '')}"
        )

    flat_record = flatten_study1_record(result)

    print("\n=== STUDY 1 FLAT RECORD ===")
    for k, v in flat_record.items():
        print(f"{k}: {v}")
