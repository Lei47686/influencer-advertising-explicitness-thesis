# Study 1

## Purpose

Study 1 examines the association between advertising explicitness and observed interaction rates in naturally occurring Douyin commercial videos. It is an observational analysis and does not establish causality.

## Advertising Explicitness Score

The Advertising Explicitness Score (AES) combines visual product similarity with transcript-based semantic similarity. The code in this directory implements the final frozen V3 specification.

Visual score:

```text
Visual V3 = 0.30 × average frame similarity
          + 0.70 × Top-10 frame similarity
```

Semantic score:

```text
Semantic V3 = 0.35 × global semantic similarity
            + 0.15 × peak sentence similarity
            + 0.50 × Top-5 sentence similarity
```

Modality-adaptive fusion:

| Video mode | Visual weight | Semantic weight |
| --- | ---: | ---: |
| Speech-rich | 0.45 | 0.55 |
| Speech-poor | 0.60 | 0.40 |
| Visual-led | 0.85 | 0.15 |
| No usable transcript or semantic evidence | 1.00 | 0.00 |

If no usable transcript or semantic evidence is available, AES equals Visual V3. This case is handled as visual-only and does not use the ordinary 0.85/0.15 visual-led fusion.

The descriptive AES bands are:

- Low: AES ≤ 0.40
- Medium: 0.40 < AES < 0.55
- High: AES ≥ 0.55

Inferential analyses use continuous AES. The Low, Medium, and High categories are descriptive only. The implementation records the calibration status as `final_v3_frozen`.

## Code Files

- `code/study1_code.py`: video preprocessing, visual and semantic feature extraction, modality control, AES scoring, and record flattening.
- `code/study1_agent.py`: input validation, single-run and batch orchestration, output persistence, batch summaries, and optional regression routines.
- `code/study1_app.py`: Streamlit interface for single-video and batch workflows.

## Run

From the repository root:

```bash
streamlit run study1/code/study1_app.py
```

## Expected Inputs

At a high level, a single-video run expects:

- a video file in MP4, MOV, AVI, or MKV format;
- a UTF-8 product-information text file;
- an independent product-reference image in JPG, JPEG, PNG, or WebP format;
- video-level metadata, entered manually or selected from a CSV row.

Single-run metadata includes `video_id`, `creator_id`, `post_date`, `likes`, `comments`, `shares`, and `views`.

Batch processing expects a structured CSV whose rows contain `video_path`, `product_info_path`, and `product_image_path`, together with the required video, engagement, date, creator, product, and brand metadata fields. Paths in the CSV must resolve on the computer running the application.

## System Requirements

- FFmpeg must be installed and available on `PATH`.
- A browser is required for the Streamlit interface.
- Internet access is needed on first use to download pretrained ResNet50 weights, `BAAI/bge-base-zh-v1.5`, and Faster-Whisper `small`.
- CPU execution is supported but may be slower than accelerated hardware.

## Output Structure

Single-video runs write a detailed `result.json`, a flattened JSON record, and a flattened CSV record in a run-specific output directory.

Batch runs write combined results, failed-row diagnostics, summary files, modeling outputs, and any explicitly requested regression outputs. Generated outputs are excluded from version control.

The result directory can be configured in the interface. By default, saved results are written under `aes_streamlit_results/` in the directory from which Streamlit is launched. Intermediate frames and audio are created under the operating system's temporary directory in `aes_v3_runs/`; uploaded working files use the temporary `aes_v3_streamlit_uploads/` directory.
