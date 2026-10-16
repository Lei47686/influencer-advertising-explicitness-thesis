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

## Expected Inputs

At a high level, a single-video run expects:

- a video file;
- a UTF-8 product-information text file;
- an independent product-reference image;
- the video-level metadata required by the workflow.

Batch processing expects a structured CSV containing media references, product-reference inputs, engagement metadata, dates, and the identifiers required by the code's input schema.

The code also requires FFmpeg and FFprobe to be available on the system path for media processing. Model specifications are declared in `study1_code.py`.

## Output Structure

Single-video runs write a detailed `result.json`, a flattened JSON record, and a flattened CSV record in a run-specific output directory.

Batch runs write combined results, failed-row diagnostics, summary files, modeling outputs, and any explicitly requested regression outputs. Generated outputs are excluded from version control.
