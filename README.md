# Research Materials for Master's Dissertation

This repository contains program code and supporting experimental materials associated with a Master's dissertation examining advertising explicitness in short-form influencer content.

## Repository Structure

- `study1/`: Study 1 computational code, the final Advertising Explicitness Score implementation, and the Streamlit analysis interface.
- `study2/`: the final Study 2 stimulus-development and curation implementation, together with supporting stimulus information.

## Installation

Python 3.10 or later is required. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Internet access is required for Study 1's first-run pretrained-model downloads and for Study 2 operations that call Google GenAI services.

## Study 1

Study 1 examines the association between advertising explicitness and observed interaction rates in naturally occurring Douyin commercial videos. Advertising explicitness is operationalized using the Advertising Explicitness Score (AES), which combines multimodal visual and semantic scoring. The included computational code and Streamlit interface implement the final frozen V3 specification.

## Study 2

Study 2 is a randomized between-subjects experiment comparing two conditions:

- High Integration
- Hard Sell

The design uses AI-generated matched stimuli and randomized condition assignment. The repository includes the final Study 2 stimulus-development, generation, analysis, screening, and selection implementation. Stimulus links are provided below and in `study2/stimuli/README.md`.

## Experimental Stimuli

High Integration:  
https://study2-experiment.vercel.app/videos/high_integration.mp4

Hard Sell:  
https://study2-experiment.vercel.app/videos/hard_sell.mp4

## Data Availability

Research datasets are not included in this public repository. The repository is intended to document the computational and experimental implementation used in the dissertation.

## Reproducibility Notes

- The repository corresponds to the final dissertation methodology.
- AES V3 parameters are frozen.
- The implementation can be inspected and rerun on user-provided inputs.
- The original Study 1 research data and Study 2 participant data are not included. The exact empirical results reported in the dissertation therefore cannot be fully regenerated from this public repository alone.
- Exact historical package versions used during initial development were not fully preserved.
- Therefore, the repository documents required software libraries and model specifications without inventing unrecoverable historical package versions.

## Author

Lei ZHANG  
emlyon business school
