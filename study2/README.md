# Study 2

Study 2 uses a randomized between-subjects manipulation comparing:

- High Integration
- Hard Sell

## Code Files

- `code/study2_code.py` contains the core stimulus-development, generation, analysis, screening, and selection logic.
- `code/study2_agent.py` provides the controller layer for run validation, candidate generation, analysis, screening decisions, and final stimulus selection.
- `code/study2_app.py` provides the Streamlit interface used to configure runs, manage candidate stimuli, inspect analysis, conduct screening, and finalize stimulus selection.

Supporting information and public access links for the experimental stimuli are provided in `stimuli/README.md`.

## Run

From the repository root:

```bash
streamlit run study2/code/study2_app.py
```

The interface can launch without API credentials. Local run configuration, artifact inspection, screening, and final-selection functions can be used without credentials when the relevant local artifacts exist.

Blueprint generation, Veo generation or regeneration, and Gemini-based candidate analysis require network access and Google GenAI credentials. Set one of the following environment variables:

```text
GEMINI_API_KEY
GOOGLE_API_KEY
```

Never commit a real key. The user also needs appropriate Google account access, quota, and availability for the models configured in the code.

Runtime files are written under `study2/code/study2_outputs/`. This directory contains run configurations, candidate artifacts, analysis and screening records, generated media, and final-selection exports. It is ignored by Git.
