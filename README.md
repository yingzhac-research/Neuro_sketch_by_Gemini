# Multi-Agent Neural Network Diagram Generator (Skeleton)

This repository is a minimal, runnable skeleton that turns a textual NN spec into a publication-style diagram via a multi-agent pipeline:
- Parser → Planner → Prompt-Generator → Image-Generator (G1) → Label-Generator (G2) → Judge → Selector → (Editor loop) → Archivist
- All model calls flow through `call_gemini(...)`, making it easy to use Gemini 2.5 Flash for text and Gemini 2.5 Flash Image Preview for images.

Key additions in this version
- Two-stage generation: G1 draws the geometry-only skeleton (no text), G2 overlays labels on top of the skeleton.
- Hard violations: Judge returns actionable violations; missing labels are flagged as HARD to trigger edits reliably.
- Parallelism: G1, G2, and Judge run in parallel; set `NNG_CONCURRENCY` (default 4).
- Remote images by default: image generate/edit use Gemini image models. If API is missing, the system can fall back to a local placeholder to stay runnable.

## Quick Start

1) Python 3.10+

2) Install deps
```
pip install -r requirements.txt
```

3) Configure Gemini (choose one)
- Env var: `export GEMINI_API_KEY=YOUR_KEY`
- File: create `app/llm/credentials.py` with `GEMINI_API_KEY = "YOUR_KEY"`

4) Run (K=candidates, T=max edit rounds)
```
python -m app.cli --spec spec/vit.txt --K 4 --T 1
```
Artifacts are saved under `artifacts/run_YYYYmmdd_HHMMSS/` with `final.png` as the chosen result.

## Models
- `GEMINI_MODEL` (default `gemini-2.5-flash`): parsing, planning, prompt generation, and judging.
- `GEMINI_IMAGE_MODEL` (e.g., `gemini-2.5-flash-image-preview`): image generation (G1).
- `GEMINI_IMAGE_EDIT_MODEL` (e.g., `gemini-2.5-flash-image-preview`): image editing (G2, Editor).
Notes: Image models vary by account/region. If not set or unavailable, offline placeholder rendering is used to keep the pipeline runnable.

## Structure
```
app/
  cli.py              # CLI entry (K/T/outdir)
  graph.py            # Orchestrator + edit loop
  state.py            # AppState + artifacts
  prompts.py          # Centralized prompts (parse/plan/G1/G2/judge/edit)
  nodes/
    parser.py, planner.py, prompt_gen.py
    gen_generate.py   # G1 skeleton images (no text)
    gen_labels.py     # G2 label overlay edits
    judge.py, select.py, edit.py, archive.py
  llm/
    gemini.py         # Unified wrapper (API + offline fallback)
    credentials.example.py
spec/
  vit.txt             # Example ViT spec (English)
artifacts/            # Outputs per run
```

## Tips
- Concurrency: `NNG_CONCURRENCY=4 python -m app.cli --spec ...`
- Tuning: Start with `K=4, T=1`; increase `T` for more correction rounds.
- Debug: image calls write `*.resp.txt`/`*.meta.json` alongside outputs (can be removed later if undesired).
