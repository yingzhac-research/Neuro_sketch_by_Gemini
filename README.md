# Neuro Sketch by Gemini — Consistent, Fusion-First, Editable Visual AI

Neuro Sketch turns natural language into precise, publication‑style neural‑network diagrams — and edits them with words. Built for the Nano Banana hackathon, it showcases Gemini 2.5 Flash Image’s strengths beyond simple text‑to‑image:

- Consistency: preserves layout, fonts, colors, arrows and dashed groups across generations and edits.
- Fusion: blends/merges multiple references and applies textual instructions to compose new variants.
- Editing: modifies an existing figure with surgical changes (e.g., “Replace the UNet backbone with a Transformer (DiT) …”).

Under the hood it uses a lightweight multi‑agent pipeline:
- Parser → Planner → Prompt‑Generator → G1 (skeleton) → G2 (labels) → Judge → Select → Edit loop → Archive
- All model calls go through `call_gemini(...)` to centralize configuration and fall back gracefully when offline.

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
# Text mode (spec -> image)
python -m app.cli --mode text --spec spec/vit.txt --K 4 --T 1

# Image mode (text + image fusion/edit)
# Example: edit an existing diagram with a component replacement using a reference image
python -m app.cli --mode image --base-image path/to/base.png \
  --ref-image path/to/transformer_ref.png \
  --instructions "Replace the UNet backbone with a Transformer (DiT); keep layout, font, and colors consistent."
```
Artifacts are saved under `artifacts/run_YYYYmmdd_HHMMSS/` with `final.png` as the chosen result.

Demo notebook
- `demo.ipynb` includes two tasks: Prompt → Image, and Image + Prompt → Edited Image.
- Example edit uses `examples/StableDiffusion.png` as the base and applies the UNet→Transformer (DiT) replacement instruction.

## Why It’s Novel (Innovation + Wow)
- Two‑stage generation for fidelity and control: G1 draws a clean geometry‑only skeleton; G2 overlays labels without disturbing layout. This separation is the key to style consistency.
- Structured, measurable editing: a judge surfaces violations (e.g., missing labels), and an edit loop applies targeted fixes — not full regenerations — so results stay stable.
- Fusion‑first composing: takes a base diagram and optional refs, then composes new variants with textual guidance (e.g., swap UNet→Transformer, keep colors/spacing/arrows).

Real‑world impact (Utility)
- Technical documentation and education: generate consistent architecture figures for papers, slides, and courseware in minutes.
- Creative workflows: rapid, on‑brand diagram variants for marketing or A/B concepts.
- E‑commerce/product explainers: instantly customize visual system diagrams per audience (dev, exec, student) without a design bottleneck.

## Gemini 2.5 Flash Image — Consistency, Fusion, Editing
- Consistency: Prompts emphasize fixed layout/arrow direction/dashed groups/fonts; the edit loop avoids unnecessary redraws to minimize drift.
- Fusion: `gen_fusion.py` composes images with instructions and multiple refs; ideal for “blend realities” edits while preserving style.
- Editing: `gen_labels.py` and `edit.py` apply surgical changes to an existing image (labels, typos, module swaps) instead of starting over.
- Fallback/Offline: If `GEMINI_API_KEY` is not present, the system produces deterministic local placeholders so the demo always runs.

## Models
- `GEMINI_MODEL` (default `gemini-2.5-flash`): parsing, planning, prompt generation, and judging.
- `GEMINI_IMAGE_MODEL` (recommended `gemini-2.5-flash-image` or `gemini-2.5-flash-image-preview`): image generation (G1).
- `GEMINI_IMAGE_EDIT_MODEL` (recommended `gemini-2.5-flash-image` or `gemini-2.5-flash-image-preview`): image editing (G2, Editor).
Notes: If `GEMINI_API_KEY` is not set, the pipeline uses offline placeholders to remain runnable. With an API key present, you must set valid image model env vars; errors are raised if image models are unset or calls fail (no automatic local fallback).

## Fusion Mode (Text + Image)
- Accepts a base diagram (`--base-image`) and optional reference images (`--ref-image` repeatable) plus instructions.
- Uses Gemini 2.5 Flash Image to compose images under textual guidance – ideal for swapping a module (e.g., UNet → Transformer) while preserving style and layout.
- Outputs multiple fused candidates (`K`) and archives the first as `final.png`.

Examples
```
# Edit the included example diagram
python -m app.cli --mode image \
  --base-image examples/StableDiffusion.png \
  --instructions "Replace the UNet backbone with a Transformer (DiT); preserve layout, arrows, dashed groups, fonts, and colors."
```

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
  vit.txt             # Example ViT spec (ASCII-safe)
  transformer.txt     # Example Transformer spec (ASCII-safe)
examples/
  StableDiffusion.png # Example base image for edit/fusion
artifacts/            # Outputs per run
```

## Tips
- Concurrency: `NNG_CONCURRENCY=4 python -m app.cli --spec ...`
- Tuning: Start with `K=4, T=1`; increase `T` for more correction rounds.
- Debug: image calls write `*.resp.txt`/`*.meta.json` alongside outputs (can be removed later if undesired).

## Gemini Integration (≤200 words)
We use Gemini 2.5 Flash Image for three core tasks that go beyond basic text‑to‑image: (1) style‑consistent generation, (2) multi‑reference fusion, and (3) surgical editing. The pipeline separates geometry (G1) from labeling (G2) to stabilize layout and typography. A judge extracts concrete violations (e.g., missing or misspelled labels), and an edit loop applies targeted corrections with the image‑edit model, avoiding full redraws and minimizing drift. In fusion mode, we compose a base diagram with optional reference images and textual instructions (e.g., swap UNet→Transformer) to blend realities while preserving arrows, spacing, fonts, and dashed groups. All calls flow through a single `call_gemini` wrapper with configurable models and strict error surfacing; when no API key is present, deterministic placeholders keep the demo fully runnable offline. This design demonstrates consistency, fusion, and editing working together to deliver fast, controllable, publication‑quality diagrams for real documentation, education, and creative workflows.
