# Repository Guidelines

## Project Structure & Module Organization
- `app/cli.py` — CLI entry point; orchestrates a full run.
- `app/graph.py` — lightweight pipeline runner and edit loop.
- `app/nodes/` — individual agent nodes (`parser.py`, `planner.py`, `prompt_gen.py`, `gen_generate.py` [G1 skeleton], `gen_labels.py` [G2 labels], `judge.py`, `select.py`, `edit.py`, `archive.py`). Each exposes `run(state)` or similar.
- `app/prompts.py` — centralized prompts for parsing/planning/generation/judging/editing.
- `app/state.py` — typed `AppState` and artifact helpers.
- `app/llm/gemini.py` — `call_gemini(kind, **kwargs)` wrapper; uses local placeholders if no API key.
- `spec/` — example specs (e.g., `spec/vit.txt`).
- `artifacts/` — run outputs (time-stamped folders with `final.png`).

## Setup, Run, and Development Commands
- Create env: `python -m venv .venv && source .venv/bin/activate` (Windows: `./.venv/Scripts/activate`).
- Install deps: `pip install -r requirements.txt`.
- Configure API (choose one):
  - Env var: `export GEMINI_API_KEY=...` (supports `.env`).
  - File: create `app/llm/credentials.py` like `credentials.example.py`.
- Run sample: `python -m app.cli --spec spec/vit.txt --K 3 --T 1`.
- Models: optionally set `GEMINI_MODEL`, `GEMINI_IMAGE_MODEL`, `GEMINI_IMAGE_EDIT_MODEL`.

## Coding Style & Naming Conventions
- Python 3.10+, PEP8, 4-space indentation, type hints required in public APIs.
- Files: snake_case; functions: `snake_case`; classes: `PascalCase`.
- Nodes are pure where possible: read from `state`, return a new `state`; side effects limited to writing under `artifacts/`.
- Centralize prompt text in `app/prompts.py`; call models via `call_gemini` only.

## Testing Guidelines
- No formal test suite yet. Prefer pytest with files under `tests/` named `test_*.py`.
- Minimal integration check: run the CLI and assert a `final.png` exists in the latest `artifacts/run_*` directory.

## Commit & Pull Request Guidelines
- Use Conventional Commits: `feat:`, `fix:`, `docs:`, `refactor:`, `chore:`.
- PRs must include: purpose, linked issues, how to run, and a sample spec plus path to produced artifact (e.g., `artifacts/run_YYYYmmdd_HHMMSS/final.png`).

## Security & Configuration Tips
- Do not commit secrets (`.env`, `app/llm/credentials.py`). Rotate keys if exposed.
- Large outputs live in `artifacts/`; avoid committing heavy assets unless necessary.
