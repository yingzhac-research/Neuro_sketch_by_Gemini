from __future__ import annotations

import json
from pathlib import Path

from ..state import AppState


def run(state: AppState) -> AppState:
    outdir = Path(state["outdir"])  # ensured by CLI
    outdir.mkdir(parents=True, exist_ok=True)

    # dump spec
    if state.get("spec"):
        (outdir / "spec.json").write_text(json.dumps(state["spec"], ensure_ascii=False, indent=2))
    if state.get("spec_text"):
        (outdir / "spec.txt").write_text(state["spec_text"]) 

    # dump prompts
    if state.get("prompts"):
        (outdir / "prompts.json").write_text(json.dumps(state["prompts"], ensure_ascii=False, indent=2))

    # dump scores
    if state.get("scores"):
        (outdir / "scores.json").write_text(json.dumps(state["scores"], ensure_ascii=False, indent=2))

    # copy/rename final image
    if state.get("best_image"):
        src = Path(state["best_image"].path)
        dst = outdir / "final.png"
        if src.exists():
            dst.write_bytes(src.read_bytes())

    return state

