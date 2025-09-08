from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Callable

from .state import AppState
from .nodes import parser, planner, prompt_gen, gen_generate, gen_labels, judge, select, edit, archive
from .nodes import gen_fusion


def run_pipeline(state: AppState) -> AppState:
    # ensure outdir
    outdir = Path(state.get("outdir") or _default_outdir())
    outdir.mkdir(parents=True, exist_ok=True)
    state["outdir"] = str(outdir)
    state["round"] = int(state.get("round", 0))

    # 1) parse → 2) plan → 3) prompts → 4) generate (skeleton) → 5) generator_2 (labels) → 6) judge → 7) select
    state = parser.run(state)
    state = planner.run(state)
    state = prompt_gen.run(state)
    state = gen_generate.run(state)
    state = gen_labels.run(state)
    state = judge.run(state)
    state = select.run(state)

    # 8) edit loop (if hard violations or any violations, and round < T)
    T = int(state.get("T", 0))
    while (state.get("hard_violations") or state.get("violations")) and state.get("round", 0) < T:
        state["round"] = int(state.get("round", 0)) + 1
        state = edit.apply_edits(state)
        # re-judge best image
        state = _judge_best_only(state)
        state = select.run(state)

    # 9) archive
    state = archive.run(state)
    return state


def run_fusion_pipeline(state: AppState) -> AppState:
    # ensure outdir
    outdir = Path(state.get("outdir") or _default_outdir())
    outdir.mkdir(parents=True, exist_ok=True)
    state["outdir"] = str(outdir)
    state["round"] = int(state.get("round", 0))

    # Generate fused candidates from images + text instructions
    state = gen_fusion.run(state)

    # If we have candidates, select first as best; optionally judge later
    if state.get("images"):
        state["best_image"] = state["images"][0]

    # Archive results (final.png etc.)
    state = archive.run(state)
    return state


def _judge_best_only(state: AppState) -> AppState:
    # Only score the current best image again
    from .llm.gemini import call_gemini

    if not state.get("best_image"):
        return state
    res = call_gemini("judge", image_path=state["best_image"].path, spec=state.get("spec", {}))
    vios = list(res.get("violations", []))
    hard = [v for v in vios if str(v).strip().lower().startswith("hard:")]
    if not hard:
        hard = [v for v in vios if ("labels" in str(v).lower() and "missing" in str(v).lower())]
    state["scores"] = [{
        "image_path": state["best_image"].path,
        "score": float(res.get("score", 0.0)),
        "violations": vios,
    }]
    state["hard_violations"] = hard
    return state


def _default_outdir() -> str:
    return f"artifacts/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
