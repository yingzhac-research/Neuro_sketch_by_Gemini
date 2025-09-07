from __future__ import annotations

from pathlib import Path
from typing import List

from ..state import AppState, ImageArtifact


def run(state: AppState) -> AppState:
    scores = sorted(state.get("scores", []), key=lambda s: s["score"], reverse=True)
    if not scores:
        return state
    best = scores[0]
    best_img_path = best["image_path"]
    # find the corresponding ImageArtifact
    best_image: ImageArtifact | None = None
    for im in state.get("images", []):
        if im.path == best_img_path:
            best_image = im
            break
    vios = [str(v) for v in best.get("violations", [])]
    # Identify hard violations: explicit HARD marker or labels missing heuristic
    hard = [v for v in vios if v.strip().lower().startswith("hard:")]
    if not hard:
        hard = [v for v in vios if ("labels" in v.lower() and "missing" in v.lower())]

    state["best_image"] = best_image
    state["violations"] = vios
    state["hard_violations"] = hard
    return state
