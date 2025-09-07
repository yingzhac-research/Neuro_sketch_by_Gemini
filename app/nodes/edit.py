from __future__ import annotations

from pathlib import Path
from typing import Dict

from ..llm.gemini import call_gemini
from ..state import AppState


def _labels_from_spec(state: AppState) -> list[str]:
    spec = state.get("spec", {}) or {}
    raw_nodes = spec.get("nodes", []) or []
    labels: list[str] = []
    for n in raw_nodes:
        label = None
        if isinstance(n, str):
            # strip index prefixes like "N0: ..."
            parts = n.split(":", 1)
            label = parts[1].strip() if len(parts) == 2 else n.strip()
        elif isinstance(n, dict):
            label = n.get("label") or n.get("name") or n.get("id")
        if label:
            labels.append(str(label))
    # dedupe sequential exact repeats only when later mapping by order still makes sense
    return labels


def _ascii_friendly_labels(state: AppState) -> list[str]:
    labels = _labels_from_spec(state)
    def is_ascii(s: str) -> bool:
        try:
            s.encode('ascii')
            return True
        except Exception:
            return False
    if not labels or sum(1 for l in labels if is_ascii(l)) == 0:
        return [
            "PATCH EMBEDDING",
            "CLS + POSENC",
            "ENCODER xL",
            "CLASS HEAD",
        ]
    return labels


def plan_edits(state: AppState) -> str:
    hard_violations = [str(v) for v in state.get("hard_violations", [])]
    violations = [str(v) for v in state.get("violations", [])]
    # If judge reports missing labels (prefer HARD), provide an add-labels instruction
    hv = hard_violations or violations
    if any(("labels" in v.lower() and "missing" in v.lower()) for v in hv):
        labels = _labels_from_spec(state)
        numbered = "\n".join([f"{i+1}: \"{lbl}\"" for i, lbl in enumerate(labels)]) or "(no labels provided)"
        return (
            "Add text labels INSIDE each rectangular block without changing geometry, arrows, spacing, sizes, or colors. "
            "Map labels in left→right, top→bottom order; reuse identical labels for repeated blocks. "
            "Use a clean sans-serif font in solid black or dark gray, consistent size.\n"
            f"Labels list:\n{numbered}"
        )

    # Default: targeted fixes based on judge violations, but always provide labels list to preserve text in offline mode
    fixes = "; ".join(violations) if violations else "typos, arrow direction, spacing/legibility, and style compliance"
    labels = _ascii_friendly_labels(state)
    numbered = "\n".join([f"{i+1}: \"{lbl}\"" for i, lbl in enumerate(labels)]) or "(no labels provided)"
    return (
        f"Fix the following issues precisely: {fixes}. "
        "Do not move or reshape elements. Only adjust text (content/position/size), arrow direction styles, and minimal styling to reach paper standards.\n"
        f"Labels list:\n{numbered}"
    )


def apply_edits(state: AppState) -> AppState:
    if not state.get("best_image"):
        return state
    src = state["best_image"].path
    out_path = str(Path(state["outdir"]) / f"edited_round_{state.get('round', 0)}.png")
    _ = call_gemini("image_edit", image_path=src, out_path=out_path, instructions=plan_edits(state))
    # replace best_image with edited one
    state["best_image"].path = out_path  # type: ignore
    return state
