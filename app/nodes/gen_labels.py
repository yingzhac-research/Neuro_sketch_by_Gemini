from __future__ import annotations

from pathlib import Path
from typing import List
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..llm.gemini import call_gemini
from ..state import AppState, ImageArtifact
from .edit import _labels_from_spec  # reuse label extraction


def run(state: AppState) -> AppState:
    images: List[ImageArtifact] = state.get("images", []) or []
    if not images:
        return state

    labels = _labels_from_spec(state)
    # Fallback to ASCII-friendly defaults if labels are missing or mostly non-ASCII
    def _is_mostly_ascii(s: str) -> bool:
        try:
            s.encode('ascii')
            return True
        except Exception:
            return False
    if not labels or sum(1 for l in labels if _is_mostly_ascii(l)) == 0:
        labels = [
            "PATCH EMBEDDING",
            "CLS + POSENC",
            "ENCODER xL",
            "CLASS HEAD",
        ]
    numbered = "\n".join([f"{i+1}: \"{lbl}\"" for i, lbl in enumerate(labels)]) or "(no labels provided)"

    instructions = (
        "Add labels INSIDE each rectangular block. Do not move/resize/add/remove shapes or arrows; keep layout, spacing, and colors unchanged. "
        "Map labels in left→right, top→bottom order; reuse identical labels for repeated blocks. Use each label string exactly as given (no translation or paraphrase). "
        "Typography: clean sans-serif, readable size, centered within blocks; at most two short lines; avoid covering arrows; no legends or titles. "
        "If block count ≠ label count, do NOT add/remove shapes; place labels sequentially on existing blocks.\n"
        f"Labels list:\n{numbered}"
    )

    outdir = Path(state["outdir"]) if state.get("outdir") else Path("artifacts")
    max_workers = max(1, min(len(images), int(os.getenv("NNG_CONCURRENCY", "4"))))
    results: List[ImageArtifact | None] = [None] * len(images)

    def _label_one(i: int, im: ImageArtifact) -> tuple[int, str]:
        src = im.path
        out_path = str(outdir / f"labeled_candidate_{i}.png")
        _ = call_gemini("image_edit", image_path=src, out_path=out_path, instructions=instructions)
        return i, out_path

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_label_one, i, im) for i, im in enumerate(images)]
        for fut in as_completed(futures):
            i, out_path = fut.result()
            results[i] = ImageArtifact(prompt=images[i].prompt, path=out_path, meta={"stage": "labels"})

    state["images"] = [im for im in results if im is not None]
    return state
