from __future__ import annotations

from typing import Dict, List
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..llm.gemini import call_gemini
from ..state import AppState, ScoreItem


def run(state: AppState) -> AppState:
    images = list(state.get("images", []))
    if not images:
        state["scores"] = []
        return state

    max_workers = max(1, min(len(images), int(os.getenv("NNG_CONCURRENCY", "4"))))
    results: List[ScoreItem | None] = [None] * len(images)

    def _judge_one(i: int) -> tuple[int, Dict]:
        im = images[i]
        res: Dict = call_gemini("judge", image_path=im.path, spec=state.get("spec", {}))
        return i, res

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_judge_one, i) for i in range(len(images))]
        for fut in as_completed(futures):
            try:
                i, res = fut.result()
                im = images[i]
                results[i] = {
                    "image_path": im.path,
                    "score": float(res.get("score", 0.0)),
                    "violations": list(res.get("violations", [])),
                }
            except Exception as e:
                im = images[futures.index(fut)] if fut in futures else None
                path = im.path if im else ""
                results[i] = {
                    "image_path": path,
                    "score": 0.0,
                    "violations": [f"judge error: {e}"]
                }

    state["scores"] = [s for s in results if s is not None]
    return state
