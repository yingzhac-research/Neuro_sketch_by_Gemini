from __future__ import annotations

from typing import Dict, List

from ..llm.gemini import call_gemini
from ..state import AppState, ImageArtifact


def run(state: AppState) -> AppState:
    prompts: List[str] = state.get("prompts", [])
    res: Dict = call_gemini("image_generate", prompts=prompts, outdir=state["outdir"])
    paths = res.get("paths", [])
    images = [ImageArtifact(prompt=p, path=pth) for p, pth in zip(prompts, paths)]
    state["images"] = images
    return state

