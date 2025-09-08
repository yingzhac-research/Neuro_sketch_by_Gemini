from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

from ..llm.gemini import call_gemini
from ..state import AppState, ImageArtifact


def run(state: AppState) -> AppState:
    """Generate K fused candidates by composing reference images under instructions.

    Expected in state:
    - outdir: str
    - K: int
    - base_image: optional path (treated as first ref image if present)
    - ref_images: optional list[str]
    - instructions: str
    """
    outdir = Path(state["outdir"])  # ensured by graph
    K = int(state.get("K", 3))
    instructions: str = str(state.get("instructions", "")).strip()

    # prepare reference list
    refs: List[str] = []
    if state.get("base_image"):
        refs.append(str(state["base_image"]))
    for r in state.get("ref_images", []) or []:
        if r and str(r) not in refs:
            refs.append(str(r))

    if not refs:
        raise ValueError("Fusion mode requires at least one reference image (base or ref_images)")

    max_workers = max(1, min(K, int(os.getenv("NNG_CONCURRENCY", "4"))))

    def _fuse_one(i: int) -> str:
        out_path = str(outdir / f"fused_candidate_{i}.png")
        # Use image_edit if base image is provided; otherwise image_fuse
        if state.get("base_image"):
            call_gemini(
                "image_edit",
                image_path=str(state["base_image"]),
                out_path=out_path,
                instructions=f"Variant {i}: {instructions}",
                ref_images=[p for p in refs if p != str(state["base_image"])],
            )
        else:
            call_gemini(
                "image_fuse",
                out_path=out_path,
                instructions=f"Variant {i}: {instructions}",
                ref_images=refs,
            )
        return out_path

    paths: List[str] = [""] * K
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_fuse_one, i) for i in range(K)]
        for fut in as_completed(futures):
            p = fut.result()
            try:
                idx = int(Path(p).stem.split("_")[-1])
            except Exception:
                idx = 0
            paths[idx] = p

    images = [ImageArtifact(prompt=instructions, path=pth) for pth in paths if pth]
    state["images"] = images
    return state

