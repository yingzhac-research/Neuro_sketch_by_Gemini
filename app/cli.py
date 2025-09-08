from __future__ import annotations

import argparse
import json
from pathlib import Path

from .graph import run_pipeline, run_fusion_pipeline
from .state import AppState


def main() -> None:
    parser = argparse.ArgumentParser(description="NN Diagram Multi-Agent Pipeline")
    parser.add_argument("--mode", type=str, choices=["text", "image"], default="text", help="'text' (spec→image) or 'image' (text+image fusion/edit)")
    parser.add_argument("--spec", type=str, required=False, help="Path to .txt user prompt or .json spec (text mode)")
    parser.add_argument("--K", type=int, default=4, help="Number of candidates")
    parser.add_argument("--T", type=int, default=1, help="Max edit rounds")
    parser.add_argument("--outdir", type=str, default="", help="Output directory (optional)")
    parser.add_argument("--base-image", type=str, default="", help="Base image to edit (image mode)")
    parser.add_argument("--ref-image", action="append", default=None, help="Additional reference image(s) (repeatable)")
    parser.add_argument("--instructions", type=str, default="", help="Edit/fusion instructions (image mode)")
    args = parser.parse_args()

    state: AppState = {"K": int(args.K), "T": int(args.T), "outdir": args.outdir or ""}

    if args.mode == "text":
        if not args.spec:
            raise SystemExit("--spec is required in text mode")
        spec_path = Path(args.spec)
        if not spec_path.exists():
            raise SystemExit(f"Spec file not found: {spec_path}")
        if spec_path.suffix.lower() == ".json":
            state["spec"] = json.loads(spec_path.read_text())
        else:
            state["user_text"] = spec_path.read_text()
        final_state = run_pipeline(state)
    else:
        # image fusion/edit mode
        base_image = args.base_image.strip()
        ref_images = args.ref_image or []
        if not base_image and not ref_images:
            raise SystemExit("image mode requires --base-image and/or at least one --ref-image")
        if base_image:
            if not Path(base_image).exists():
                raise SystemExit(f"Base image not found: {base_image}")
            state["base_image"] = base_image
        valid_refs = [p for p in ref_images if p and Path(p).exists()]
        state["ref_images"] = valid_refs
        state["instructions"] = args.instructions or "Compose and update the figure to reflect the requested component changes while keeping overall style consistent."
        final_state = run_fusion_pipeline(state)

    print(f"Artifacts saved under: {final_state['outdir']}")


if __name__ == "__main__":
    main()
