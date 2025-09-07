from __future__ import annotations

import argparse
import json
from pathlib import Path

from .graph import run_pipeline
from .state import AppState


def main() -> None:
    parser = argparse.ArgumentParser(description="NN Diagram Multi-Agent Pipeline")
    parser.add_argument("--spec", type=str, required=True, help="Path to .txt user prompt or .json spec")
    parser.add_argument("--K", type=int, default=4, help="Number of candidates")
    parser.add_argument("--T", type=int, default=1, help="Max edit rounds")
    parser.add_argument("--outdir", type=str, default="", help="Output directory (optional)")
    args = parser.parse_args()

    spec_path = Path(args.spec)
    if not spec_path.exists():
        raise SystemExit(f"Spec file not found: {spec_path}")

    state: AppState = {
        "K": int(args.K),
        "T": int(args.T),
        "outdir": args.outdir or "",
    }

    if spec_path.suffix.lower() == ".json":
        state["spec"] = json.loads(spec_path.read_text())
    else:
        state["user_text"] = spec_path.read_text()

    final_state = run_pipeline(state)
    print(f"Artifacts saved under: {final_state['outdir']}")


if __name__ == "__main__":
    main()
