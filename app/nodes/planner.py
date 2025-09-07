from __future__ import annotations

from typing import Dict

from ..llm.gemini import call_gemini
from ..state import AppState


def run(state: AppState) -> AppState:
    if state.get("spec_text"):
        return state
    res: Dict = call_gemini("plan", spec=state.get("spec", {}))
    state["spec_text"] = res.get("spec_text", "")
    return state

