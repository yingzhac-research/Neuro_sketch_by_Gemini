from __future__ import annotations

from typing import Dict

from ..llm.gemini import call_gemini
from ..state import AppState


def run(state: AppState) -> AppState:
    if state.get("spec"):
        return state
    res: Dict = call_gemini("parse", user_text=state.get("user_text", ""))
    state["spec"] = res.get("spec", {})
    return state

