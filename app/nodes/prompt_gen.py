from __future__ import annotations

from typing import Dict

from ..llm.gemini import call_gemini
from ..state import AppState


def run(state: AppState) -> AppState:
    if state.get("prompts"):
        return state
    res: Dict = call_gemini("prompt_generate", spec_text=state.get("spec_text", ""), K=state.get("K", 3))
    state["prompts"] = res.get("prompts", [])
    return state

