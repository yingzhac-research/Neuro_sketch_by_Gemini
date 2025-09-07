from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TypedDict


class ScoreItem(TypedDict):
    image_path: str
    score: float
    violations: List[str]


@dataclass
class ImageArtifact:
    prompt: str
    path: str
    meta: Dict[str, Any] = field(default_factory=dict)


class AppState(TypedDict, total=False):
    user_text: str
    spec: Dict[str, Any]
    spec_text: str
    K: int
    T: int
    round: int
    prompts: List[str]
    images: List[ImageArtifact]
    scores: List[ScoreItem]
    best_image: Optional[ImageArtifact]
    violations: List[str]
    hard_violations: List[str]
    outdir: str
