from __future__ import annotations


def build_parse_prompt() -> str:
    return (
        "You are a strict parser for neural network architecture specs. "
        "Input is natural language. Return ONLY a JSON object with fields: "
        "nodes: string[], edges: [fromIndex, toIndex][], constraints: object. "
        "No prose."
    )


def build_plan_prompt() -> str:
    return (
        "Given a structured NN spec (JSON), produce a concise, fillable template text "
        "that preserves nodes, edges, and key constraints for diagram rendering. "
        "Emphasize left-to-right flow, explicit layer counts, and unambiguous labels."
    )


def build_promptgen_prompt(K: int, spec_text: str) -> str:
    # Stage G1: lighter, cleaner skeleton-only prompts (no hard stylistic numbers)
    return (
        "Create K concise prompts for an image model to draw ONLY the skeleton of a neural network diagram (no text).\n"
        "Aim for a clean paper-figure look: flat 2D, simple shapes, balanced margins, and a calm palette. Use rectangles for modules and clear left→right arrows.\n"
        "If the spec implies repetition (e.g., Encoder × L), you may show a dashed grouping around the repeated blocks. Avoid flashy effects (no 3D or heavy glow).\n"
        "Return ONLY a JSON array of exactly K strings; each item is one full prompt for image generation.\n"
        f"K={K}.\n"
        f"Spec (summary):\n{spec_text}\n"
        "Each prompt must mention: 'skeleton-only, no text'."
    )


def build_judge_prompt() -> str:
    # Judge content & style, optimized for two-stage (skeleton→labels) flow
    return (
        "You are a strict publication-figure QA judge. Given a spec (JSON) and a NN diagram image, "
        "evaluate (A) Content correctness and (B) Paper-style compliance.\n"
        "(A) Content (0.6): required modules present; edges/arrows reflect correct order; arrows left→right; labels exist and are spelled correctly; "
        "layer count L indicated when applicable. If the image has no labels, include violation EXACTLY 'HARD: labels: missing'.\n"
        "(B) Style (0.4): flat 2D; white background; minimal color (black/gray + ≤2 accents); no gradients/3D/glow/shadows/neon; "
        "consistent stroke width; consistent sans-serif font; adequate spacing; dashed boxes for repeated blocks; high print readability.\n"
        "Return ONLY strict JSON: {score: number in [0,1], violations: string[]}. Violations must be concrete and actionable."
    )


def build_image_edit_prompt(instructions: str) -> str:
    # G2 and later edits: add/adjust labels only; keep geometry fixed (light constraints)
    base = (
        "Add or adjust labels INSIDE each block, without changing any shapes, arrows, layout, spacing, or colors. "
        "Keep a clean, readable look: flat 2D, simple sans-serif font, good contrast, and consistent size across blocks. "
        "Center labels within blocks; use at most two short lines; avoid covering arrows; do not add legends or titles. "
        "Use each label string exactly as provided (no translation or paraphrase). "
    )
    return base + f"Instructions: {instructions}"


def build_image_fusion_prompt(instructions: str) -> str:
    # Compose multiple images guided by text while preserving key visual constraints
    return (
        "Compose a new, clean technical diagram by integrating the following reference images. "
        "Preserve the overall paper-style look: flat 2D, white background, minimal color, consistent line width, and sans-serif text. "
        "Follow the instructions precisely; keep geometry aligned and readable; avoid extra decorations. "
        f"Instructions: {instructions}"
    )
