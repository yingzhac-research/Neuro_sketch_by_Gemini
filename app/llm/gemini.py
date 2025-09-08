from __future__ import annotations

import base64
import json
import os
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional
import os as _os

try:
    # optional local credentials file
    from . import credentials  # type: ignore
except Exception:
    credentials = None  # type: ignore

# Load .env if present to populate environment variables
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()  # searches for .env in CWD/parents
except Exception:
    pass

def _get_api_key() -> Optional[str]:
    key = os.getenv("GEMINI_API_KEY")
    if key:
        return key
    if credentials and getattr(credentials, "GEMINI_API_KEY", None):
        return credentials.GEMINI_API_KEY  # type: ignore
    # Optional: read from ~/.config/gemini/api_key
    try:
        cfg_path = Path.home() / ".config" / "gemini" / "api_key"
        if cfg_path.exists():
            return cfg_path.read_text().strip()
    except Exception:
        pass
    return None


def call_gemini(kind: str, **kwargs) -> Dict[str, Any]:
    """Unified entry for Gemini calls.

    kind: one of {"parse", "plan", "prompt_generate", "image_generate", "judge", "image_edit", "image_fuse"}
    kwargs: payload for the corresponding action

    If API key is missing or a call fails, falls back to deterministic local placeholders
    so the pipeline remains runnable offline.
    """
    api_key = _get_api_key()
    if not api_key:
        # Simplified behavior: if no API key, always use local placeholders
        return _local_placeholder(kind, **kwargs)

    # With an API key present, call the real service and surface errors directly
    return _real_gemini(kind, api_key=api_key, **kwargs)


def _local_placeholder(kind: str, **kwargs) -> Dict[str, Any]:
    # Deterministic pseudo behavior for offline usage
    rng = random.Random(42)

    if kind == "parse":
        user_text = kwargs.get("user_text", "")
        # Very rough parse: split by arrows/lines → nodes & edges
        lines = [ln.strip() for ln in user_text.splitlines() if ln.strip()]
        nodes = [f"N{i}:{ln[:24]}" for i, ln in enumerate(lines)] or ["Input", "Conv", "FC", "Softmax"]
        edges = [[i, i + 1] for i in range(len(nodes) - 1)]
        spec = {"nodes": nodes, "edges": edges, "constraints": {"arrows": "left_to_right"}}
        return {"spec": spec}

    if kind == "plan":
        spec = kwargs.get("spec", {})
        spec_text = (
            "Neural Net Diagram\n" +
            f"Nodes: {len(spec.get('nodes', []))}\n" +
            f"Edges: {len(spec.get('edges', []))}\n" +
            f"Constraints: {spec.get('constraints', {})}\n"
        )
        return {"spec_text": spec_text}

    if kind == "prompt_generate":
        K = int(kwargs.get("K", 3))
        spec_text = kwargs.get("spec_text", "")
        layouts = ["left-right", "top-down", "circular", "grid", "hierarchical"]
        colors = ["blue", "green", "purple", "orange", "teal"]
        prompts = [
            f"Draw NN diagram ({spec_text[:40]}...) layout={layouts[i % len(layouts)]} color={colors[i % len(colors)]} seed={i}"
            for i in range(K)
        ]
        return {"prompts": prompts}

    if kind == "image_generate":
        prompts: List[str] = kwargs.get("prompts", [])
        outdir: str = kwargs.get("outdir", "artifacts")
        paths: List[str] = []
        for i, p in enumerate(prompts):
            pth = Path(outdir) / f"candidate_{i}.png"
            _write_placeholder_diagram(pth, with_labels=False)
            paths.append(str(pth))
        return {"paths": paths}

    if kind == "judge":
        image_path: str = kwargs.get("image_path")
        # produce a stable pseudo-score based on filename
        base = sum(ord(c) for c in Path(image_path).name) % 100
        score = 0.5 + (base / 200.0)
        # fake violations: if filename has odd index
        violations: List[str] = []
        try:
            idx = int(Path(image_path).stem.split("_")[-1])
            if idx % 2 == 1:
                violations = ["typo: layer name", "arrow: wrong direction"]
        except Exception:
            pass
        # If still skeleton (no 'labeled_' in name), mark missing labels as HARD
        name = Path(image_path).name.lower()
        # Heuristic for offline mode: consider labeled or edited images as having labels
        if ("labeled_" not in name) and ("edited_" not in name):
            violations = ["HARD: labels: missing"] + violations
        return {"score": score, "violations": violations}

    if kind == "image_edit":
        image_path: str = kwargs.get("image_path")
        out_path: str = kwargs.get("out_path")
        instructions: str = kwargs.get("instructions", "")
        # Extract labels from instructions if present
        import re
        labels = re.findall(r'\d+\s*:\s*"([^"]+)"', instructions)
        if not labels:
            # fallback: quoted strings
            labels = re.findall(r'"([^"]+)"', instructions)
        # standardize
        labels = [l.strip() for l in labels if l.strip()]
        _write_placeholder_diagram(Path(out_path), with_labels=True, labels=labels)
        return {"path": out_path}

    raise ValueError(f"Unsupported kind={kind}")


def _write_1x1_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # 1x1 black pixel PNG
    png_bytes = base64.b64decode(
        b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAA" \
        b"AAC0lEQVR42mP8/xcAAwMB/ax4u6kAAAAASUVORK5CYII="
    )
    with open(path, "wb") as f:
        f.write(png_bytes)


def _write_placeholder_diagram(path: Path, *, with_labels: bool, labels: Optional[List[str]] = None) -> None:
    """Generate a simple skeleton diagram (and optionally labels).

    If Pillow is available, draw with anti-aliased vectors and real text;
    otherwise fall back to a pure-stdlib bitmap renderer.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore
        _write_placeholder_diagram_pil(path, with_labels=with_labels, labels=labels)
        return
    except Exception:
        pass

    # Fallback: stdlib bitmap renderer
    # White background, 3px black strokes, arrows, dashed group
    import zlib, struct, binascii

    W, H = 1200, 420
    # initialize white canvas
    pixels: List[List[List[int]]] = [[[255, 255, 255] for _ in range(W)] for _ in range(H)]

    def set_px(x: int, y: int, c: tuple[int, int, int]):
        if 0 <= x < W and 0 <= y < H:
            pixels[y][x][0] = c[0]
            pixels[y][x][1] = c[1]
            pixels[y][x][2] = c[2]

    def draw_line(x0: int, y0: int, x1: int, y1: int, c=(0, 0, 0), t: int = 3):
        dx = abs(x1 - x0)
        sx = 1 if x0 < x1 else -1
        dy = -abs(y1 - y0)
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            for ox in range(-t // 2, t // 2 + 1):
                for oy in range(-t // 2, t // 2 + 1):
                    set_px(x0 + ox, y0 + oy, c)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    def draw_rect(x: int, y: int, w: int, h: int, c=(0, 0, 0), t: int = 3, dashed: bool = False):
        def dash_points(x0, y0, x1, y1):
            # Bresenham plus on/off dashes
            dx = abs(x1 - x0)
            sx = 1 if x0 < x1 else -1
            dy = -abs(y1 - y0)
            sy = 1 if y0 < y1 else -1
            err = dx + dy
            on = True
            step = 0
            period = 10
            points = []
            while True:
                if on:
                    points.append((x0, y0))
                step = (step + 1) % period
                if step == 0:
                    on = not on
                if x0 == x1 and y0 == y1:
                    break
                e2 = 2 * err
                if e2 >= dy:
                    err += dy
                    x0 += sx
                if e2 <= dx:
                    err += dx
                    y0 += sy
            return points

        if dashed:
            for (x0, y0, x1, y1) in [
                (x, y, x + w, y),
                (x + w, y, x + w, y + h),
                (x + w, y + h, x, y + h),
                (x, y + h, x, y),
            ]:
                for px, py in dash_points(x0, y0, x1, y1):
                    for ox in range(-t // 2, t // 2 + 1):
                        for oy in range(-t // 2, t // 2 + 1):
                            set_px(px + ox, py + oy, c)
        else:
            draw_line(x, y, x + w, y, c, t)
            draw_line(x + w, y, x + w, y + h, c, t)
            draw_line(x + w, y + h, x, y + h, c, t)
            draw_line(x, y + h, x, y, c, t)

    def draw_arrow(x0: int, y0: int, x1: int, y1: int, c=(0, 0, 0)):
        draw_line(x0, y0, x1, y1, c, 3)
        # simple arrow head
        vx, vy = x1 - x0, y1 - y0
        length = max((vx * vx + vy * vy) ** 0.5, 1.0)
        ux, uy = vx / length, vy / length
        # perpendicular
        px, py = -uy, ux
        ah = 10  # head length
        aw = 6   # head width
        hx, hy = int(x1 - ux * ah), int(y1 - uy * ah)
        lx, ly = int(hx + px * aw), int(hy + py * aw)
        rx, ry = int(hx - px * aw), int(hy - py * aw)
        draw_line(x1, y1, lx, ly, c, 2)
        draw_line(x1, y1, rx, ry, c, 2)

    # layout
    margin_x, margin_y = 60, 140
    box_w, box_h = 220, 90
    gap = 90
    y = margin_y
    xs = [margin_x + i * (box_w + gap) for i in range(4)]

    # dashed group around middle two blocks
    group_x = xs[1] - 20
    group_y = y - 20
    group_w = box_w * 2 + gap + 40
    group_h = box_h + 40
    draw_rect(group_x, group_y, group_w, group_h, c=(140, 140, 140), t=2, dashed=True)

    # blocks
    for idx, x in enumerate(xs):
        draw_rect(x, y, box_w, box_h, c=(0, 0, 0), t=2)
        if with_labels:
            # draw simple 5x7 bitmap text using ASCII-only; non-ASCII removed
            label = None
            if labels and idx < len(labels):
                label = labels[idx]
            _draw_label_text(pixels, x, y, box_w, box_h, label)

    # arrows between blocks (center-right to center-left)
    cy = y + box_h // 2
    for i in range(3):
        x0 = xs[i] + box_w
        x1 = xs[i + 1]
        draw_arrow(x0 + 4, cy, x1 - 4, cy, c=(0, 0, 0))

    # write PNG
    def png_chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", binascii.crc32(tag + data) & 0xFFFFFFFF)

    raw = bytearray()
    for row in pixels:
        raw.append(0)  # filter type 0
        for r, g, b in row:
            raw.extend((r & 255, g & 255, b & 255))
    comp = zlib.compress(bytes(raw), level=9)
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0)  # 8-bit, truecolor RGB
    png = sig + png_chunk(b"IHDR", ihdr) + png_chunk(b"IDAT", comp) + png_chunk(b"IEND", b"")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(png)

    # end _write_placeholder_diagram


def _write_placeholder_diagram_pil(path: Path, *, with_labels: bool, labels: Optional[List[str]] = None) -> None:
    from PIL import Image, ImageDraw, ImageFont  # type: ignore

    W, H = 1280, 480
    img = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    margin_x, margin_y = 80, 160
    box_w, box_h = 240, 110
    gap = 110
    y = margin_y
    xs = [margin_x + i * (box_w + gap) for i in range(4)]

    # dashed group
    group_x = xs[1] - 24
    group_y = y - 24
    group_w = box_w * 2 + gap + 48
    group_h = box_h + 48
    draw.rounded_rectangle([group_x, group_y, group_x + group_w, group_y + group_h], radius=14, outline=(150, 150, 150), width=2)
    # manual dash overlay
    # top and bottom dashed
    def dashed_line(p0, p1, dash=12, gaplen=8, width=2, fill=(150, 150, 150)):
        from math import hypot
        (x0, y0), (x1, y1) = p0, p1
        dx, dy = x1 - x0, y1 - y0
        length = (dx * dx + dy * dy) ** 0.5
        if length == 0:
            return
        ux, uy = dx / length, dy / length
        dist = 0.0
        on = True
        while dist < length:
            l = dash if on else gaplen
            nx0 = x0 + ux * dist
            ny0 = y0 + uy * dist
            nx1 = x0 + ux * min(length, dist + l)
            ny1 = y0 + uy * min(length, dist + l)
            if on:
                draw.line([(nx0, ny0), (nx1, ny1)], fill=fill, width=width)
            dist += l
            on = not on

    dashed_line((group_x, group_y), (group_x + group_w, group_y), width=2)
    dashed_line((group_x, group_y + group_h), (group_x + group_w, group_y + group_h), width=2)
    dashed_line((group_x, group_y), (group_x, group_y + group_h), width=2)
    dashed_line((group_x + group_w, group_y), (group_x + group_w, group_y + group_h), width=2)

    # blocks
    for x in xs:
        draw.rounded_rectangle([x, y, x + box_w, y + box_h], radius=12, outline=(0, 0, 0), width=3)

    # arrows
    cy = y + box_h // 2
    for i in range(3):
        x0 = xs[i] + box_w + 6
        x1 = xs[i + 1] - 6
        draw.line([(x0, cy), (x1, cy)], fill=(0, 0, 0), width=3)
        # head
        ah, aw = 14, 8
        draw.line([(x1, cy), (x1 - ah, cy - aw)], fill=(0, 0, 0), width=3)
        draw.line([(x1, cy), (x1 - ah, cy + aw)], fill=(0, 0, 0), width=3)

    # labels
    if with_labels:
        # pick font
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 22)
        except Exception:
            font = ImageFont.load_default()
        fallback = ["PATCH EMBED", "+ CLS + POSENC", "ENCODER xL", "CLASS HEAD"]
        for i, x in enumerate(xs):
            text = None
            if labels and i < len(labels):
                text = labels[i]
            if not text or not isinstance(text, str) or not text.strip():
                text = fallback[i % len(fallback)]
            # center text
            tw, th = draw.textlength(text, font=font), font.size + 6
            tx = x + (box_w - tw) / 2
            ty = y + (box_h - th) / 2
            draw.text((tx, ty), text, fill=(40, 40, 40), font=font, align="center")

    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def _draw_label_text(pixels: List[List[List[int]]], x: int, y: int, w: int, h: int, label: Optional[str]) -> None:
    # Render up to two lines of ASCII uppercase text inside a box using a 5x7 bitmap font
    import re
    inner_x = x + 10
    inner_y = y + 10
    inner_w = w - 20
    inner_h = h - 20
    if inner_w <= 0 or inner_h <= 0:
        return
    if not label:
        # fallback: generic placeholders
        label = "BLOCK"
    # Normalize to ASCII upper
    text = label.upper()
    # allow only A-Z, 0-9, space and '-'
    text = re.sub(r"[^A-Z0-9 \-]", " ", text)
    if not text.strip():
        text = "BLOCK"
    # simple wrap by width
    scale = 3  # enlarge 5x7 glyphs for readability
    char_w = 6 * scale  # 5px glyph +1px spacing
    max_chars = max(1, inner_w // char_w)
    words = text.split()
    lines: List[str] = []
    line = ""
    for wtok in words:
        token = wtok
        if line:
            candidate = line + " " + token
        else:
            candidate = token
        if len(candidate) <= max_chars:
            line = candidate
        else:
            if line:
                lines.append(line)
                line = token
            else:
                # force cut
                lines.append(token[:max_chars])
                line = token[max_chars:]
    if line:
        lines.append(line)
    # limit to 2 lines for clarity
    lines = lines[:2]
    # vertical centering
    total_h = len(lines) * ((7 * scale) + scale) - scale
    start_y = inner_y + (inner_h - total_h) // 2

    for i, ln in enumerate(lines):
        # center each line horizontally
        line_w = len(ln) * (6 * scale)
        start_x = inner_x + max(0, (inner_w - line_w) // 2)
        draw_text_5x7(pixels, start_x, start_y + i * ((7 * scale) + scale), ln, color=(40, 40, 40), scale=scale)


_FONT_5x7: Dict[str, List[str]] = {
    'A': ["01110","10001","10001","11111","10001","10001","10001"],
    'B': ["11110","10001","11110","10001","10001","10001","11110"],
    'C': ["01111","10000","10000","10000","10000","10000","01111"],
    'D': ["11110","10001","10001","10001","10001","10001","11110"],
    'E': ["11111","10000","11110","10000","10000","10000","11111"],
    'F': ["11111","10000","11110","10000","10000","10000","10000"],
    'G': ["01110","10000","10000","10111","10001","10001","01110"],
    'H': ["10001","10001","11111","10001","10001","10001","10001"],
    'I': ["11111","00100","00100","00100","00100","00100","11111"],
    'J': ["00111","00010","00010","00010","10010","10010","01100"],
    'K': ["10001","10010","10100","11000","10100","10010","10001"],
    'L': ["10000","10000","10000","10000","10000","10000","11111"],
    'M': ["10001","11011","10101","10101","10001","10001","10001"],
    'N': ["10001","11001","10101","10011","10001","10001","10001"],
    'O': ["01110","10001","10001","10001","10001","10001","01110"],
    'P': ["11110","10001","10001","11110","10000","10000","10000"],
    'Q': ["01110","10001","10001","10001","10101","10010","01101"],
    'R': ["11110","10001","10001","11110","10100","10010","10001"],
    'S': ["01111","10000","10000","01110","00001","00001","11110"],
    'T': ["11111","00100","00100","00100","00100","00100","00100"],
    'U': ["10001","10001","10001","10001","10001","10001","01110"],
    'V': ["10001","10001","10001","10001","01010","01010","00100"],
    'W': ["10001","10001","10001","10101","10101","11011","10001"],
    'X': ["10001","01010","00100","00100","01010","10001","10001"],
    'Y': ["10001","01010","00100","00100","00100","00100","00100"],
    'Z': ["11111","00001","00010","00100","01000","10000","11111"],
    '0': ["01110","10001","10011","10101","11001","10001","01110"],
    '1': ["00100","01100","00100","00100","00100","00100","01110"],
    '2': ["01110","10001","00001","00010","00100","01000","11111"],
    '3': ["11110","00001","00001","00110","00001","00001","11110"],
    '4': ["00010","00110","01010","10010","11111","00010","00010"],
    '5': ["11111","10000","11110","00001","00001","10001","01110"],
    '6': ["00110","01000","10000","11110","10001","10001","01110"],
    '7': ["11111","00001","00010","00100","01000","01000","01000"],
    '8': ["01110","10001","10001","01110","10001","10001","01110"],
    '9': ["01110","10001","10001","01111","00001","00010","01100"],
    '-': ["00000","00000","00000","11111","00000","00000","00000"],
    ' ': ["00000","00000","00000","00000","00000","00000","00000"],
}


def draw_text_5x7(pixels: List[List[List[int]]], x: int, y: int, text: str, color=(60, 60, 60), scale: int = 1) -> None:
    max_y = len(pixels) - 1
    max_x = len(pixels[0]) - 1 if pixels else -1

    def set_px(px: int, py: int):
        if 0 <= px <= max_x and 0 <= py <= max_y:
            pixels[py][px][0] = color[0]
            pixels[py][px][1] = color[1]
            pixels[py][px][2] = color[2]

    cx = x
    for ch in text:
        glyph = _FONT_5x7.get(ch, _FONT_5x7[' '])
        for gy, row in enumerate(glyph):
            for gx, bit in enumerate(row):
                if bit == '1':
                    # draw scaled pixel
                    for sy in range(scale):
                        for sx in range(scale):
                            set_px(cx + gx * scale + sx, y + gy * scale + sy)
        cx += 6 * scale  # 5px width + spacing


# ------------------------- Real Google GenAI calls -------------------------

def _real_gemini(kind: str, *, api_key: str, **kwargs) -> Dict[str, Any]:
    import google.generativeai as genai

    genai.configure(api_key=api_key)

    # Model names are configurable via env with safe defaults.
    # You can set GEMINI_MODEL to your provisioned model, e.g. "gemini-2.0-flash-exp" or "gemini-1.5-flash".
    text_model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    image_model_name = os.getenv("GEMINI_IMAGE_MODEL", "")  # e.g. "imagen-3.0-generate"
    image_edit_model_name = os.getenv("GEMINI_IMAGE_EDIT_MODEL", "")  # e.g. "imagen-3.0-edit"

    if kind == "parse":
        from .. import prompts as _p
        prompt = _p.build_parse_prompt()
        user_text = kwargs.get("user_text", "")
        model = genai.GenerativeModel(text_model_name)
        resp = model.generate_content([prompt, user_text])
        content = _first_text(resp)
        data = _robust_json(content)
        if not isinstance(data, dict):
            raise ValueError("parse: model did not return JSON dict")
        return {"spec": data}

    if kind == "plan":
        spec = kwargs.get("spec", {})
        from .. import prompts as _p
        prompt = _p.build_plan_prompt()
        model = genai.GenerativeModel(text_model_name)
        resp = model.generate_content([prompt, json.dumps(spec, ensure_ascii=False)])
        spec_text = _first_text(resp)
        return {"spec_text": spec_text.strip()}

    if kind == "prompt_generate":
        K = int(kwargs.get("K", 3))
        spec_text = kwargs.get("spec_text", "")
        from .. import prompts as _p
        prompt = _p.build_promptgen_prompt(K, spec_text)
        model = genai.GenerativeModel(text_model_name)
        resp = model.generate_content(prompt)
        content = _first_text(resp)
        arr = _robust_json(content)
        if not isinstance(arr, list):
            # fallback: split lines
            arr = [ln.strip("- ") for ln in content.splitlines() if ln.strip()][:K]
        return {"prompts": arr[:K]}

    if kind == "image_generate":
        prompts: List[str] = kwargs.get("prompts", [])
        outdir: str = kwargs.get("outdir", "artifacts")
        if not image_model_name:
            raise ValueError("GEMINI_IMAGE_MODEL is not set. Please set it to a valid image model (e.g., 'gemini-2.5-flash-image' or 'gemini-2.5-flash-image-preview').")
        try:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            Path(outdir).mkdir(parents=True, exist_ok=True)
            max_workers = max(1, min(len(prompts), int(_os.getenv("NNG_CONCURRENCY", "4"))))

            def _gen_one(i: int, p: str) -> str:
                # new model per thread to avoid cross-thread state issues
                mdl = genai.GenerativeModel(model_name=image_model_name)
                resp = mdl.generate_content(p, request_options={"timeout": 180})
                try:
                    (Path(outdir) / f"candidate_{i}.resp.txt").write_text(str(resp))
                except Exception:
                    pass
                img_bytes, mime = _first_image_bytes(resp)
                if not img_bytes:
                    raise ValueError("image model did not return image bytes; see *.resp.txt")
                ext = ".png" if mime == "image/png" else ".jpg"
                pth = Path(outdir) / f"candidate_{i}{ext}"
                with open(pth, "wb") as f:
                    f.write(img_bytes)
                with open(str(pth) + ".meta.json", "w", encoding="utf-8") as mf:
                    mf.write(json.dumps({"source": "gemini", "mime": mime, "bytes": len(img_bytes)}, ensure_ascii=False))
                return str(pth)

            futures = []
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                for i, p in enumerate(prompts):
                    futures.append(ex.submit(_gen_one, i, p))
            # preserve order by index
            results = [None] * len(prompts)
            for fut in as_completed(futures):
                # find index by result path name
                path = fut.result()
                stem = Path(path).stem
                try:
                    idx = int(stem.split("_")[-1])
                except Exception:
                    idx = 0
                results[idx] = path
            # fill any missing in order fallback
            paths: List[str] = [r or "" for r in results]
            return {"paths": paths}
        except Exception:
            raise

    if kind == "judge":
        image_path: str = kwargs.get("image_path")
        spec = kwargs.get("spec", {})
        model = genai.GenerativeModel(text_model_name)
        from .. import prompts as _p
        judge_prompt = _p.build_judge_prompt()
        image_part = _image_part_from_path(image_path)
        resp = model.generate_content([
            {"text": judge_prompt},
            {"text": json.dumps(spec, ensure_ascii=False)},
            image_part,
        ])
        content = _first_text(resp)
        data = _robust_json(content)
        if not isinstance(data, dict):
            raise ValueError("judge: non-JSON")
        score = float(max(0.0, min(1.0, data.get("score", 0.0))))
        violations = list(data.get("violations", []))
        return {"score": score, "violations": violations}

    if kind == "image_edit":
        image_path: str = kwargs.get("image_path")
        out_path: str = kwargs.get("out_path")
        instructions: str = kwargs.get("instructions", "")
        ref_images: List[str] = list(kwargs.get("ref_images", []) or [])
        if not image_edit_model_name:
            raise ValueError("GEMINI_IMAGE_EDIT_MODEL is not set. Please set it to a valid image edit model (e.g., 'gemini-2.5-flash-image' or 'gemini-2.5-flash-image-preview').")
        try:
            model = genai.GenerativeModel(model_name=image_edit_model_name)
            base_img = _image_part_from_path(image_path)
            from .. import prompts as _p
            parts = [{"text": _p.build_image_edit_prompt(instructions)}, base_img]
            for rp in ref_images:
                try:
                    parts.append(_image_part_from_path(rp))
                except Exception:
                    continue
            resp = model.generate_content(parts, request_options={"timeout": 120})
            try:
                out_p = Path(out_path)
                out_p.parent.mkdir(parents=True, exist_ok=True)
                (out_p.parent / (out_p.stem + ".resp.txt")).write_text(str(resp))
            except Exception:
                pass
            img_bytes, mime = _first_image_bytes(resp)
            if not img_bytes:
                raise ValueError("image edit returned no image; see *.resp.txt for raw response")
            ext = ".png" if mime == "image/png" else ".jpg"
            out_p = Path(out_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "wb") as f:
                f.write(img_bytes)
            with open(str(out_p) + ".meta.json", "w", encoding="utf-8") as mf:
                mf.write(json.dumps({"source": "gemini", "mime": mime, "bytes": len(img_bytes)}, ensure_ascii=False))
            return {"path": str(out_p)}
        except Exception as e:
            # surface error rather than fallback, per user's requirement to avoid local rendering
            raise

    if kind == "image_fuse":
        # Create a new image by composing multiple reference images under textual instructions
        out_path: str = kwargs.get("out_path")
        instructions: str = kwargs.get("instructions", "")
        ref_images: List[str] = list(kwargs.get("ref_images", []) or [])
        if not image_model_name:
            raise ValueError("GEMINI_IMAGE_MODEL is not set. Please set it to a valid image model (e.g., 'gemini-2.5-flash-image' or 'gemini-2.5-flash-image-preview').")
        try:
            model = genai.GenerativeModel(model_name=image_model_name)
            from .. import prompts as _p
            parts = [{"text": _p.build_image_fusion_prompt(instructions)}]
            for rp in ref_images:
                try:
                    parts.append(_image_part_from_path(rp))
                except Exception:
                    continue
            resp = model.generate_content(parts, request_options={"timeout": 120})
            try:
                out_p = Path(out_path)
                out_p.parent.mkdir(parents=True, exist_ok=True)
                (out_p.parent / (out_p.stem + ".resp.txt")).write_text(str(resp))
            except Exception:
                pass
            img_bytes, mime = _first_image_bytes(resp)
            if not img_bytes:
                raise ValueError("image fuse returned no image; see *.resp.txt for raw response")
            out_p = Path(out_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "wb") as f:
                f.write(img_bytes)
            with open(str(out_p) + ".meta.json", "w", encoding="utf-8") as mf:
                mf.write(json.dumps({"source": "gemini", "mime": mime, "bytes": len(img_bytes)}, ensure_ascii=False))
            return {"path": str(out_p)}
        except Exception:
            raise

    raise ValueError(f"Unsupported kind={kind}")


def _first_text(resp: Any) -> str:
    try:
        if hasattr(resp, "text"):
            return resp.text
        # Some SDK versions: candidates[0].content.parts[0].text
        cands = getattr(resp, "candidates", [])
        if cands:
            parts = getattr(cands[0], "content", None)
            if parts and getattr(parts, "parts", None):
                for part in parts.parts:
                    if getattr(part, "text", None):
                        return part.text
        return str(resp)
    except Exception:
        return str(resp)


def _first_image_bytes(resp: Any) -> tuple[bytes | None, str]:
    # Try to walk through content parts and return first inline image bytes
    try:
        # Newer SDK: resp.candidates[].content.parts[].inline_data
        cands = getattr(resp, "candidates", [])
        for c in cands or []:
            content = getattr(c, "content", None)
            parts = getattr(content, "parts", None) if content else None
            for part in parts or []:
                inline = getattr(part, "inline_data", None)
                if inline and getattr(inline, "data", None):
                    data = inline.data
                    mime = getattr(inline, "mime_type", "image/png")
                    if isinstance(data, bytes):
                        return data, mime
                    # some versions may base64-encode
                    try:
                        return base64.b64decode(data), mime
                    except Exception:
                        pass
        return None, ""
    except Exception:
        return None, ""


def _image_part_from_path(path: str) -> Dict[str, Any]:
    # google-generativeai accepts dict with mime_type and data bytes for images
    p = Path(path)
    mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
    data = p.read_bytes()
    return {"mime_type": mime, "data": data}


def _robust_json(text: str) -> Any:
    # Try parse whole, then attempt to extract first {...} or [...] block
    try:
        return json.loads(text)
    except Exception:
        pass
    # crude extraction
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            pass
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            pass
    return {}
