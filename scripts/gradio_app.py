from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import List, Tuple

import gradio as gr

from app.graph import run_pipeline, run_fusion_pipeline
from app.state import AppState


def _zip_outdir(outdir: str) -> str:
    out = Path(outdir)
    if not out.exists():
        return ""
    zip_path = str(out) + ".zip"
    # remove if exists
    try:
        if Path(zip_path).exists():
            Path(zip_path).unlink()
    except Exception:
        pass
    shutil.make_archive(str(out), "zip", root_dir=str(out))
    return zip_path


def run_text_mode(user_text: str, K: int, T: int, make_zip: bool) -> Tuple[str, List[str], str, str]:
    state: AppState = {"K": int(K), "T": int(T), "user_text": user_text or "", "outdir": ""}
    final_state = run_pipeline(state)
    outdir = final_state["outdir"]
    # Collect candidates if present
    candidates = [im.path for im in (final_state.get("images") or [])]
    final_img = str(Path(outdir) / "final.png")
    zip_path = _zip_outdir(outdir) if make_zip else ""
    return final_img, candidates, outdir, zip_path


def run_image_mode(base_image, ref_images, instructions: str, K: int, make_zip: bool) -> Tuple[str, List[str], str, str]:
    state: AppState = {"K": int(K), "T": 0, "outdir": "", "instructions": instructions or ""}
    if base_image is not None:
        state["base_image"] = base_image if isinstance(base_image, str) else base_image.name
    refs: List[str] = []
    for f in (ref_images or []):
        p = f if isinstance(f, str) else getattr(f, "name", None)
        if p:
            refs.append(p)
    state["ref_images"] = refs

    final_state = run_fusion_pipeline(state)
    outdir = final_state["outdir"]
    candidates = [im.path for im in (final_state.get("images") or [])]
    final_img = str(Path(outdir) / "final.png")
    zip_path = _zip_outdir(outdir) if make_zip else ""
    return final_img, candidates, outdir, zip_path


def app() -> gr.Blocks:
    with gr.Blocks(title="NNGen — Gemini 2.5 Flash Image") as demo:
        gr.Markdown("""
        # NNGen — Gemini 2.5 Flash Image
        - Text mode: enter a natural language spec to generate a diagram (G1/G2/judge/edit).
        - Image mode: edit/fuse images with textual instructions (e.g., replace UNet with Transformer).
        - Offline works with placeholders if no `GEMINI_API_KEY` is set. With an API key, set `GEMINI_IMAGE_MODEL` and `GEMINI_IMAGE_EDIT_MODEL`.
        """)

        with gr.Tab("Text Mode"):
            user_text = gr.Textbox(label="NN spec (text)", lines=10, placeholder="Describe the architecture... e.g., Transformer encoder-decoder with cross-attention...")
            with gr.Row():
                K = gr.Slider(1, 6, value=4, step=1, label="K candidates")
                T = gr.Slider(0, 3, value=1, step=1, label="Max edit rounds (T)")
                zip_output = gr.Checkbox(value=False, label="Zip outputs")
            run_btn = gr.Button("Generate")
            final_img = gr.Image(label="final.png", type="filepath")
            gallery = gr.Gallery(label="Candidates").style(grid=4)
            outdir = gr.Textbox(label="Artifacts directory", interactive=False)
            zip_file = gr.File(label="Download run.zip", interactive=False)

            run_btn.click(run_text_mode, inputs=[user_text, K, T, zip_output], outputs=[final_img, gallery, outdir, zip_file])

        with gr.Tab("Image Mode (Fusion/Edit)"):
            base = gr.Image(label="Base image (optional)", type="filepath")
            refs = gr.Files(label="Reference images (0..N)")
            instr = gr.Textbox(label="Instructions", lines=4, placeholder="Replace the UNet backbone with a Transformer (DiT); keep layout, fonts, colors, arrows, and dashed groups unchanged.")
            with gr.Row():
                K2 = gr.Slider(1, 6, value=4, step=1, label="K candidates")
                zip_output2 = gr.Checkbox(value=False, label="Zip outputs")
            run_btn2 = gr.Button("Compose / Edit")
            final_img2 = gr.Image(label="final.png", type="filepath")
            gallery2 = gr.Gallery(label="Fused Candidates").style(grid=4)
            outdir2 = gr.Textbox(label="Artifacts directory", interactive=False)
            zip_file2 = gr.File(label="Download run.zip", interactive=False)

            run_btn2.click(run_image_mode, inputs=[base, refs, instr, K2, zip_output2], outputs=[final_img2, gallery2, outdir2, zip_file2])

    return demo


if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    app().launch(server_name="0.0.0.0", server_port=port)

