from __future__ import annotations

# Hugging Face Spaces entrypoint for Gradio
# Exposes a global `demo` variable that HF will serve.

from scripts.gradio_app import app as create_app

demo = create_app()

