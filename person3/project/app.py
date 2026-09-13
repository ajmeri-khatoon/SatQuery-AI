"""Gradio web interface for REMOTE-SENSING VISION AI using Qwen3-VL fast-all."""

from __future__ import annotations

from pathlib import Path
import sys
import threading
from typing import Any

import gradio as gr

INFERENCE_ROOT = Path(__file__).resolve().parent / "inference"
if str(INFERENCE_ROOT) not in sys.path:
    sys.path.insert(0, str(INFERENCE_ROOT))

from model import Qwen3VLInference
from structured_output import CompactStructuredResult, fast_all


DEFAULT_QUESTION = "What objects are visible in this satellite image?"
ANALYSIS_LOCK = threading.Lock()
_MODEL_INSTANCE: Qwen3VLInference | None = None
_MODEL_LOCK = threading.Lock()


def get_inference_backend() -> Qwen3VLInference:
    """Return the singleton Qwen3-VL inference backend, loaded only once."""
    global _MODEL_INSTANCE
    with _MODEL_LOCK:
        if _MODEL_INSTANCE is None:
            print("[REMOTE-SENSING VISION AI] Loading Qwen3-VL model (one-time load)...", flush=True)
            _MODEL_INSTANCE = Qwen3VLInference()
            print("[REMOTE-SENSING VISION AI] Model initialized and ready.", flush=True)
        return _MODEL_INSTANCE


def _format_results(
    result: CompactStructuredResult | None,
    status_message: str,
) -> tuple[str, str, str, list[list[str]], str, str, dict[str, Any], str]:
    """Convert a CompactStructuredResult into clean, readable display values."""
    if result is None:
        return (
            "",
            "",
            "",
            [],
            "null (not provided)",
            "None",
            {},
            status_message,
        )

    payload = result.to_dict()

    answer = str(payload.get("answer", "") or "(No answer produced)")
    caption = str(payload.get("caption", "") or "(No caption produced)")

    raw_objects = payload.get("detected_objects", [])
    if isinstance(raw_objects, list) and raw_objects:
        detected_objects_str = ", ".join(str(obj) for obj in raw_objects)
    elif raw_objects:
        detected_objects_str = str(raw_objects)
    else:
        detected_objects_str = "(No objects detected)"

    evidence_rows: list[list[str]] = []
    raw_evidence = payload.get("evidence", [])
    if isinstance(raw_evidence, list):
        for item in raw_evidence:
            if isinstance(item, dict):
                evidence_rows.append([
                    str(item.get("object", "")),
                    str(item.get("location", "")),
                    str(item.get("description", item.get("evidence", ""))),
                ])
            elif isinstance(item, str):
                evidence_rows.append(["", "", item])

    confidence = payload.get("confidence")
    confidence_str = str(confidence) if confidence is not None else "null (not estimated by model)"

    errors = payload.get("errors", [])
    if errors:
        errors_str = "\n".join(f"- {err}" for err in errors)
    else:
        errors_str = "None"

    return (
        answer,
        caption,
        detected_objects_str,
        evidence_rows,
        confidence_str,
        errors_str,
        payload,
        status_message,
    )


def analyze_satellite_image(
    image: str | Path | None,
    question: str,
    inference: Qwen3VLInference,
    progress: gr.Progress = gr.Progress(),
) -> tuple[str, str, str, list[list[str]], str, str, dict[str, Any], str]:
    """Execute the fast-all inference pipeline against an uploaded image."""
    if not image:
        return _format_results(
            None,
            "**Status:** Error: Please upload an image first.",
        )

    clean_question = (question or "").strip()
    if not clean_question:
        return _format_results(
            None,
            "**Status:** Error: Please enter a question before analyzing.",
        )

    image_path = str(image)
    try:
        progress(0.1, desc="Validating input image...")
        Qwen3VLInference.validate_image(image_path)
    except (FileNotFoundError, ValueError) as error:
        return _format_results(
            None,
            f"**Status:** Error: Invalid image file: {error}",
        )

    try:
        progress(0.3, desc="Analyzing image with Qwen3-VL (fast-all)...")
        with ANALYSIS_LOCK:
            result = fast_all(inference, image_path, clean_question)
        progress(1.0, desc="Complete")

        if result.errors:
            status = "**Status:** Complete (with warnings/errors)"
        else:
            status = "**Status:** Complete"

        return _format_results(result, status)
    except Exception as error:
        return _format_results(
            None,
            f"**Status:** Error: Inference failed: {error}",
        )


def build_demo(inference: Qwen3VLInference) -> gr.Blocks:
    """Build the Gradio UI Blocks application reusing the provided inference instance."""
    custom_css = """
    .gradio-container { max-width: 1400px !important; margin: 0 auto; }
    .hero-banner { padding: 12px 0 16px 0; border-bottom: 1px solid #e5e7eb; margin-bottom: 20px; }
    .hero-title { font-size: 1.75rem; font-weight: 700; margin: 0; }
    .hero-desc { font-size: 0.95rem; color: #6b7280; margin-top: 4px; }
    .status-text { font-size: 0.95rem; margin-top: 8px; }
    """

    with gr.Blocks(title="REMOTE-SENSING VISION AI") as demo:
        with gr.Row(elem_classes="hero-banner"):
            with gr.Column():
                gr.Markdown(
                    "# REMOTE-SENSING VISION AI\n"
                    "High-speed satellite and aerial remote-sensing image question answering with Qwen3-VL.",
                    elem_classes="hero-title",
                )
                status_display = gr.Markdown(
                    "**Status:** Ready — Upload an image and click Analyze.",
                    elem_classes="status-text",
                )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1. Input")
                image_input = gr.Image(
                    label="Satellite / Aerial Image",
                    type="filepath",
                    sources=["upload", "clipboard"],
                )
                question_input = gr.Textbox(
                    label="Question",
                    value=DEFAULT_QUESTION,
                    placeholder="Enter your question about the satellite image...",
                    lines=2,
                )
                analyze_button = gr.Button(
                    "Analyze Satellite Image",
                    variant="primary",
                    size="lg",
                )

                example_image = Path("datasets/test_satellite.jpg")
                if example_image.is_file():
                    gr.Examples(
                        examples=[
                            [str(example_image), DEFAULT_QUESTION],
                            [str(example_image), "What type of land use is visible in this scene?"],
                            [str(example_image), "Are there any waterways or ports visible?"],
                        ],
                        inputs=[image_input, question_input],
                        label="Example Inputs",
                    )

            with gr.Column(scale=1):
                gr.Markdown("### 2. Analysis Results")
                answer_output = gr.Textbox(
                    label="Answer",
                    placeholder="The AI's answer will appear here...",
                    lines=2,
                    interactive=False,
                )
                caption_output = gr.Textbox(
                    label="Satellite Image Caption",
                    placeholder="Image description will appear here...",
                    lines=3,
                    interactive=False,
                )
                objects_output = gr.Textbox(
                    label="Detected Objects",
                    placeholder="Identified objects will appear here...",
                    lines=2,
                    interactive=False,
                )
                evidence_output = gr.Dataframe(
                    headers=["Object", "Location", "Description"],
                    datatype=["str", "str", "str"],
                    label="Visual Evidence",
                    interactive=False,
                )
                with gr.Row():
                    confidence_output = gr.Textbox(
                        label="Confidence",
                        value="null (not estimated by model)",
                        interactive=False,
                    )
                    errors_output = gr.Textbox(
                        label="Errors / Warnings",
                        value="None",
                        interactive=False,
                    )

        with gr.Row():
            with gr.Column():
                with gr.Accordion("Raw JSON", open=False):
                    raw_json_output = gr.JSON(
                        label="Complete Structured Output",
                    )

        outputs = [
            answer_output,
            caption_output,
            objects_output,
            evidence_output,
            confidence_output,
            errors_output,
            raw_json_output,
            status_display,
        ]

        def on_click(
            image: str | None,
            question: str,
            progress: gr.Progress = gr.Progress(),
        ):
            return analyze_satellite_image(
                image=image,
                question=question,
                inference=inference,
                progress=progress,
            )

        analyze_button.click(
            fn=on_click,
            inputs=[image_input, question_input],
            outputs=outputs,
        )

    return demo


def main() -> None:
    """Entry point: pre-load Qwen3-VL once and launch the Gradio server."""
    print("=" * 60, flush=True)
    print("Starting REMOTE-SENSING VISION AI Web Interface...", flush=True)
    print("=" * 60, flush=True)

    # Load the model exactly once when the application starts
    inference = get_inference_backend()

    demo = build_demo(inference)
    print("[REMOTE-SENSING VISION AI] Launching web server at http://127.0.0.1:7860...", flush=True)
    custom_css = """
    .gradio-container { max-width: 1400px !important; margin: 0 auto; }
    .hero-banner { padding: 12px 0 16px 0; border-bottom: 1px solid #e5e7eb; margin-bottom: 20px; }
    .hero-title { font-size: 1.75rem; font-weight: 700; margin: 0; }
    .hero-desc { font-size: 0.95rem; color: #6b7280; margin-top: 4px; }
    .status-text { font-size: 0.95rem; margin-top: 8px; }
    """
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        css=custom_css,
        share=False,
    )


if __name__ == "__main__":
    main()
