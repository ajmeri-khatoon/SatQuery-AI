"""Command-line entry point for Qwen3-VL image question answering."""

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask Qwen3-VL a question about an image.")
    parser.add_argument("--image", required=True, help="Path to the input image.")
    output_mode = parser.add_mutually_exclusive_group(required=True)
    output_mode.add_argument("--question", help="Question to ask about the image.")
    output_mode.add_argument(
        "--caption",
        action="store_true",
        help="Generate a satellite image caption.",
    )
    output_mode.add_argument(
        "--features",
        action="store_true",
        help="Analyze visible satellite objects and land-use features.",
    )
    output_mode.add_argument(
        "--ground",
        help="Estimate a normalized bounding box for a requested object or feature.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run VQA, captioning, feature analysis, and grounding in one JSON result.",
    )
    parser.add_argument(
        "--fast-all",
        action="store_true",
        help="Run all capabilities with one compact model generation.",
    )
    parser.add_argument(
        "--structured",
        action="store_true",
        help="Return the selected inference result in unified JSON format.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.all and not args.question:
        print("Error: --all requires --question.", file=sys.stderr)
        return 2
    if args.fast_all and not args.question:
        print("Error: --fast-all requires --question.", file=sys.stderr)
        return 2
    if args.all and args.fast_all:
        print("Error: choose either --all or --fast-all.", file=sys.stderr)
        return 2
    if not args.all and not (args.question or args.caption or args.features or args.ground):
        print("Error: choose --question, --caption, --features, or --ground.", file=sys.stderr)
        return 2

    if not Path(args.image).is_file():
        print(f"Error: Image not found: {args.image}", file=sys.stderr)
        return 1

    try:
        from model import Qwen3VLInference
        from structured_output import (
            from_caption,
            from_features,
            from_all,
            fast_all,
            from_grounding,
            from_vqa,
        )

        Qwen3VLInference.validate_image(args.image)
        inference = Qwen3VLInference()
        if args.all:
            structured_result = from_all(inference, args.image, args.question)
            print(structured_result.to_json())
            return 0
        if args.fast_all:
            print(fast_all(inference, args.image, args.question).to_json())
            return 0
        if args.caption:
            answer = inference.caption_image(args.image)
            structured_result = from_caption(answer)
        elif args.features:
            answer = inference.analyze_features(args.image)
            structured_result = from_features(answer)
        elif args.ground is not None:
            answer = inference.ground_object(args.image, args.ground)
            structured_result = from_grounding(answer)
        else:
            answer = inference.answer_question(args.image, args.question)
            structured_result = from_vqa(answer)
    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except ImportError as error:
        print(
            "Error: Required packages are not installed. "
            "Run 'pip install -r requirements.txt' first. "
            f"Details: {error}",
            file=sys.stderr,
        )
        return 1
    except RuntimeError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    if args.structured:
        print(structured_result.to_json())
        return 0

    print(f"Image path: {args.image}")
    if args.caption:
        print("Satellite Image Caption:")
        print(answer)
    elif args.features:
        print("Satellite Object/Feature Analysis:")
        print(answer)
    elif args.ground is not None:
        print("Satellite Visual Grounding:")
        print(answer)
    else:
        print(f"Question: {args.question}")
        print(f"Answer: {answer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())