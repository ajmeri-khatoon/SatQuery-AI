"""Small-sample VRSBench evaluation for the existing Qwen3-VL pipeline."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from itertools import chain
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DATASETS_ROOT = PROJECT_ROOT / "datasets"
if str(DATASETS_ROOT) not in sys.path:
    sys.path.insert(0, str(DATASETS_ROOT))

from vrsbench.loader import VRSBenchLoader, VRSBenchSample
from inference.model import Qwen3VLInference


def normalize_text(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", value.lower()).split())


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_text(prediction) == normalize_text(reference))


def _tokens(value: str) -> list[str]:
    return normalize_text(value).split()


def rouge_l_f1(prediction: str, reference: str) -> float:
    """Compute lightweight ROUGE-L F1 for one generated/reference caption."""
    predicted = _tokens(prediction)
    expected = _tokens(reference)
    if not predicted or not expected:
        return 0.0
    row = [0] * (len(expected) + 1)
    for predicted_token in predicted:
        previous = 0
        for index, expected_token in enumerate(expected, start=1):
            saved = row[index]
            if predicted_token == expected_token:
                row[index] = previous + 1
            else:
                row[index] = max(row[index], row[index - 1])
            previous = saved
    lcs = row[-1]
    precision = lcs / len(predicted)
    recall = lcs / len(expected)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def box_iou(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def _iter_boxes(prediction: str) -> Iterable[tuple[float, float, float, float]]:
    parsed = json.loads(prediction)
    for box in parsed.get("bounding_boxes", []):
        values = tuple(float(box[key]) for key in ("x1", "y1", "x2", "y2"))
        if 0 <= values[0] < values[2] <= 1 and 0 <= values[1] < values[3] <= 1:
            yield values


def evaluate_samples(
    inference: Qwen3VLInference,
    samples: Iterable[VRSBenchSample],
    task: str,
) -> dict[str, float | int]:
    counts = {"samples": 0, "evaluated": 0, "skipped": 0, "vqa": 0, "caption": 0, "grounding": 0}
    totals = {"vqa": 0.0, "caption": 0.0, "grounding": 0.0}

    for sample in samples:
        counts["samples"] += 1
        if sample.image_path is None:
            counts["skipped"] += 1
            print(f"[VRSBench] Skipping {sample.sample_id}: image path unavailable.", file=sys.stderr)
            continue
        sample_evaluated = False
        try:
            if task in {"all", "vqa"}:
                for annotation in sample.vqa:
                    totals["vqa"] += exact_match(
                        inference.answer_question(str(sample.image_path), annotation.question),
                        annotation.answer,
                    )
                    counts["vqa"] += 1
                    sample_evaluated = True
            if task in {"all", "caption"} and sample.caption:
                totals["caption"] += rouge_l_f1(
                    inference.caption_image(str(sample.image_path)), sample.caption
                )
                counts["caption"] += 1
                sample_evaluated = True
            if task in {"all", "grounding"}:
                for annotation in sample.grounding:
                    if annotation.bounding_box is None:
                        continue
                    prediction = inference.ground_object(
                        str(sample.image_path), annotation.referring_sentence
                    )
                    predicted_boxes = list(_iter_boxes(prediction))
                    best_iou = max(
                        (box_iou(box, annotation.bounding_box) for box in predicted_boxes),
                        default=0.0,
                    )
                    totals["grounding"] += best_iou
                    counts["grounding"] += 1
                    sample_evaluated = True
        except Exception as error:
            counts["skipped"] += 1
            print(f"[VRSBench] Skipping {sample.sample_id}: {error}", file=sys.stderr)
            continue
        if sample_evaluated:
            counts["evaluated"] += 1

    result: dict[str, float | int] = dict(counts)
    for metric in totals:
        count = counts[metric]
        result[f"{metric}_score"] = totals[metric] / count if count else 0.0
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Qwen3-VL on a bounded VRSBench sample.")
    parser.add_argument("--split", default="test", help="Published dataset split to read, such as train.")
    parser.add_argument("--limit", type=int, default=10, help="Maximum source samples to inspect.")
    parser.add_argument("--task", choices=("all", "vqa", "caption", "grounding"), default="all")
    parser.add_argument("--data-root", default="datasets/vrsbench", help="Local VRSBench root/cache directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.limit < 1:
        print("--limit must be at least 1.", file=sys.stderr)
        return 2
    try:
        loader = VRSBenchLoader(args.data_root)
        samples = iter(loader.iter_samples(args.split, limit=args.limit))
        first_sample = next(samples, None)
        if first_sample is None:
            print("No usable VRSBench samples were found.", file=sys.stderr)
            return 1
        samples = chain((first_sample,), samples)
        inference = Qwen3VLInference()
        results = evaluate_samples(inference, samples, args.task)
    except Exception as error:
        print(f"VRSBench evaluation failed: {error}", file=sys.stderr)
        return 1

    print(f"VRSBench split: {args.split}")
    print(f"Requested samples: {args.limit}")
    print(f"Source samples inspected: {results['samples']}")
    print(f"Samples evaluated: {results['evaluated']}")
    print(f"Samples skipped: {results['skipped']}")
    print(f"VQA records: {results['vqa']} | normalized exact match: {results['vqa_score']:.3f}")
    print(f"Caption records: {results['caption']} | ROUGE-L F1: {results['caption_score']:.3f}")
    print(f"Grounding records: {results['grounding']} | mean best IoU: {results['grounding_score']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
