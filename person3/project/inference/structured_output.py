"""Validated unified JSON output for the existing inference modes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
import time
from typing import Any


@dataclass(frozen=True)
class BoundingBox:
    """An approximate normalized bounding box."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        coordinates = (self.x1, self.y1, self.x2, self.y2)
        if any(not isinstance(value, (int, float)) or isinstance(value, bool) for value in coordinates):
            raise ValueError("Bounding-box coordinates must be numeric.")
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in coordinates):
            raise ValueError("Bounding-box coordinates must be between 0 and 1.")
        if self.x1 >= self.x2 or self.y1 >= self.y2:
            raise ValueError("Bounding boxes must have positive area.")


@dataclass
class StructuredResult:
    """Unified result shared by VQA, captioning, features, and grounding."""

    answer: str = ""
    caption: str = ""
    detected_objects: list[dict[str, Any]] = field(default_factory=list)
    grounding: list[dict[str, Any]] = field(default_factory=list)
    evidence: list[str | dict[str, str | None]] = field(default_factory=list)
    confidence: None = None
    errors: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.detected_objects = list(self.detected_objects or [])
        self.grounding = _validate_grounding_entries(self.grounding)
        self.evidence = [
            item
            for item in self.evidence
            if (isinstance(item, str) and item.strip())
            or (isinstance(item, dict) and item.get("description", "").strip())
        ]
        self.errors = [item for item in self.errors if isinstance(item, str) and item.strip()]
        self.confidence = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


FAST_ALL_MAX_NEW_TOKENS = 128


class CompactStructuredResult(StructuredResult):
    """Structured fast-all result serialized without grounding output."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "caption": self.caption,
            "detected_objects": self.detected_objects,
            "evidence": self.evidence,
            "confidence": None,
            "errors": self.errors,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


def _parse_fast_all_response(value: str) -> CompactStructuredResult:
    """Parse a compact fast-all response without inventing missing fields."""
    cleaned = value.replace("```json", "").replace("```", "").strip()
    decoder = json.JSONDecoder()
    json_candidates = []
    cursor = 0
    while True:
        json_start = cleaned.find("{", cursor)
        if json_start == -1:
            break
        json_candidates.append(cleaned[json_start:])
        cursor = json_start + 1

    if not json_candidates:
        raise ValueError("Fast all response did not contain a JSON object.")

    parsed = None
    parse_error = None
    for candidate in json_candidates:
        try:
            parsed, _ = decoder.raw_decode(candidate)
            break
        except json.JSONDecodeError as error:
            parse_error = error
            continue

    if parsed is None:
        raise ValueError(
            "Fast all response was incomplete or malformed JSON; "
            "the model output may have been truncated."
        ) from parse_error
    if not isinstance(parsed, dict):
        raise ValueError("Fast all response must be a JSON object.")

    required_fields = (
        "answer",
        "caption",
        "detected_objects",
        "evidence",
        "confidence",
        "errors",
    )
    missing_fields = [field for field in required_fields if field not in parsed]
    if missing_fields:
        raise ValueError(
            "Fast all response is missing required fields: "
            + ", ".join(missing_fields)
            + "."
        )

    answer = parsed["answer"]
    caption = parsed["caption"]
    objects = parsed["detected_objects"]
    evidence = parsed["evidence"]
    confidence = parsed["confidence"]
    errors = parsed["errors"]
    if not isinstance(answer, str) or not isinstance(caption, str):
        raise ValueError("Fast all answer and caption must be strings.")
    if not isinstance(objects, list) or any(not isinstance(item, str) for item in objects):
        raise ValueError("Fast all detected_objects must be a list of strings.")
    if isinstance(evidence, dict):
        evidence = [evidence]
    elif not isinstance(evidence, list):
        raise ValueError("Fast all evidence must be a list.")

    normalized_evidence = []
    for item in evidence:
        if isinstance(item, dict):
            obj = item.get("object")
            loc = item.get("location")
            desc = item.get("description") or item.get("evidence")
            if (
                isinstance(obj, str) and obj.strip()
                and isinstance(loc, str) and loc.strip()
                and isinstance(desc, str) and desc.strip()
            ):
                normalized_evidence.append(
                    {"object": obj.strip(), "location": loc.strip(), "description": desc.strip()}
                )
                continue
        raise ValueError("Fast all evidence entries require object, location, and description.")
    evidence = normalized_evidence

    if confidence is not None:
        raise ValueError("Fast all confidence must be null.")
    if not isinstance(errors, list) or any(not isinstance(item, str) for item in errors):
        raise ValueError("Fast all errors must be a list of strings.")

    return CompactStructuredResult(
        answer=answer,
        caption=caption,
        detected_objects=objects,
        evidence=evidence,
        errors=errors,
    )


def fast_all(inference: Any, image_path: str, question: str) -> CompactStructuredResult:
    """Run a compact one-shot fast demo response with no grounding generation."""
    prompt = (
        "Analyze this satellite image and answer the user's question. "
        "Return ONLY compact JSON with this schema: {"
        "\"answer\": \"short answer\", "
        "\"caption\": \"short caption\", "
        "\"detected_objects\": [\"object1\", \"object2\"], "
        "\"evidence\": [{\"object\": \"object1\", \"location\": \"center\", \"description\": \"short evidence\"}], "
        "\"confidence\": null, "
        "\"errors\": []}. "
        "At most 5 objects. Keep answer, caption, and descriptions very short. "
        "Do not invent info. No coordinates or bounding boxes. confidence=null. errors=[]. "
        f"User question: {json.dumps(question)}"
    )
    try:
        inference.validate_image(image_path)
        if hasattr(inference, "fast_all_answer"):
            response = inference.fast_all_answer(image_path, question)
        else:
            response = inference._generate_text(
                image_path,
                prompt,
                max_new_tokens=FAST_ALL_MAX_NEW_TOKENS,
                max_image_side=512,
            )
        result = _parse_fast_all_response(response)
    except Exception as error:
        result = CompactStructuredResult(
            errors=[f"Fast all analysis failed: {error}"]
        )
    return result


def _validate_grounding_entries(entries: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    validated = []
    for entry in entries or []:
        if not isinstance(entry, dict):
            raise ValueError("Grounding entries must be JSON objects.")
        normalized = dict(entry)
        boxes = normalized.get("bounding_boxes", [])
        if boxes is None:
            boxes = []
        if not isinstance(boxes, list):
            raise ValueError("Grounding bounding_boxes must be a list.")
        normalized["bounding_boxes"] = [
            asdict(BoundingBox(**box)) if isinstance(box, dict) else _reject_box()
            for box in boxes
        ]
        validated.append(normalized)
    return validated


def _reject_box() -> dict[str, float]:
    raise ValueError("Grounding bounding boxes must be JSON objects.")


def _parse_json_object(value: str, label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} was not valid JSON: {error}") from error
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must be a JSON object.")
    return parsed


def from_vqa(answer: str) -> StructuredResult:
    return StructuredResult(answer=answer)


def from_caption(caption: str) -> StructuredResult:
    return StructuredResult(caption=caption)


def from_features(value: str) -> StructuredResult:
    parsed = _parse_json_object(value, "Feature analysis")
    objects = parsed.get("features", [])
    if not isinstance(objects, list):
        raise ValueError("Feature analysis 'features' must be a list.")
    evidence = [
        feature["evidence"]
        for feature in objects
        if isinstance(feature, dict) and isinstance(feature.get("evidence"), str)
    ]
    return StructuredResult(detected_objects=objects, evidence=evidence)


def from_grounding(value: str) -> StructuredResult:
    parsed = _parse_json_object(value, "Grounding result")
    grounding = [parsed]
    evidence = [parsed["evidence"]] if isinstance(parsed.get("evidence"), str) else []
    return StructuredResult(grounding=grounding, evidence=evidence)


def _feature_objects_and_evidence(value: str) -> tuple[list[str], list[dict[str, str]]]:
    parsed = _parse_json_object(value, "Feature analysis")
    features = parsed.get("features", [])
    if not isinstance(features, list):
        raise ValueError("Feature analysis 'features' must be a list.")

    objects = []
    evidence = []
    for feature in features:
        if not isinstance(feature, dict):
            raise ValueError("Feature analysis entries must be JSON objects.")
        object_name = feature.get("object")
        location = feature.get("location")
        description = feature.get("evidence")
        if not all(isinstance(item, str) and item.strip() for item in (object_name, location, description)):
            raise ValueError("Feature analysis entries require object, location, and evidence strings.")
        if object_name not in objects:
            objects.append(object_name)
        evidence.append(
            {
                "object": object_name,
                "location": location,
                "description": description,
            }
        )
    return objects, evidence


def from_all(
    inference: Any,
    image_path: str,
    question: str,
    max_groundings: int = 5,
) -> StructuredResult:
    """Run all existing capabilities once through one already-loaded model."""
    result = StructuredResult()

    try:
        result.answer = inference.answer_question(image_path, question)
    except Exception as error:
        result.errors.append(f"VQA generation failed: {error}")

    try:
        result.caption = inference.caption_image(image_path)
    except Exception as error:
        result.errors.append(f"Caption generation failed: {error}")

    try:
        feature_response = inference.analyze_features(image_path)
        result.detected_objects, result.evidence = _feature_objects_and_evidence(feature_response)
    except Exception as error:
        result.errors.append(f"Feature analysis failed: {error}")

    for object_name in result.detected_objects[:max_groundings]:
        try:
            grounding_response = inference.ground_object(image_path, object_name)
            grounding = from_grounding(grounding_response)
            result.grounding.extend(grounding.grounding)
            result.evidence.extend(
                {
                    "object": object_name,
                    "location": entry.get("location"),
                    "description": entry.get("evidence", ""),
                }
                for entry in grounding.grounding
            )
        except Exception as error:
            result.grounding.append(
                {
                    "object": object_name,
                    "found": False,
                    "bounding_boxes": [],
                    "location": None,
                    "evidence": "The object could not be grounded reliably.",
                }
            )
            result.errors.append(f"Grounding failed for '{object_name}': {error}")

    return result
