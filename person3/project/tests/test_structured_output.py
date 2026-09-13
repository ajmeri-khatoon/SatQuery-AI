"""Tests for unified structured inference output."""

import json

import pytest

from inference.structured_output import (
    BoundingBox,
    StructuredResult,
    from_features,
    from_grounding,
    from_all,
    fast_all,
    from_vqa,
)


def test_valid_structured_output() -> None:
    result = StructuredResult(
        answer="Two fields are visible.",
        detected_objects=[{"object": "field", "location": "center", "evidence": "Rows are visible."}],
        grounding=[
            {
                "object": "field",
                "found": True,
                "bounding_boxes": [{"x1": 0.1, "y1": 0.2, "x2": 0.6, "y2": 0.8}],
                "location": "center",
                "evidence": "Rows are visible.",
            }
        ],
        evidence=["Rows are visible."],
    )

    payload = result.to_dict()
    assert payload["confidence"] is None
    assert payload["grounding"][0]["bounding_boxes"][0]["x2"] == 0.6


def test_missing_optional_fields_use_empty_values() -> None:
    payload = StructuredResult().to_dict()

    assert payload == {
        "answer": "",
        "caption": "",
        "detected_objects": [],
        "grounding": [],
        "evidence": [],
        "confidence": None,
        "errors": [],
    }


def test_invalid_bounding_boxes_are_rejected() -> None:
    with pytest.raises(ValueError):
        BoundingBox(0.8, 0.2, 0.4, 0.9)
    with pytest.raises(ValueError):
        StructuredResult(grounding=[{"bounding_boxes": [{"x1": -0.1, "y1": 0.0, "x2": 0.5, "y2": 0.5}]}])


def test_json_serialization_and_existing_adapters() -> None:
    result = from_vqa("A harbor is visible.")
    assert json.loads(result.to_json())["answer"] == "A harbor is visible."
    assert from_features(
        '{"features": [{"object": "harbor", "evidence": "Ships are visible."}]}'
    ).detected_objects[0]["object"] == "harbor"
    assert from_grounding(
        '{"object": "harbor", "found": false, "bounding_boxes": [], "evidence": "Not clear."}'
    ).grounding[0]["found"] is False


class FakeInference:
    def answer_question(self, image_path: str, question: str) -> str:
        return "A harbor is visible."

    def caption_image(self, image_path: str) -> str:
        return "A harbor and vegetation are visible."

    def analyze_features(self, image_path: str) -> str:
        return json.dumps(
            {
                "features": [
                    {"object": "harbor", "location": "center", "evidence": "Ships are visible."},
                    {"object": "vegetation", "location": "lower-left", "evidence": "Green areas are visible."},
                ]
            }
        )

    def ground_object(self, image_path: str, object_name: str) -> str:
        return json.dumps(
            {
                "object": object_name,
                "found": True,
                "bounding_boxes": [{"x1": 0.1, "y1": 0.2, "x2": 0.4, "y2": 0.5}],
                "location": "center",
                "evidence": "The feature is visible.",
            }
        )


def test_all_mode_unifies_results_and_limits_grounding() -> None:
    result = from_all(FakeInference(), "image.jpg", "What is visible?")
    payload = result.to_dict()

    assert payload["answer"] == "A harbor is visible."
    assert payload["caption"]
    assert payload["detected_objects"] == ["harbor", "vegetation"]
    assert len(payload["grounding"]) == 2
    assert payload["grounding"][0]["bounding_boxes"][0]["x2"] == 0.4
    assert payload["errors"] == []


def test_all_mode_keeps_defaults_when_operations_fail() -> None:
    class FailingInference(FakeInference):
        def caption_image(self, image_path: str) -> str:
            raise RuntimeError("caption unavailable")

        def analyze_features(self, image_path: str) -> str:
            raise ValueError("malformed feature JSON")

    result = from_all(FailingInference(), "image.jpg", "What is visible?")

    assert result.answer == "A harbor is visible."
    assert result.caption == ""
    assert result.detected_objects == []
    assert result.grounding == []
    assert len(result.errors) == 2


class FastFakeInference:
    def __init__(self, response: str) -> None:
        self.response = response
        self.generation_count = 0

    @staticmethod
    def validate_image(image_path: str) -> None:
        return None

    def _generate_text(self, image_path: str, prompt: str, max_new_tokens: int, max_image_side: int = 768) -> str:
        self.generation_count += 1
        assert max_new_tokens == 128
        assert max_image_side == 512
        return self.response


def test_fast_all_uses_one_generation_and_parses_compact_json() -> None:
    inference = FastFakeInference(
        """```json
        {"answer":"A harbor is visible.","caption":"A coastal harbor.",
        "detected_objects":["harbor"],"evidence":[{"object":"harbor",
        "location":"center","description":"Ships are visible."}],"confidence":null,"errors":[]}
        ```"""
    )

    result = fast_all(inference, "image.jpg", "What is visible?")

    assert inference.generation_count == 1
    assert result.answer == "A harbor is visible."
    assert result.errors == []
    assert '"grounding"' not in result.to_json()


def test_fast_all_returns_error_for_truncated_json() -> None:
    inference = FastFakeInference('{"answer":"incomplete')

    result = fast_all(inference, "image.jpg", "What is visible?")

    assert inference.generation_count == 1
    assert result.answer == ""
    assert result.errors and "incomplete or malformed" in result.errors[0]


def test_fast_all_accepts_single_dict_evidence_and_evidence_alias() -> None:
    inference = FastFakeInference(
        """```json
        {"answer":"A harbor is visible.","caption":"A coastal harbor.",
        "detected_objects":["harbor"],"evidence":{"object":"harbor",
        "location":"center","evidence":"Ships are visible."},"confidence":null,"errors":[]}
        ```"""
    )

    result = fast_all(inference, "image.jpg", "What is visible?")

    assert inference.generation_count == 1
    assert result.answer == "A harbor is visible."
    assert result.errors == []
    assert len(result.evidence) == 1
    assert result.evidence[0]["description"] == "Ships are visible."