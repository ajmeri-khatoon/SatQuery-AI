"""Focused tests for feature-analysis JSON parsing."""

import pytest

from inference.model import Qwen3VLInference


def test_feature_parser_strips_markdown_fences() -> None:
    response = """```json
{"features": [{"object": "waterways", "location": "center", "evidence": "A winding channel is visible."}]}
```"""

    parsed = Qwen3VLInference._parse_feature_analysis(response)

    assert '"object": "waterways"' in parsed


def test_feature_parser_reports_truncated_json() -> None:
    with pytest.raises(ValueError, match="incomplete or malformed"):
        Qwen3VLInference._parse_feature_analysis(
            '{"features": [{"object": "buildings", "evidence": "Dense buildings'
        )
