"""Focused tests for VRSBench local schema normalization."""

import json
from pathlib import Path
import sys

DATASETS_ROOT = Path(__file__).resolve().parents[1] / "datasets"
sys.path.insert(0, str(DATASETS_ROOT))

from vrsbench.loader import VRSBenchLoader


def test_loader_normalizes_official_annotation_shapes(tmp_path: Path) -> None:
    annotation_dir = tmp_path / "Annotations_train"
    image_dir = tmp_path / "Images_train"
    annotation_dir.mkdir()
    image_dir.mkdir()
    (image_dir / "scene.png").write_bytes(b"placeholder")
    record = {
        "image": "scene.png",
        "caption": "A rural scene with fields.",
        "objects": [
            {
                "referring_sentence": "the field in the upper left",
                "obj_cls": "field",
                "obj_corner": [0.1, 0.2, 0.3, 0.2, 0.3, 0.4, 0.1, 0.4],
            }
        ],
        "qa_pairs": [{"question": "What is visible?", "answer": "fields", "type": "image"}],
    }
    (annotation_dir / "scene.json").write_text(json.dumps(record), encoding="utf-8")

    samples = list(VRSBenchLoader(tmp_path).iter_samples("train", limit=1))

    assert len(samples) == 1
    assert samples[0].image_path == image_dir / "scene.png"
    assert samples[0].caption == record["caption"]
    assert samples[0].vqa[0].answer == "fields"
    assert samples[0].grounding[0].bounding_box == (0.1, 0.2, 0.3, 0.4)
