"""Bounded, schema-aware access to the official VRSBench dataset.

The loader does not download files. It reads an existing local VRSBench layout or
uses Hugging Face streaming and materializes only samples requested by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
import json
from pathlib import Path
import sys
from typing import Any, Iterator

from PIL import Image


OFFICIAL_DATASET_ID = "xiang709/VRSBench"
OFFICIAL_SPLITS = ("train", "val")


@dataclass(frozen=True)
class VQAAnnotation:
    question: str
    answer: str
    question_type: str | None = None


@dataclass(frozen=True)
class GroundingAnnotation:
    referring_sentence: str
    bounding_box: tuple[float, float, float, float] | None
    object_class: str | None = None
    is_unique: bool | None = None


@dataclass
class VRSBenchSample:
    sample_id: str
    image_path: Path | None
    caption: str | None = None
    vqa: list[VQAAnnotation] = field(default_factory=list)
    grounding: list[GroundingAnnotation] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


class VRSBenchLoader:
    """Load normalized VRSBench samples without downloading the full dataset."""

    def __init__(
        self,
        root: str | Path = "datasets/vrsbench",
        dataset_id: str = OFFICIAL_DATASET_ID,
    ) -> None:
        self.root = Path(root)
        self.dataset_id = dataset_id

    def iter_samples(self, split: str = "train", limit: int | None = None) -> Iterator[VRSBenchSample]:
        """Yield at most ``limit`` samples from local files or HF streaming."""
        if limit is not None and limit < 1:
            return

        local_records = self._iter_local_records(split)
        if local_records is not None:
            records = local_records
        else:
            records = self._iter_huggingface_records(split)

        yielded = 0
        for record in records:
            sample = self._normalize_record(record, split)
            if sample is None:
                continue
            yield sample
            yielded += 1
            if limit is not None and yielded >= limit:
                return

    def _iter_local_records(self, split: str) -> Iterator[dict[str, Any]] | None:
        annotation_dir = self._first_existing(
            self.root / f"Annotations_{split}",
            self.root / f"annotations_{split}",
            self.root / split / "annotations",
        )
        if annotation_dir is not None and annotation_dir.is_dir():
            return self._iter_json_files(sorted(annotation_dir.glob("*.json")))

        json_file = self._first_existing(
            self.root / f"Annotations_{split}.json",
            self.root / f"annotations_{split}.json",
            self.root / f"Annotations_{split}.jsonl",
            self.root / f"annotations_{split}.jsonl",
        )
        return self._iter_json_file(json_file) if json_file is not None else None

    @staticmethod
    def _first_existing(*paths: Path) -> Path | None:
        return next((path for path in paths if path.exists()), None)

    def _iter_json_files(self, paths: list[Path]) -> Iterator[dict[str, Any]]:
        for path in paths:
            if path.name.endswith("_input.json"):
                continue
            yield from self._iter_json_file(path)

    @staticmethod
    def _iter_json_file(path: Path) -> Iterator[dict[str, Any]]:
        if path.suffix.lower() == ".jsonl":
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        record = json.loads(line)
                        if isinstance(record, dict):
                            yield record
            return

        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            yield payload
        elif isinstance(payload, list):
            yield from (record for record in payload if isinstance(record, dict))

    def _iter_huggingface_records(self, split: str) -> Iterator[dict[str, Any]]:
        if split not in OFFICIAL_SPLITS:
            raise RuntimeError(
                f"VRSBench split '{split}' is not published by the official "
                f"repository. Verified archive-backed splits: {', '.join(OFFICIAL_SPLITS)}."
            )

        try:
            load_dataset = self._import_huggingface_loader()
        except ImportError as error:
            raise RuntimeError(
                "No local VRSBench annotations were found and the 'datasets' "
                "package is unavailable. Install the existing requirements or "
                "place the official annotations under datasets/vrsbench."
            ) from error

        try:
            dataset = load_dataset(
                self.dataset_id,
                split=split,
                streaming=True,
            )
        except Exception as error:
            raise RuntimeError(
                f"Could not stream verified split '{split}' from {self.dataset_id} "
                "using its default configuration. The dataset is public and the "
                "current Hugging Face metadata publishes 'train' and 'val'. "
                f"Original error: {error!r}"
            ) from error

        for record in dataset:
            if isinstance(record, dict):
                yield record

    @staticmethod
    def _import_huggingface_loader():
        """Import HF datasets even when this project folder is named datasets."""
        project_root = Path(__file__).resolve().parents[2]
        original_paths = list(sys.path)
        original_module = sys.modules.get("datasets")
        try:
            sys.path[:] = [
                path for path in sys.path
                if Path(path or ".").resolve() not in {
                    project_root.resolve(),
                    (project_root / "datasets").resolve(),
                }
            ]
            if original_module is not None and not hasattr(original_module, "load_dataset"):
                del sys.modules["datasets"]
            from datasets import load_dataset
            return load_dataset
        except ModuleNotFoundError as error:
            raise ImportError("Hugging Face datasets is not installed in this interpreter.") from error
        finally:
            sys.path[:] = original_paths

    def _normalize_record(self, record: dict[str, Any], split: str) -> VRSBenchSample | None:
        sample_id = self._sample_id(record)
        if sample_id is None:
            return None

        image_path = self._resolve_image(record.get("image"), sample_id, split)
        if image_path is None:
            image_path = self._materialize_streamed_image(record.get("image"), sample_id, split)

        return VRSBenchSample(
            sample_id=sample_id,
            image_path=image_path,
            caption=self._caption(record),
            vqa=self._vqa(record),
            grounding=self._grounding(record),
            raw=record,
        )

    @staticmethod
    def _sample_id(record: dict[str, Any]) -> str | None:
        image = record.get("image")
        if isinstance(image, str) and image:
            return Path(image).name
        if image_filename := getattr(image, "filename", None):
            return Path(str(image_filename)).name
        for key in ("image_id", "filename", "id"):
            value = record.get(key)
            if value is not None and str(value).strip():
                return str(value)
        if isinstance(image, dict):
            for key in ("path", "filename", "name"):
                if image.get(key):
                    return Path(str(image[key])).name
        return None

    @staticmethod
    def _caption(record: dict[str, Any]) -> str | None:
        for key in ("caption", "description"):
            value = record.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    @staticmethod
    def _vqa(record: dict[str, Any]) -> list[VQAAnnotation]:
        raw_pairs = record.get("qa_pairs", [])
        if not raw_pairs and "question" in record and "answer" in record:
            raw_pairs = [record]
        annotations = []
        for pair in raw_pairs if isinstance(raw_pairs, list) else []:
            if not isinstance(pair, dict):
                continue
            question = pair.get("question")
            answer = pair.get("answer")
            if isinstance(question, str) and isinstance(answer, str):
                annotations.append(
                    VQAAnnotation(question, answer, pair.get("type"))
                )
        return annotations

    @staticmethod
    def _grounding(record: dict[str, Any]) -> list[GroundingAnnotation]:
        raw_objects = record.get("objects", record.get("refer_objects", []))
        if not isinstance(raw_objects, list):
            return []

        annotations = []
        for obj in raw_objects:
            if not isinstance(obj, dict):
                continue
            sentence = obj.get("referring_sentence")
            if not isinstance(sentence, str) or not sentence.strip():
                continue
            bbox = VRSBenchLoader._object_bbox(obj)
            annotations.append(
                GroundingAnnotation(
                    referring_sentence=sentence.strip(),
                    bounding_box=bbox,
                    object_class=obj.get("obj_cls"),
                    is_unique=obj.get("is_unique"),
                )
            )
        return annotations

    @staticmethod
    def _object_bbox(obj: dict[str, Any]) -> tuple[float, float, float, float] | None:
        raw_bbox = obj.get("obj_coord")
        if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) == 4:
            return VRSBenchLoader._normalize_bbox(raw_bbox)

        corners = obj.get("obj_corner")
        if not isinstance(corners, (list, tuple)) or len(corners) < 4:
            return None
        try:
            points = list(zip(corners[::2], corners[1::2]))
            xs = [float(point[0]) for point in points]
            ys = [float(point[1]) for point in points]
        except (TypeError, ValueError):
            return None
        if not xs or not ys:
            return None
        return VRSBenchLoader._normalize_bbox((min(xs), min(ys), max(xs), max(ys)))

    @staticmethod
    def _normalize_bbox(values: Any) -> tuple[float, float, float, float] | None:
        try:
            numbers = tuple(float(value) for value in values)
        except (TypeError, ValueError):
            return None
        if max(numbers) > 1:
            numbers = tuple(value / 100 for value in numbers)
        x1, y1, x2, y2 = numbers
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            return None
        return x1, y1, x2, y2

    def _resolve_image(self, image: Any, sample_id: str, split: str) -> Path | None:
        candidates = []
        if isinstance(image, str):
            candidates.append(Path(image))
        if isinstance(image, dict):
            candidates.extend(
                Path(str(image[key]))
                for key in ("path", "filename", "name")
                if image.get(key)
            )
        candidates.extend(
            self.root / directory / sample_id
            for directory in (
                f"Images_{split}",
                f"images_{split}",
                "images",
                split,
            )
        )
        candidates.append(self.root / sample_id)
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        for extension in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
            if matches := list(self.root.rglob(Path(sample_id).stem + extension)):
                return matches[0]
        return None

    def _materialize_streamed_image(self, image: Any, sample_id: str, split: str) -> Path | None:
        """Cache one streamed image so path-based inference can consume it."""
        image_type = type(image).__name__
        representation = "unknown"
        image_object = None

        if isinstance(image, Image.Image):
            representation = "PIL.Image.Image"
            image_object = image
        elif isinstance(image, dict):
            representation = f"dict(keys={sorted(image)})"
            image_path = image.get("path")
            image_bytes = image.get("bytes")
            if image_path and Path(str(image_path)).is_file():
                print(
                    f"VRSBench image field type: {image_type}\n"
                    f"VRSBench image representation: {representation}\n"
                    f"Materialized image path: {image_path}",
                    flush=True,
                )
                return Path(str(image_path))
            if isinstance(image_bytes, (bytes, bytearray)):
                image_object = Image.open(BytesIO(image_bytes))
        elif isinstance(image, (bytes, bytearray)):
            representation = "raw bytes"
            image_object = Image.open(BytesIO(image))

        if image_object is None:
            print(
                f"VRSBench image field type: {image_type}\n"
                f"VRSBench image representation: {representation}\n"
                "Materialized image path: unavailable",
                file=sys.stderr,
                flush=True,
            )
            return None

        cache_dir = self.root / ".cache" / "images" / split
        cache_dir.mkdir(parents=True, exist_ok=True)
        image_path = cache_dir / Path(sample_id).name
        if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            image_path = image_path.with_suffix(".png")
        if not image_path.exists():
            image_object.convert("RGB").save(image_path)
        print(
            f"VRSBench image field type: {image_type}\n"
            f"VRSBench image representation: {representation}\n"
            f"Materialized image path: {image_path}\n"
            f"Image size: {image_object.size}",
            flush=True,
        )
        return image_path
