"""Simple Qwen3-VL image question-answering wrapper."""

from pathlib import Path
import importlib.util
import json
import math
import sys
import time
import traceback

import torch
from PIL import Image, UnidentifiedImageError
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
MAX_IMAGE_SIDE = 768
MAX_NEW_TOKENS = 64
CAPTION_MAX_NEW_TOKENS = 128
FEATURES_MAX_NEW_TOKENS = 384
GROUNDING_MAX_NEW_TOKENS = 128
GPU_MEMORY_LIMIT = "3.5GiB"
CPU_MEMORY_LIMIT = "12GiB"
BITSANDBYTES_INSTALL_COMMAND = "python -m pip install bitsandbytes"


def _classify_model_load_error(error: Exception) -> str:
    """Return a useful category for a model-loading failure."""
    message = str(error).lower()
    error_name = type(error).__name__.lower()
    error_module = type(error).__module__.lower()

    if isinstance(error, torch.cuda.OutOfMemoryError) or "cuda out of memory" in message:
        return "CUDA/GPU memory error"
    if "cuda/gpu" in message or "gpu memory" in message or "vram" in message:
        return "CUDA/GPU availability or VRAM error"
    if isinstance(error, MemoryError) or (
        "out of memory" in message
        and "cuda" not in message
        and "gpu" not in message
        and "cudaerror" not in message
    ):
        return "CPU/RAM memory error"
    if any(
        term in error_name or term in message
        for term in (
            "authentication",
            "unauthorized",
            "forbidden",
            "invalidtoken",
            "gatedrepo",
            "access denied",
            "401",
            "403",
        )
    ):
        return "Hugging Face authentication/access error"
    if "huggingface_hub" in error_module or any(
        term in message
        for term in (
            "hugging face",
            "huggingface",
            "download",
            "snapshot",
            "repository",
            "model files",
            "connection",
            "timed out",
            "ssl",
        )
    ):
        return "Hugging Face/download error"
    if "transformers" in error_module or any(
        term in message
        for term in (
            "transformers",
            "qwen3vl",
            "qwen3-vl",
            "model type",
            "architecture",
            "unexpected keyword",
            "not implemented",
        )
    ):
        return "Transformers/model compatibility error"
    return "Other unexpected error"


class Qwen3VLInference:
    """Load Qwen3-VL and answer questions about one image at a time."""

    def __init__(self, model_id: str = MODEL_ID) -> None:
        load_started = time.perf_counter()

        try:
            cuda_enabled = torch.cuda.is_available()
            print(f"[Qwen3-VL] CUDA enabled: {cuda_enabled}", flush=True)
            if not cuda_enabled:
                raise RuntimeError(
                    "CUDA/GPU is not available. This prototype refuses CPU or "
                    "disk-offloaded inference because Qwen3-VL-4B would be "
                    "unreasonably slow. Run it on a CUDA GPU with sufficient VRAM."
                )

            self.device = torch.device("cuda:0")
            gpu_name = torch.cuda.get_device_name(self.device)
            free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)
            free_gb = free_bytes / 1024**3
            total_gb = total_bytes / 1024**3
            print(f"[Qwen3-VL] GPU: {gpu_name}", flush=True)
            print(
                f"[Qwen3-VL] GPU memory available: {free_gb:.2f} GB / "
                f"{total_gb:.2f} GB",
                flush=True,
            )

            max_memory = {0: GPU_MEMORY_LIMIT, "cpu": CPU_MEMORY_LIMIT}
            quantization_config = self._build_quantization_config()
            load_options = {
                "device_map": "auto",
                "max_memory": max_memory,
                "dtype": torch.float16,
                "low_cpu_mem_usage": True,
            }
            if quantization_config is None:
                raise RuntimeError(
                    "4-bit bitsandbytes quantization is unavailable. Loading Qwen3-VL-4B in "
                    "float16 would cause extensive CPU offloading on this 4GB GPU, leading to "
                    "unacceptable inference latency (~400s). "
                    f"To enable 4-bit quantization, run: {BITSANDBYTES_INSTALL_COMMAND}"
                )

            load_options["quantization_config"] = quantization_config
            print(
                "[Qwen3-VL] 4-bit quantization enabled; using GPU plus "
                "CPU RAM only (disk offloading is disabled).",
                flush=True,
            )

            print(
                f"[Qwen3-VL] Loading {model_id}. Hugging Face cache will be "
                "reused when available...",
                flush=True,
            )
            try:
                self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                    model_id,
                    **load_options,
                )
            except Exception as error:
                quantization_traceback = traceback.format_exc()
                print(
                    "[Qwen3-VL] 4-bit model loading failed; silent float16 fallback disabled.",
                    file=sys.stderr,
                    flush=True,
                )
                print(quantization_traceback, file=sys.stderr, end="")
                raise RuntimeError(
                    f"Failed to load Qwen3-VL model '{model_id}' with 4-bit quantization. "
                    "Silent fallback to FP16 CPU offloading is disabled to prevent severe "
                    f"performance degradation. Error: {error}"
                ) from error
            print("[Qwen3-VL] Model loading complete.", flush=True)
            print("[Qwen3-VL] Loading processor...", flush=True)
            self.processor = AutoProcessor.from_pretrained(model_id)
            print("[Qwen3-VL] Processor loading complete.", flush=True)
            self.model.eval()
            print(
                f"[Qwen3-VL] Model loading time: "
                f"{time.perf_counter() - load_started:.2f}s",
                flush=True,
            )
        except Exception as error:
            print(
                f"[Qwen3-VL] Model loading time: "
                f"{time.perf_counter() - load_started:.2f}s",
                file=sys.stderr,
                flush=True,
            )
            category = _classify_model_load_error(error)
            original_traceback = traceback.format_exc()
            print(
                f"Qwen3-VL model-loading category: {category}",
                file=sys.stderr,
            )
            print("Original model-loading exception and traceback:", file=sys.stderr)
            print(original_traceback, file=sys.stderr, end="")
            raise RuntimeError(
                f"Could not load Qwen3-VL model '{model_id}' ({category}).\n"
                f"Original exception: {error!r}\n"
                f"Original traceback:\n{original_traceback}"
            ) from error

    @staticmethod
    def _build_quantization_config():
        """Return 4-bit settings when bitsandbytes works in this environment."""
        if importlib.util.find_spec("bitsandbytes") is None:
            print(
                "[Qwen3-VL] 4-bit quantization package is not installed. "
                f"To enable it later, run: {BITSANDBYTES_INSTALL_COMMAND}",
                file=sys.stderr,
                flush=True,
            )
            return None

        try:
            import bitsandbytes  # noqa: F401
            from transformers import BitsAndBytesConfig

            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        except Exception as error:
            original_traceback = traceback.format_exc()
            print(
                "[Qwen3-VL] 4-bit quantization is unavailable in this "
                "environment; complete detection error:",
                file=sys.stderr,
                flush=True,
            )
            print(original_traceback, file=sys.stderr, end="")
            print(
                f"[Qwen3-VL] To retry with 4-bit support, run: "
                f"{BITSANDBYTES_INSTALL_COMMAND}",
                file=sys.stderr,
                flush=True,
            )
            return None

    @staticmethod
    def validate_image(image_path: str | Path) -> None:
        """Raise a helpful error if the path is missing or is not a valid image."""
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")

        try:
            with Image.open(path) as image:
                image.verify()
        except (UnidentifiedImageError, OSError) as error:
            raise ValueError(f"Invalid image file: {path}") from error

    @staticmethod
    def _open_rgb_image(
        image_path: str | Path,
        max_image_side: int = MAX_IMAGE_SIDE,
    ) -> Image.Image:
        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")
        rgb_image.thumbnail(
            (max_image_side, max_image_side),
            Image.Resampling.LANCZOS,
        )
        return rgb_image

    def _prepare_inputs(
        self,
        image_path: str | Path,
        question: str,
        max_image_side: int = MAX_IMAGE_SIDE,
        enable_thinking: bool | None = None,
    ):
        processing_started = time.perf_counter()
        print("[Qwen3-VL] Processing image...", flush=True)
        image = self._open_rgb_image(image_path, max_image_side=max_image_side)
        print("[Qwen3-VL] Image processing complete.", flush=True)
        print("[Qwen3-VL] Processing question...", flush=True)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]
        chat_template_kwargs = {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_dict": True,
            "return_tensors": "pt",
        }
        if enable_thinking is not None:
            try:
                inputs = self.processor.apply_chat_template(
                    messages,
                    enable_thinking=enable_thinking,
                    **chat_template_kwargs,
                )
            except TypeError:
                inputs = self.processor.apply_chat_template(
                    messages,
                    **chat_template_kwargs,
                )
        else:
            inputs = self.processor.apply_chat_template(
                messages,
                **chat_template_kwargs,
            )
        inputs = inputs.to(self.device)
        print("[Qwen3-VL] Question processing complete.", flush=True)
        print(
            f"[Qwen3-VL] Image processing time: "
            f"{time.perf_counter() - processing_started:.2f}s",
            flush=True,
        )
        return inputs

    # sourcery skip: extract-method
    def _generate_answer(self, image_path: str | Path, question: str) -> str:
        return self._generate_text(
            image_path,
            question,
            max_new_tokens=MAX_NEW_TOKENS,
        )

    def fast_all_answer(self, image_path: str | Path, question: str) -> str:
        """Generate one compact fast-all JSON response for demo use."""
        fast_prompt = (
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
            f"User question: {question}"
        )
        inference_started = time.perf_counter()
        try:
            self.validate_image(image_path)
            generation_started = time.perf_counter()
            print(
                "[Qwen3-VL] Fast all generation count: 1",
                flush=True,
            )
            response = self._generate_text(
                image_path,
                fast_prompt,
                max_new_tokens=128,
                max_image_side=512,
                enable_thinking=False,
            )
            print(
                f"[Qwen3-VL] Fast all generation time: {time.perf_counter() - generation_started:.2f}s",
                flush=True,
            )
            return response
        finally:
            print(
                f"[Qwen3-VL] Fast all total time: {time.perf_counter() - inference_started:.2f}s",
                flush=True,
            )

    def _generate_text(
        self,
        image_path: str | Path,
        question: str,
        max_new_tokens: int,
        max_image_side: int = MAX_IMAGE_SIDE,
        enable_thinking: bool | None = None,
    ) -> str:
        try:
            # sourcery skip: extract-method
            inputs = self._prepare_inputs(
                image_path,
                question,
                max_image_side=max_image_side,
                enable_thinking=enable_thinking,
            )

            generation_started = time.perf_counter()
            print(
                f"[Qwen3-VL] Generating answer "
                f"(max_new_tokens={max_new_tokens})...",
                flush=True,
            )
            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                )
            print("[Qwen3-VL] Model generation complete.", flush=True)
            print(
                f"[Qwen3-VL] Generation time: "
                f"{time.perf_counter() - generation_started:.2f}s",
                flush=True,
            )

            generated_ids_trimmed = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
            ]
            return self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
        except (FileNotFoundError, ValueError):
            raise
        except Exception as error:
            original_traceback = traceback.format_exc()
            print("[Qwen3-VL] Generation failed with the original exception:", file=sys.stderr)
            print(original_traceback, file=sys.stderr, end="")
            raise RuntimeError(
                f"Inference failed for image '{image_path}'.\n"
                f"Original exception: {error!r}\n"
                f"Original traceback:\n{original_traceback}"
            ) from error

    def caption_image(self, image_path: str | Path) -> str:
        """Generate a concise, evidence-based caption for a satellite image."""
        caption_prompt = (
            "Describe this satellite or remote-sensing image in a detailed but "
            "concise caption. Cover the major land-use and land-cover types; "
            "buildings and urban areas; roads and transportation infrastructure; "
            "water bodies and waterways; vegetation; agricultural areas; and "
            "industrial or port infrastructure when visible. Mention notable "
            "spatial patterns and approximate relative locations such as "
            "north/south/east/west or upper/lower/left/right only when clearly "
            "visible. Do not invent objects, locations, labels, or details that "
            "cannot reasonably be seen. Return only the caption."
        )
        inference_started = time.perf_counter()
        try:
            self.validate_image(image_path)
            return self._generate_text(
                image_path,
                caption_prompt,
                max_new_tokens=CAPTION_MAX_NEW_TOKENS,
            )
        finally:
            print(
                f"[Qwen3-VL] Total captioning time: "
                f"{time.perf_counter() - inference_started:.2f}s",
                flush=True,
            )

    @staticmethod
    def _parse_feature_analysis(response: str) -> str:
        """Validate and normalize the model response as clean feature JSON."""
        cleaned_response = response.replace("```json", "").replace("```", "").strip()
        json_start = cleaned_response.find("{")
        if json_start == -1:
            raise ValueError("Feature analysis did not contain a JSON object.")

        try:
            parsed, _ = json.JSONDecoder().raw_decode(cleaned_response[json_start:])
        except json.JSONDecodeError as error:
            raise ValueError(
                "Feature analysis JSON is incomplete or malformed; "
                "the model response may have been truncated."
            ) from error
        features = parsed.get("features") if isinstance(parsed, dict) else None
        if not isinstance(features, list):
            raise ValueError("Feature analysis JSON must contain a 'features' list.")

        required_fields = ("object", "location", "evidence")
        for index, feature in enumerate(features):
            if not isinstance(feature, dict) or any(
                not isinstance(feature.get(field), str) or not feature[field].strip()
                for field in required_fields
            ):
                raise ValueError(
                    "Feature analysis entry "
                    f"{index} must contain non-empty object, location, and evidence strings."
                )

        return json.dumps({"features": features}, indent=2)

    def analyze_features(self, image_path: str | Path) -> str:
        """Identify visible satellite-image objects and land-use features as JSON."""
        feature_prompt = (
            "Analyze this satellite image and return JSON only, with no Markdown "
            "or explanations. Return at most 5-8 important clearly visible "
            "features, without repeating objects, using exactly this compact "
            "schema: {\"features\":[{\"object\":\"buildings\","
            "\"location\":\"center\",\"evidence\":\"Dense buildings visible.\"}]}. "
            "Use short evidence descriptions of fewer than 12 words. Consider "
            "buildings, roads, bridges, waterways, vegetation, fields, industrial "
            "areas, ports, railways, coastlines, islands, and other land-use or "
            "land-cover features, but include only features reasonably visible. "
            "Use approximate locations upper-left, upper-center, upper-right, "
            "center, lower-left, lower-center, or lower-right. Do not invent "
            "objects. Return {\"features\":[]} if nothing is reliable."
        )
        retry_prompt = (
            "Return ONLY this compact JSON object: {\"features\":[{"
            "\"object\":\"...\",\"location\":\"...\","
            "\"evidence\":\"...\"}]}. Maximum 5 features. Keep each evidence "
            "under 12 words. Do not use Markdown, explanations, repeated objects, "
            "or any text outside JSON. Include only clearly visible satellite "
            "features; otherwise return {\"features\":[]}."
        )
        inference_started = time.perf_counter()
        try:
            self.validate_image(image_path)
            response = self._generate_text(
                image_path,
                feature_prompt,
                max_new_tokens=FEATURES_MAX_NEW_TOKENS,
            )
            try:
                return self._parse_feature_analysis(response)
            except ValueError as first_error:
                print(
                    "[Qwen3-VL] Feature JSON was incomplete or malformed; "
                    "retrying once with a compact prompt.",
                    file=sys.stderr,
                    flush=True,
                )
                retry_response = self._generate_text(
                    image_path,
                    retry_prompt,
                    max_new_tokens=256,
                )
                try:
                    return self._parse_feature_analysis(retry_response)
                except ValueError as retry_error:
                    raise ValueError(
                        f"Feature analysis remained invalid after one retry: {retry_error}"
                    ) from first_error
        except Exception as error:
            original_traceback = traceback.format_exc()
            print(
                "[Qwen3-VL] Feature analysis failed with the original exception:",
                file=sys.stderr,
                flush=True,
            )
            print(original_traceback, file=sys.stderr, end="")
            raise RuntimeError(
                f"Feature analysis failed for image '{image_path}'.\n"
                f"Original exception: {error!r}\n"
                f"Original traceback:\n{original_traceback}"
            ) from error
        finally:
            print(
                f"[Qwen3-VL] Total feature analysis time: "
                f"{time.perf_counter() - inference_started:.2f}s",
                flush=True,
            )

    @staticmethod
    def _parse_grounding(response: str, requested_object: str) -> str:
        """Convert and validate Qwen's approximate 0-1000 grounding boxes."""
        json_start = response.find("{")
        if json_start == -1:
            raise ValueError("Grounding response did not contain a JSON object.")

        parsed, _ = json.JSONDecoder().raw_decode(response[json_start:])
        if not isinstance(parsed, dict):
            raise ValueError("Grounding response must be a JSON object.")

        object_name = parsed.get("object", requested_object)
        found = parsed.get("found")
        location = parsed.get("location")
        evidence = parsed.get("evidence")
        boxes = parsed.get("boxes")
        if not isinstance(object_name, str) or not object_name.strip():
            raise ValueError("Grounding JSON must contain a non-empty object name.")
        if not isinstance(found, bool):
            raise ValueError("Grounding JSON must contain a boolean 'found' value.")
        if location is not None and not isinstance(location, str):
            raise ValueError("Grounding 'location' must be a string or null.")
        if not isinstance(evidence, str) or not evidence.strip():
            raise ValueError("Grounding JSON must contain non-empty evidence.")

        result = {
            "object": object_name,
            "found": found,
            "bounding_boxes": [],
            "location": location,
            "evidence": evidence,
        }
        if not found or not isinstance(boxes, list):
            result["found"] = False
            result["bounding_boxes"] = []
            result["location"] = None
            return json.dumps(result, indent=2)

        justified_full_image = any(
            phrase in evidence.lower()
            for phrase in ("nearly entire image", "almost entire image", "fills the image")
        )
        normalized_boxes = []
        for box in boxes:
            valid_box = (
                isinstance(box, list)
                and len(box) == 4
                and all(
                    isinstance(coordinate, (int, float))
                    and not isinstance(coordinate, bool)
                    and math.isfinite(coordinate)
                    and 0 <= coordinate <= 1000
                    for coordinate in box
                )
            )
            if not valid_box:
                continue

            x1, y1, x2, y2 = (coordinate / 1000 for coordinate in box)
            area = (x2 - x1) * (y2 - y1)
            is_full_image = x1 == 0 and y1 == 0 and x2 == 1 and y2 == 1
            if (
                x1 >= x2
                or y1 >= y2
                or area <= 0
                or (is_full_image and not justified_full_image)
                or (area > 0.95 and not justified_full_image)
            ):
                continue

            normalized_boxes.append(
                {
                    "x1": round(x1, 4),
                    "y1": round(y1, 4),
                    "x2": round(x2, 4),
                    "y2": round(y2, 4),
                }
            )

        if not normalized_boxes:
            result["found"] = False
            result["bounding_boxes"] = []
            result["location"] = None
            result["evidence"] = (
                f"{evidence} No reliable, reasonably sized bounding box could be "
                "validated, so this result is treated as uncertain."
            )
        else:
            result["bounding_boxes"] = normalized_boxes

        return json.dumps(result, indent=2)

    def ground_object(self, image_path: str | Path, object_name: str) -> str:
        """Estimate an object's approximate normalized image bounding box."""
        requested_object = object_name.strip()
        if not requested_object:
            raise ValueError("Object name cannot be empty.")

        grounding_prompt = (
            "Visually ground ONLY the requested object or feature in this "
            "satellite or remote-sensing image. The requested object is "
            f"{json.dumps(requested_object)}. Return only valid JSON with exactly "
            "this shape: {\"object\": \"requested object\", \"found\": true, "
            "\"boxes\": [[100, 200, 800, 600]], \"location\": \"center\", "
            "\"evidence\": \"brief visible evidence\"}. Coordinates must be "
            "Qwen grounding coordinates from 0 to 1000, in the order [x1, y1, "
            "x2, y2], where x1 is left, y1 is top, x2 is right, and y2 is "
            "bottom. The program will convert them to normalized 0-1 values. "
            "Locate ONLY the requested object. Use the smallest reasonable "
            "axis-aligned region containing it, distinguish it from surrounding "
            "background, and never use the whole image unless the target truly "
            "occupies nearly the entire image. If clearly separated regions exist, "
            "return one box for each region. Use an approximate box, not pixel- "
            "perfect detection. Use one of these "
            "locations when appropriate: upper-left, upper-center, upper-right, "
            "center, lower-left, lower-center, lower-right. If the requested "
            "object cannot be confidently located, return found false, "
            "boxes [], location null, and explain why in evidence. Never invent "
            "coordinates. Return only JSON, without markdown."
        )
        inference_started = time.perf_counter()
        try:
            self.validate_image(image_path)
            response = self._generate_text(
                image_path,
                grounding_prompt,
                max_new_tokens=GROUNDING_MAX_NEW_TOKENS,
            )
            return self._parse_grounding(response, requested_object)
        except Exception as error:
            original_traceback = traceback.format_exc()
            print(
                "[Qwen3-VL] Visual grounding failed with the original exception:",
                file=sys.stderr,
                flush=True,
            )
            print(original_traceback, file=sys.stderr, end="")
            raise RuntimeError(
                f"Visual grounding failed for image '{image_path}'.\n"
                f"Original exception: {error!r}\n"
                f"Original traceback:\n{original_traceback}"
            ) from error
        finally:
            print(
                f"[Qwen3-VL] Total grounding time: "
                f"{time.perf_counter() - inference_started:.2f}s",
                flush=True,
            )

    def answer_question(self, image_path: str | Path, question: str) -> str:
        """Return Qwen3-VL's answer to a question about an image."""
        inference_started = time.perf_counter()
        try:
            self.validate_image(image_path)
            if not question.strip():
                raise ValueError("Question cannot be empty.")
            return self._generate_answer(image_path, question)
        finally:
            print(
                f"[Qwen3-VL] Total inference time: "
                f"{time.perf_counter() - inference_started:.2f}s",
                flush=True,
            )