# REMOTE-SENSING VISION AI

AI-powered remote-sensing and satellite image understanding using **Qwen3-VL**, 4-bit quantization, Hugging Face Transformers, and Gradio.

![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6%2B%20%28CUDA%29-orange)
![Transformers](https://img.shields.io/badge/Transformers-4.57%2B-yellow)
![Model](https://img.shields.io/badge/Model-Qwen3--VL--4B--Instruct-blueviolet)
![Quantization](https://img.shields.io/badge/Quantization-4--bit%20NF4-green)
![Gradio](https://img.shields.io/badge/UI-Gradio%206-red)

---

## Overview

**REMOTE-SENSING VISION AI** is an end-to-end vision-language application tailored for satellite and aerial earth-observation imagery. It combines the multimodal power of Alibaba's **Qwen3-VL-4B-Instruct** model with 4-bit `bitsandbytes` NormalFloat4 (NF4) quantization, enabling full on-device GPU inference on budget GPUs such as the **NVIDIA GeForce RTX 3050 (4 GB VRAM)** with zero CPU-offloading performance penalties.

The system provides:
* An interactive **Gradio Web Interface** with instant visual results, structured evidence tables, and expandable raw JSON.
* A versatile **CLI tool** supporting single-pass fast demo analysis, standalone VQA, image captioning, land-use feature extraction, visual grounding, and unified structured JSON output.
* Built-in support for the official **VRSBench** remote-sensing vision-language benchmark.

---

## Features

* **Satellite Visual Question Answering (VQA):** Answers specific natural language questions about land cover, infrastructure, water bodies, and terrain.
* **Detailed Satellite Captioning:** Generates concise, evidence-based descriptions covering urban density, vegetation, transportation, agriculture, and waterways.
* **Object & Land-Use Feature Analysis:** Detects and categorizes visible geographic features along with spatial locations (e.g., *center*, *upper-left*, *lower-right*) and visual evidence.
* **Visual Grounding:** Estimates normalized bounding boxes (`0.0` to `1.0`) for requested satellite objects and targets.
* **Fast-All Unified Inference (`--fast-all`):** Executes VQA, captioning, object detection, and visual evidence in **a single forward pass (~12–15s)** rather than running 4 separate sequential model queries.
* **VRAM-Optimized Loading (RTX 3050 4GB Compatible):** Uses 4-bit NF4 quantization with double quantization and float16 compute to fit 100% of the model layers (~2.77 GB) onto a 4 GB GPU.
* **Web Dashboard:** Clean, user-friendly Gradio interface with one-time model pre-loading and multi-user concurrency locks.

---

## Architecture & Workflow

```
                  +-----------------------------------+
                  |  Satellite / Aerial Image Input   |
                  +-----------------+-----------------+
                                    |
                                    v
                  +-----------------------------------+
                  |      Validation & Preprocessing   |
                  |  - PIL RGB format verification    |
                  |  - Lanczos aspect-ratio thumbnail |
                  |    (512px fast / 768px full)      |
                  +-----------------+-----------------+
                                    |
                                    v
                  +-----------------------------------+
                  |   Qwen3-VL-4B-Instruct Backend    |
                  |  - 4-bit NF4 bitsandbytes quant   |
                  |  - 559/559 leaf modules in VRAM   |
                  |  - ~2.77 GB allocated (RTX 3050)  |
                  |  - enable_thinking=False guard    |
                  +-----------------+-----------------+
                                    |
            +-----------------------+-----------------------+
            |                                               |
            v                                               v
+-----------------------+                       +-----------------------+
|   Gradio Web Server   |                       |    Command-Line Tool  |
|      (app.py)         |                       | (inference/predict.py)|
| - Instant feedback    |                       | - --fast-all          |
| - Answer & Caption    |                       | - --question / --vqa  |
| - Detected Objects    |                       | - --caption           |
| - Evidence Table      |                       | - --features          |
| - Raw JSON Accordion  |                       | - --ground            |
+-----------------------+                       | - --all / --structured|
                                                +-----------------------+
```

---

## Technology Stack

| Component | Technology | Description |
|---|---|---|
| **Base Language** | Python 3.10 – 3.12 | Modern type-annotated code |
| **Deep Learning Framework** | PyTorch 2.6+ (CUDA 12.x) | Tensor acceleration and GPU kernel dispatch |
| **Vision-Language Model** | Qwen3-VL-4B-Instruct | Multimodal autoregressive transformer by Alibaba |
| **Quantization** | bitsandbytes 0.45+ | 4-bit NormalFloat4 (NF4) with double quant |
| **Transformer Pipeline** | Hugging Face Transformers & Accelerate | Tokenization, chat templating, model dispatch |
| **Image Processing** | Pillow (PIL) | Safe loading, Lanczos downsampling, annotation |
| **Web Interface** | Gradio 6 | Modern reactive web UI with progress tracking |
| **Testing** | Pytest | Unit and integration test suite |

---

## Hardware & Software Requirements

### Hardware Requirements
* **GPU (Recommended):** NVIDIA GPU with **4 GB+ VRAM** (e.g., GeForce RTX 3050 Laptop / Desktop, RTX 3060, RTX 4050/4060, T4, A10, V100).
* **System RAM:** 8 GB minimum (16 GB recommended).
* **Disk Space:** ~10 GB free disk space for downloading and caching Hugging Face model weights.
* *Note on CPU-only:* Running Qwen3-VL without a CUDA GPU is blocked by default because autoregressive execution on CPU takes ~400s per query.

### Software Requirements
* **Operating System:** Windows 10/11 (64-bit) or Linux (Ubuntu 20.04+).
* **Python:** Version 3.10, 3.11, or 3.12.
* **CUDA Driver:** NVIDIA Driver compatible with CUDA 12.1+ or CUDA 12.8+.

---

## Installation (Windows)

### 1. Clone the Repository
```powershell
git clone https://github.com/your-username/remote-sensing-vision-ai.git
cd remote-sensing-vision-ai
```

### 2. Create and Activate a Virtual Environment
```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install PyTorch with CUDA Support
Before installing other dependencies, install PyTorch matching your CUDA version (CUDA 12.4/12.8 recommended):
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### 4. Install Project Dependencies
```powershell
pip install -r requirements.txt
```

---

## Hugging Face Model Information

The project uses [`Qwen/Qwen3-VL-4B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct).

* **Automatic Download:** When you start the Gradio app or run a CLI inference command for the first time, Hugging Face automatically downloads the model weights (~8.5 GB) and caches them in your local directory (`~/.cache/huggingface/hub/`).
* **Offline Reuse:** Subsequent runs load directly from the local cache in ~95 seconds without re-downloading.
* **Hugging Face Token (Optional):** Setting an `HF_TOKEN` environment variable is optional, but helps avoid unauthenticated download rate limits:
  ```powershell
  $env:HF_TOKEN = "your_token_here"
  ```

---

## How to Run

### Method 1: Using the Batch Launcher (Windows)
Double-click `run_app.bat` or run it from the terminal:
```powershell
.\run_app.bat
```
This script automatically navigates to the project directory, activates `.venv`, and starts `app.py`.

### Method 2: Command Line Web Server
```powershell
.venv\Scripts\python.exe app.py
```

Once the model finishes loading, open your web browser at:
👉 **`http://127.0.0.1:7860`**

---

## How to Use the Gradio Web Interface

1. **Upload Image:** Drag and drop or upload any satellite, aerial, or drone image (PNG, JPG, WEBP) in the left panel.
2. **Enter Question:** Type a question or use the default:
   `"What objects are visible in this satellite image?"`
3. **Analyze:** Click the **Analyze Satellite Image** button.
4. **Inspect Results:**
   * **Answer:** Direct concise answer from the vision model.
   * **Caption:** Satellite image scene summary.
   * **Detected Objects:** Comma-separated list of identified land-cover objects.
   * **Visual Evidence:** Structured table showing `Object`, `Location` (e.g., *center*, *upper-left*), and `Description`.
   * **Confidence:** Model confidence value (`null` when not estimated).
   * **Errors / Warnings:** Any runtime alerts or validation warnings.
5. **Raw JSON:** Expand the **Raw JSON** accordion at the bottom to inspect the complete structured dictionary.

---

## CLI Usage Examples

The CLI tool [`inference/predict.py`](inference/predict.py) provides high-performance terminal access:

### 1. Fast Unified Demo Mode (`--fast-all`)
Runs all capabilities in **one single model generation (~12–15s)** and prints clean JSON:
```powershell
.venv\Scripts\python.exe inference\predict.py --image datasets\test_satellite.jpg --question "What objects are visible in this satellite image?" --fast-all
```

### 2. Standalone Visual Question Answering (VQA)
```powershell
.venv\Scripts\python.exe inference\predict.py --image datasets\test_satellite.jpg --question "Is there a harbor or port visible?"
```

### 3. Satellite Image Captioning
```powershell
.venv\Scripts\python.exe inference\predict.py --image datasets\test_satellite.jpg --caption
```

### 4. Feature and Object Analysis
```powershell
.venv\Scripts\python.exe inference\predict.py --image datasets\test_satellite.jpg --features
```

### 5. Visual Grounding
```powershell
.venv\Scripts\python.exe inference\predict.py --image datasets\test_satellite.jpg --ground "buildings"
```

### 6. Full Multi-Query Analysis (`--all`)
Executes separate specialized generations for VQA, captioning, feature analysis, and visual grounding, returning unified JSON:
```powershell
.venv\Scripts\python.exe inference\predict.py --image datasets\test_satellite.jpg --question "Analyze this scene" --all
```

---

## Project Structure

```text
REMOTE-SENSING VISION AI/
├── datasets/
│   ├── vrsbench/
│   │   ├── __init__.py
│   │   ├── loader.py              # VRSBench dataset streaming & local loader
│   │   └── README.md              # Official VRSBench benchmark access guide
│   └── test_satellite.jpg         # Sample test satellite image (~1.9 MB)
├── evaluation/
│   ├── __init__.py
│   └── vrsbench_eval.py           # Bounded benchmark evaluator (VQA, caption, grounding)
├── inference/
│   ├── __init__.py
│   ├── model.py                   # Qwen3VLInference wrapper (4-bit loading & inference)
│   ├── predict.py                 # CLI entry point for all inference modes
│   └── structured_output.py       # Pydantic/dataclass schema normalization & fast_all parser
├── tests/
│   ├── test_feature_analysis.py   # Feature JSON parser tests
│   ├── test_structured_output.py  # Fast-all and unified schema tests
│   └── test_vrsbench.py           # VRSBench annotation normalization tests
├── app.py                         # Gradio web dashboard (loads model once)
├── run_app.bat                    # Windows one-click launcher
├── requirements.txt               # Pinned project dependencies
├── .gitignore                     # Git exclusion rules (venv, caches, models, secrets)
└── README.md                      # Comprehensive project documentation
```

---

## Testing

Run the automated test suite using `pytest`:

```powershell
.venv\Scripts\python.exe -m pytest -v tests/
```

All 12 unit tests validate:
* Markdown code-fence stripping and JSON parsing
* Unified structured output serialization
* Single-dict and alias handling in `fast_all`
* VRSBench corner/bounding box coordinate normalization (`0–100` to `0–1`)

To verify syntax and compilation across all core files:
```powershell
.venv\Scripts\python.exe -m py_compile app.py inference\model.py inference\predict.py
```

---

## Performance Information

| Operation | Time | Notes |
|---|---|---|
| **Model Pre-Loading** | ~95 – 105s | Executed **only once** at startup |
| **Image Processing** | ~0.35 – 0.45s | PIL Lanczos thumbnail downsampling (512px) |
| **Autoregressive Generation (`--fast-all`)** | **~12.5 – 15.5s** | Single generation, max 128 new tokens |
| **Total Query Latency (`--fast-all`)** | **~13.0 – 16.0s** | Zero model reloading overhead on subsequent clicks |
| **GPU VRAM Consumption** | **~2.77 GB** | Fits comfortably in 4.0 GB VRAM (~1.26 GB headroom) |

---

## VRSBench Benchmark Status

The repository includes a complete local loader and evaluator for the official **VRSBench** benchmark (*Li et al., 2024*):
* **Loader (`datasets/vrsbench/loader.py`):** Supports reading local extractions or streaming annotations from Hugging Face `xiang709/VRSBench`.
* **Evaluator (`evaluation/vrsbench_eval.py`):** Calculates exact match for VQA, ROUGE-L F1 for captions, and mean IoU for visual grounding boxes.
* **Safety:** The evaluator processes only a requested `--limit` sample count and never attempts to download full multi-gigabyte image archives automatically.

---

## Troubleshooting

### 1. `bitsandbytes` quantization unavailable / fails
* Ensure you are running Python from `.venv` where `bitsandbytes` is installed.
* Verify your NVIDIA GPU driver is updated.
* If running on Windows without pre-built wheels, ensure `bitsandbytes>=0.45.0` is installed via `pip install bitsandbytes`.

### 2. CUDA Out of Memory (OOM)
* Close background 3D applications or games using VRAM.
* The model requires ~2.8 GB VRAM. If your display consumes >1.2 GB of VRAM, launch without other GPU-intensive tasks open.

### 3. Warning: `triton not found; flop counting will not work`
* This warning is harmless on Windows and does not affect model loading or inference speed.

### 4. `TypeError: Blocks.launch() got an unexpected keyword argument`
* Make sure you are using the installed Gradio 6.x configuration in `app.py` (which passes `css` directly to `demo.launch()`).

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details. Model weights for `Qwen3-VL-4B-Instruct` are governed by Alibaba's Qwen license agreement.
