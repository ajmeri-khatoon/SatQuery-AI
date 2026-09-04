PERSON 3 — REMOTE-SENSING VISION AI

WORK:
Build the AI that understands a single satellite image.

Main capabilities:
- Visual Question Answering
- Image captioning
- Object/feature understanding
- Visual grounding
- Satellite-specific image understanding

TOOLS:
- Qwen3-VL
- Hugging Face
- PyTorch
- PEFT / LoRA
- VRSBench

FOLDERS:

models/
Model files/configurations.

inference/
Code that takes an image + question and produces an answer.

datasets/
VRSBench and other remote-sensing datasets.

tests/
Test model responses and performance.

FINAL OUTPUT:
Input:
Image + question

Output:
- Answer
- Detected objects/features
- Location/evidence where possible
- Confidence