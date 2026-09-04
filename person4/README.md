PERSON 4 — CHANGE DETECTION + OPTICAL/SAR

WORK:
Build the specialist AI for:
1. Before/after change detection
2. Optical + SAR analysis

CHANGE DETECTION:

Input:
Image A + Image B

Output:
- Change mask
- Changed regions
- Change percentage
- Type of change
- Confidence

OPTICAL + SAR:

Input:
Sentinel-2 optical + Sentinel-1 SAR

Output:
- Optical analysis
- SAR analysis
- Combined interpretation
- Confidence/evidence

TOOLS:
- Open-CD
- PyTorch
- Sentinel-1
- Sentinel-2
- BigEarthNet
- Change-detection datasets

FOLDERS:

change_detection/
All before/after analysis.

optical_sar/
Optical + SAR analysis/fusion.

models/
Models used for these tasks.

tests/
Test change detection and optical/SAR results.

FINAL OUTPUT:
Provide results in a format that Person 5 can connect directly to the backend and Person 1's agent can call as a tool.