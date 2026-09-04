PERSON 2 — SATELLITE DATA / PREPROCESSING

WORK:
Build the satellite-data pipeline that provides clean imagery to the AI.

Handle:
- Sentinel-1 SAR
- Sentinel-2 Optical
- Before/after images
- Optical + SAR pairs
- GeoTIFF files
- Image alignment
- Resizing/tiling
- Metadata
- Coordinates
- Dates

TOOLS:
- Google Earth Engine
- Sentinel Hub
- Sentinel-1
- Sentinel-2
- Python
- Rasterio
- GDAL
- GeoPandas
- NumPy

FOLDERS:

data/
Satellite images and datasets.
Do not commit huge datasets to GitHub.

preprocessing/
Code for cleaning, converting, aligning and preparing images.

tests/
Test that the processed satellite data is correct.

FINAL OUTPUT:
Provide clean images + metadata that the other AI components can directly use.