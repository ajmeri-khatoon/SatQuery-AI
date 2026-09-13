"""VRSBench dataset access and schema normalization."""

from .loader import (
    GroundingAnnotation,
    VQAAnnotation,
    VRSBenchLoader,
    VRSBenchSample,
)

__all__ = [
    "GroundingAnnotation",
    "VQAAnnotation",
    "VRSBenchLoader",
    "VRSBenchSample",
]
