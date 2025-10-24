"""Facial landmark detection helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from PIL import Image

from ..logging_utils import get_logger

logger = get_logger(__name__)

try:
    import mediapipe as mp
except ImportError:  # pragma: no cover - optional dependency at runtime
    mp = None


Landmark = Tuple[float, float]


@dataclass
class LandmarkResult:
    """Stores landmark positions for a face."""

    landmarks: List[Landmark]
    score: float


class FacialLandmarkDetector:
    """Wrapper around MediaPipe Face Mesh detector."""

    def __init__(self, min_detection_confidence: float = 0.5) -> None:
        self._confidence = min_detection_confidence
        self._mesh = None

    def _ensure_model(self):
        if mp is None:
            raise RuntimeError(
                "mediapipe is not installed. Install it or disable landmark detection."
            )
        if self._mesh is None:
            logger.info("Initialising MediaPipe Face Mesh")
            self._mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=self._confidence,
                min_tracking_confidence=0.5,
            )

    def detect(self, image: Image.Image) -> LandmarkResult | None:
        """Return landmarks for the first detected face."""
        self._ensure_model()
        assert self._mesh is not None  # for type checker

        rgb_array = np.array(image.convert("RGB"))
        results = self._mesh.process(rgb_array)
        if not results.multi_face_landmarks:
            logger.warning("No face detected in image %s", image)
            return None

        height, width, _ = rgb_array.shape
        face_landmarks = results.multi_face_landmarks[0].landmark
        coords: List[Landmark] = [
            (landmark.x * width, landmark.y * height) for landmark in face_landmarks
        ]
        return LandmarkResult(landmarks=coords, score=1.0)

