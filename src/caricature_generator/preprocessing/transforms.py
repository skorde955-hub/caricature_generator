"""Image transformation routines prior to generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from PIL import Image, ImageFilter, ImageOps

from ..config import PreprocessingConfig
from ..logging_utils import get_logger
from .facial_landmarks import FacialLandmarkDetector, LandmarkResult
from .image_loader import ImageBatch

logger = get_logger(__name__)


@dataclass
class ProcessedImage:
    """Return type from the preprocessing pipeline."""

    image: Image.Image
    metadata: dict[str, object]


class PreprocessingPipeline:
    """Apply alignment, resizing and cosmetic filters before generation."""

    def __init__(self, config: PreprocessingConfig) -> None:
        self._config = config
        self._landmark_detector: Optional[FacialLandmarkDetector] = (
            FacialLandmarkDetector() if config.align_faces else None
        )

    def _align_face(self, image: Image.Image, landmarks: LandmarkResult) -> Image.Image:
        """Align the face by simply centering the bounding box."""
        xs = [x for x, _ in landmarks.landmarks]
        ys = [y for _, y in landmarks.landmarks]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        crop = image.crop((min_x, min_y, max_x, max_y))
        logger.debug("Cropping face to region (%s, %s, %s, %s)", min_x, min_y, max_x, max_y)
        return crop

    def _background_filter(self, image: Image.Image) -> Image.Image:
        mode = self._config.background_mode
        if mode == "blur":
            return image.filter(ImageFilter.GaussianBlur(radius=4))
        if mode == "remove":
            # Placeholder for segmentation removal
            logger.debug("Background removal placeholder executed.")
        return image

    def process(self, batch: ImageBatch) -> ProcessedImage:
        image = batch.image.copy()
        metadata = dict(batch.metadata)

        if self._landmark_detector:
            try:
                landmarks = self._landmark_detector.detect(image)
                if landmarks:
                    image = self._align_face(image, landmarks)
                    metadata["landmarks_detected"] = True
                else:
                    metadata["landmarks_detected"] = False
            except RuntimeError as exc:
                logger.warning("Landmark detection skipped: %s", exc)
                metadata["landmarks_detected"] = False

        image = ImageOps.fit(
            image, (self._config.image_size, self._config.image_size), Image.Resampling.LANCZOS
        )

        image = self._background_filter(image)

        metadata["preprocessed"] = True
        metadata["target_size"] = self._config.image_size
        return ProcessedImage(image=image, metadata=metadata)

