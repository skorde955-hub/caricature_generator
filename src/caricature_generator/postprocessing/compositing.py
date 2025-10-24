"""Post-processing routines for polished outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PIL import Image

from ..config import PostprocessingConfig
from ..logging_utils import get_logger

logger = get_logger(__name__)


class PostProcessingPipeline:
    """Apply compositing and exporting to generator outputs."""

    def __init__(self, config: PostprocessingConfig) -> None:
        self._config = config

    def apply(self, stylised: Image.Image, original: Optional[Image.Image] = None) -> Image.Image:
        result = stylised.convert("RGB")

        if original and self._config.blend_alpha < 1.0:
            logger.debug("Blending stylised output with original at alpha=%s", self._config.blend_alpha)
            resized_original = original.resize(result.size)
            result = Image.blend(resized_original, result, alpha=self._config.blend_alpha)

        if self._config.upscale > 1:
            new_size = tuple(dim * self._config.upscale for dim in result.size)
            logger.debug("Upscaling output to %s", new_size)
            result = result.resize(new_size, Image.Resampling.LANCZOS)

        return result

    def save(self, image: Image.Image, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        fmt = self._config.output_format.upper()
        result_path = destination.with_suffix(f".{self._config.output_format.lower()}")
        logger.debug("Saving post-processed image to %s", result_path)
        image.save(result_path, format=fmt)
        return result_path

