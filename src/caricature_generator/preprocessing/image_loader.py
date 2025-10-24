"""Image ingestion utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterable

from PIL import Image, UnidentifiedImageError

from ..logging_utils import get_logger

logger = get_logger(__name__)


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


@dataclass
class ImageBatch:
    """Container for image data and metadata."""

    path: Path
    image: Image.Image
    metadata: dict[str, object]


class ImageLoader:
    """Load images from disk into PIL objects with metadata."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def list_files(self) -> Iterable[Path]:
        return sorted(
            path for path in self.root.rglob("*") if path.suffix.lower() in SUPPORTED_EXTENSIONS
        )

    def load(self) -> Generator[ImageBatch, None, None]:
        for path in self.list_files():
            try:
                image = Image.open(path).convert("RGB")
            except (FileNotFoundError, UnidentifiedImageError) as exc:
                logger.warning("Skipping %s: %s", path, exc)
                continue

            metadata = {
                "width": image.width,
                "height": image.height,
                "filesize": path.stat().st_size,
            }
            logger.debug("Loaded %s (%sx%s)", path.name, image.width, image.height)
            yield ImageBatch(path=path, image=image, metadata=metadata)

