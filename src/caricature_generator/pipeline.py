"""End-to-end caricature generation pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

from .config import PipelineConfig
from .logging_utils import configure_logging, get_logger
from .models.diffusers_wrapper import DiffusersCaricatureModel, DiffusersInput
from .postprocessing import PostProcessingPipeline
from .preprocessing import ImageLoader, PreprocessingPipeline

logger = get_logger(__name__)


@dataclass
class PipelineArtifact:
    """Represents an output produced by the pipeline."""

    input_path: Path
    output_path: Path
    metadata: dict[str, object]


class CaricaturePipeline:
    """Coordinates ingestion, pre-processing, generation and post-processing."""

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        configure_logging(config.logging.level, config.logging.log_dir)

        self._loader = ImageLoader(config.input_dir)
        self._preprocess = PreprocessingPipeline(config.preprocessing)
        self._generator = DiffusersCaricatureModel(config.model, device=config.device)
        self._postprocess = PostProcessingPipeline(config.postprocessing)

    def _prompts(self) -> Sequence[DiffusersInput]:
        return [
            DiffusersInput(
                prompt="Tasteful, artistic caricature, clean lines, professional illustration",
                negative_prompt="distorted, grotesque, low quality, nsfw",
                strength=0.65,
                guidance_scale=self._config.model.guidance_scale,
            )
        ]

    def run(self) -> List[PipelineArtifact]:
        logger.info("Starting caricature generation pipeline")
        artifacts: List[PipelineArtifact] = []

        for batch in self._loader.load():
            logger.info("Processing %s", batch.path.name)
            processed = self._preprocess.process(batch)
            prompts = self._prompts()
            generated_images = self._generator.generate(processed.image, prompts)

            for idx, generated in enumerate(generated_images):
                composed = self._postprocess.apply(generated, batch.image)
                base_name = batch.path.stem
                name = f"{base_name}_caricature_{idx}"
                output_path = self._config.output_dir / name
                saved_path = self._postprocess.save(composed, output_path)

                metadata = {
                    **batch.metadata,
                    **processed.metadata,
                    "prompt": prompts[idx % len(prompts)].prompt,
                }
                artifacts.append(
                    PipelineArtifact(input_path=batch.path, output_path=saved_path, metadata=metadata)
                )
                logger.info("Saved caricature to %s", saved_path)

        logger.info("Pipeline completed with %s artifacts", len(artifacts))
        return artifacts
