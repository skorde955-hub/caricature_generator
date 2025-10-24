"""Diffusers backend for caricature generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionImg2ImgPipeline
from PIL import Image

from ..config import ModelConfig
from ..logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class DiffusersInput:
    """Container for inputs passed to the diffusers pipeline."""

    prompt: str
    negative_prompt: Optional[str]
    strength: float
    guidance_scale: Optional[float] = None


class DiffusersCaricatureModel:
    """Wrapper around a Diffusers pipeline geared towards caricature generation."""

    def __init__(self, config: ModelConfig, device: str = "cuda") -> None:
        self._config = config
        self._device = self._resolve_device(device)
        self._pipeline: Optional[StableDiffusionImg2ImgPipeline] = None

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            return "cpu"
        return device

    def _load_pipeline(self) -> StableDiffusionImg2ImgPipeline:
        if self._pipeline:
            return self._pipeline

        logger.info("Loading diffusers pipeline %s", self._config.pretrained_model)
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            self._config.pretrained_model,
            safety_checker=None,
            torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
        )

        if self._config.scheduler != "DPMSolverMultistepScheduler":
            logger.warning(
                "Scheduler %s not explicitly supported. Falling back to DPMSolverMultistepScheduler.",
                self._config.scheduler,
            )

        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

        pipe.to(self._device)
        self._pipeline = pipe
        return pipe

    def generate(
        self, base_image: Image.Image, prompts: DiffusersInput | Iterable[DiffusersInput]
    ) -> list[Image.Image]:
        """Generate caricatures given one base image and one or more prompt bundles."""
        pipe = self._load_pipeline()
        if isinstance(prompts, DiffusersInput):
            prompts = [prompts]

        outputs: list[Image.Image] = []
        for prompt_bundle in prompts:
            logger.debug("Generating stylised image for prompt: %s", prompt_bundle.prompt)
            images = pipe(
                prompt=prompt_bundle.prompt,
                negative_prompt=prompt_bundle.negative_prompt,
                image=base_image,
                strength=prompt_bundle.strength,
                guidance_scale=prompt_bundle.guidance_scale
                or self._config.guidance_scale,
                num_inference_steps=self._config.num_inference_steps,
            ).images
            outputs.extend(images)
        return outputs
