"""Configuration models for the caricature generation pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, validator


class ModelConfig(BaseModel):
    backend: str = Field(default="diffusers", description="Name of the generator backend")
    pretrained_model: str = Field(
        default="SG161222/Realistic_Vision_V5.1_noVAE", description="Model checkpoint or repo id"
    )
    scheduler: str = Field(default="DPMSolverMultistepScheduler", description="Diffusion scheduler")
    guidance_scale: float = Field(default=7.5, ge=0.0, le=20.0)
    num_inference_steps: int = Field(default=25, ge=1, le=150)


class PreprocessingConfig(BaseModel):
    image_size: int = Field(default=512, ge=64, le=2048)
    align_faces: bool = Field(default=True)
    background_mode: str = Field(default="preserve")
    safety_filter: bool = Field(default=True)
    face_detector: str = Field(default="mediapipe")

    @validator("background_mode")
    def validate_background_mode(cls, value: str) -> str:
        allowed = {"preserve", "remove", "blur"}
        if value not in allowed:
            raise ValueError(f"background_mode must be one of {allowed}")
        return value


class PostprocessingConfig(BaseModel):
    blend_alpha: float = Field(default=0.75, ge=0.0, le=1.0)
    output_format: str = Field(default="png")
    upscale: int = Field(default=1, ge=1, le=4)

    @validator("output_format")
    def validate_output_format(cls, value: str) -> str:
        allowed = {"png", "jpg", "webp"}
        if value not in allowed:
            raise ValueError(f"output_format must be one of {allowed}")
        return value


class LoggingConfig(BaseModel):
    level: str = Field(default="INFO")
    log_dir: Path = Field(default=Path("logs"))


class PipelineConfig(BaseModel):
    input_dir: Path = Field(default=Path("examples/input"))
    output_dir: Path = Field(default=Path("examples/output"))
    batch_size: int = Field(default=4, ge=1, le=32)
    device: str = Field(default="cuda")
    model: ModelConfig = Field(default_factory=ModelConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    postprocessing: PostprocessingConfig = Field(default_factory=PostprocessingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineConfig":
        return cls(**data)

    @classmethod
    def load(cls, path: Path) -> "PipelineConfig":
        with path.open("r", encoding="utf-8") as fh:
            payload = yaml.safe_load(fh)
        return cls.from_dict(payload or {})

