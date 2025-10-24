"""Typer CLI for running the caricature generator pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from .config import PipelineConfig
from .pipeline import CaricaturePipeline

app = typer.Typer(help="Generate tasteful caricatures from input images.")


@app.command()
def run(
    config: Path = typer.Option(
        Path("configs/default.yaml"), "--config", "-c", help="Path to pipeline configuration."
    ),
    input_dir: Optional[Path] = typer.Option(
        None, "--input", "-i", help="Override input directory set in the config."
    ),
    output_dir: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Override output directory set in the config."
    ),
    device: Optional[str] = typer.Option(
        None, "--device", "-d", help="Override compute device (cuda/cpu)."
    ),
) -> None:
    """Run the full caricature generation pipeline."""
    cfg = PipelineConfig.load(config)
    overrides = {}
    if input_dir:
        overrides["input_dir"] = input_dir
    if output_dir:
        overrides["output_dir"] = output_dir
    if device:
        overrides["device"] = device

    if overrides:
        cfg = cfg.model_copy(update=overrides)

    pipeline = CaricaturePipeline(cfg)
    artifacts = pipeline.run()
    typer.echo(f"Generated {len(artifacts)} caricature(s). Outputs stored in {cfg.output_dir}.")


def main() -> None:
    app()


if __name__ == "__main__":
    main()

