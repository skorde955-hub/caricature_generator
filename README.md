# Caricature Generator

Tasteful caricature generation pipeline built with Python. The repository scaffolds the data ingestion, pre-processing and generative modelling stages needed to transform input portraits into stylised caricatures while remaining easy to extend with custom models or deployment targets.

## Project Goals

- **Automate ingestion** of single images or batches while capturing metadata needed for downstream processing.
- **Normalise and prepare** faces (alignment, background handling, scaling) to feed into generative backends consistently.
- **Generate caricatures** via a modular interface that can wrap diffusion, GAN or neural warping models.
- **Post-process outputs** to ensure aesthetic quality (color correction, compositing, output sizing).
- **Remain production-ready** with configuration management, logging, testing hooks and reproducible environments.

## High-Level Architecture

1. **Ingestion:** Load local files or URLs, normalise to the internal representation and persist tracking metadata.
2. **Pre-processing:** Detect facial landmarks, align faces, segment foreground/background and apply safety filters.
3. **Generation:** Feed prepared tensors to a caricature model wrapper (e.g., CartoonGAN, Diffusers fine-tune, control-net).
4. **Post-processing:** Blend stylised foreground back onto curated backgrounds, apply tone adjustments and export assets.
5. **Orchestration:** The `CaricaturePipeline` coordinates steps using dependency-injected components so alternative models or pre-processing strategies can be swapped by configuration.

## Tech Stack

- **Python 3.11** runtime.
- **PyTorch** + **Diffusers** for generative modelling (easily swap for CartoonGAN or custom backends).
- **OpenCV**, **mediapipe**, **numpy**, **Pillow** for classical vision operations and landmark detection.
- **Hydra / pydantic** configuration pattern via YAML configs to keep experiments reproducible.
- **FastAPI** stub (optional future work) for service deployment.
- Tooling: `poetry` for packaging, `ruff` + `black` for lint/format, `pytest` for testing.

## Repository Layout

```text
.
├── README.md                Project overview and architecture notes
├── pyproject.toml           Poetry-based dependency and tooling config
├── configs/
│   └── default.yaml         Default pipeline configuration
├── scripts/
│   └── run_pipeline.py      CLI entry point for batch caricature generation
└── src/
    └── caricature_generator/
        ├── __init__.py
        ├── config.py
        ├── logging_utils.py
        ├── models/
        │   ├── __init__.py
        │   └── diffusers_wrapper.py
        ├── pipeline.py
        ├── preprocessing/
        │   ├── __init__.py
        │   ├── facial_landmarks.py
        │   ├── image_loader.py
        │   └── transforms.py
        └── postprocessing/
            ├── __init__.py
            └── compositing.py
```

## Getting Started

```bash
poetry install
poetry run python scripts/run_pipeline.py \
  --input ./examples/input \
  --output ./examples/output
```

If you prefer virtual environments without Poetry, export dependencies:

```bash
poetry export -f requirements.txt --output requirements.txt --without-hashes
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Roadmap

- Integrate real caricature checkpoints (CartoonDiffusion, WarpGAN) behind the model wrapper.
- Add FastAPI microservice for serving generated caricatures on demand.
- Extend evaluation metrics (FID, stylisation score) and automated QA.
- Provide CI presets (GitHub Actions) with linting, tests and optional GPU inference smoke checks.

