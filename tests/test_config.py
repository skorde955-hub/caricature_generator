from pathlib import Path

from caricature_generator.config import PipelineConfig


def test_load_default_config(tmp_path: Path) -> None:
    cfg = PipelineConfig.load(Path("configs/default.yaml"))
    assert cfg.input_dir == Path("examples/input")
    assert cfg.output_dir == Path("examples/output")
    assert cfg.model.backend == "diffusers"
    assert cfg.preprocessing.image_size == 512

