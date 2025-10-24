"""Caricature Generator package."""

from importlib.metadata import version

__all__ = ["__version__"]

try:  # pragma: no cover - defensive shim for newer huggingface_hub
    import huggingface_hub
    if not hasattr(huggingface_hub, "cached_download"):
        from huggingface_hub import hf_hub_download

        def _cached_download(*args, **kwargs):
            return hf_hub_download(*args, **kwargs)

        huggingface_hub.cached_download = _cached_download  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - best effort shim
    pass


def __getattr__(name: str) -> str:
    if name == "__version__":
        try:
            return version("caricature-generator")
        except Exception:
            return "0.1.0"
    raise AttributeError(name)
