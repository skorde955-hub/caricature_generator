"""Caricature Generator package."""

from importlib.metadata import version

__all__ = ["__version__"]


def __getattr__(name: str) -> str:
    if name == "__version__":
        try:
            return version("caricature-generator")
        except Exception:
            return "0.1.0"
    raise AttributeError(name)

