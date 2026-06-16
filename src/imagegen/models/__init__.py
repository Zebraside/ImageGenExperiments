"""Model loading: a registry of base models + a uniform :class:`ModelBundle`."""

from imagegen.models.bundle import ModelBundle
from imagegen.models.factory import MODEL_REGISTRY, load_model

__all__ = ["ModelBundle", "MODEL_REGISTRY", "load_model"]
