"""Configuration package for Diabetic Retinopathy screening pipeline."""

from config.settings import (
    APP_VERSION,
    DR_GRADES,
    REFERABLE_GRADES,
    SUPPORTED_IMAGE_TYPES,
)

__all__ = [
    "DR_GRADES",
    "REFERABLE_GRADES",
    "SUPPORTED_IMAGE_TYPES",
    "APP_VERSION",
]
