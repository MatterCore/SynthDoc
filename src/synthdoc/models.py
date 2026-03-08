# This module is deprecated. All models are now in synthdoc.config.
# Kept for backward compatibility during migration.
from synthdoc.config import (
    ContentConfig,
    DegradationConfig,
    GenerationConfig,
    LayoutConfig,
    PageConfig,
)

__all__ = [
    "PageConfig",
    "LayoutConfig",
    "ContentConfig",
    "DegradationConfig",
    "GenerationConfig",
]
