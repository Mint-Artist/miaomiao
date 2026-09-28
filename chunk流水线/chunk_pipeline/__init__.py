"""Versioned, lossless HTML chunking experiments."""

__version__ = '0.2.0'

from .model import ChunkConfig, PipelineError
from .normalize import normalize_html

__all__ = ['ChunkConfig', 'PipelineError', 'normalize_html']
