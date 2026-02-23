"""Extractors create emotional state vectors from text."""

from .fast import FastExtractor
from .local import LocalExtractor

__all__ = ["FastExtractor", "LocalExtractor"]
