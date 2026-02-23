"""Anima - Real-time emotional affect engine for robots."""

from .anima import Anima
from .config import Config
from .utils.time_source import TimeSource
from .telemetry import AnimaTelemetryWriter

__all__ = ["Anima", "Config", "TimeSource", "AnimaTelemetryWriter"]