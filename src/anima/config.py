import logging
import logging.config
from dataclasses import dataclass, field
from pathlib import Path
from .utils.nrc_vad_lexicon import ensure_nrc_lexicon


# ---------------------------------------------------------------------------
# Third-party loggers: name → the minimum verbosity level required to unlock.
# Ordered from least to most noisy — verbosity is cumulative (>= threshold).
# ---------------------------------------------------------------------------
_THIRD_PARTY_VERBOSITY: dict[str, int] = {
    "httpx":    1,   # -v:   clean "HTTP Request: ..." summary lines
    "fsspec":   2,   # -vv:  local file open/read events
    "filelock": 2,   # -vv:  lock acquire/release events
    "httpcore": 3,   # -vvv: full HTTP lifecycle (very noisy)
}


def _apply_logging(debug: bool, verbosity: int) -> None:
    """
    Configure logging for Anima.

    debug=True  → your code logs at DEBUG; third-party stays quiet by default.
    verbosity   → independently unlocks noisy third-party loggers:
                  0  all third-party quiet (default)
                  1  httpx INFO summaries
                  2  + fsspec, filelock DEBUG
                  3  + httpcore full HTTP lifecycle
    """
    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s %(levelname)-8s %(name)s | %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
            },
        },
        "loggers": {
            "anima": {
                "level": logging.DEBUG if debug else logging.INFO,
                "propagate": True,
            }
        },
    })

    # Third-party loggers stay quiet regardless of debug,
    # unlocking only when verbosity meets their threshold.
    for name, unlocks_at in _THIRD_PARTY_VERBOSITY.items():

        if verbosity < unlocks_at:
            level = logging.WARNING

        elif name == "httpx":
            # httpx: INFO at -v/-vv, DEBUG only at -vvv
            level = logging.DEBUG if verbosity >= 3 else logging.INFO

        else:
            # everyone else: INFO until -vvv
            level = logging.DEBUG if verbosity >= 3 else logging.INFO

        logging.getLogger(name).setLevel(level)



@dataclass
class Config:
    """
    Configuration for Anima.
    All tuning parameters live here.
    Mutate before calling engine.start().
    """
    # ---------------------------------------------------------
    # Core Emotional Tuning
    # ---------------------------------------------------------
    passion: float = 2.25
    drama: float = 0.65
    default_influence: float = 0.22
    baseline_influence: float = 0.70
    dwell_seconds: float = 1.2
    hold_seconds: float = 0.8
    decay_rate: float = 0.06
    tick_rate_hz: float = 10.0
    # ---------------------------------------------------------
    # Baseline Blending
    # ---------------------------------------------------------
    drift_strength: float = 0.04
    drift_damping: float = 0.90
    max_velocity: float = 0.05
    # ---------------------------------------------------------
    # Bounds
    # ---------------------------------------------------------
    min_vadcc: float = 0.0
    max_vadcc: float = 1.0
    # ---------------------------------------------------------
    # NLP Models
    # ---------------------------------------------------------
    nrc_lexicon_path: str = "data/NRC-VAD-Lexicon-v2.1/NRC-VAD-Lexicon-v2.1.txt"
    sentence_model_name: str = "all-MiniLM-L6-v2"
    # ---------------------------------------------------------
    # Anchor (Slow Baseline)
    # ---------------------------------------------------------
    enable_anchor: bool = False
    anchor_model_name: str = "nemotron-mini:4b-instruct-q5_K_M"
    warm_baseline_on_start: bool = True
    baseline_window_words: int = 100
    # ---------------------------------------------------------
    # Context Tracking
    # ---------------------------------------------------------
    context_buffer_words: int = 200
    # ---------------------------------------------------------
    # Logging
    # ---------------------------------------------------------
    debug: bool = False
    log_verbosity: int = 0
    # Controls third-party logger noise, independent of debug.
    # Third-party loggers stay quiet even when debug=True unless verbosity is raised.
    #   0  quiet across the board (default)
    #   1  httpx request summaries  (-v)
    #   2  + fsspec, filelock       (-vv)
    #   3  + httpcore full lifecycle (-vvv, very noisy)
    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------
    def validate(self) -> None:
        if not (0.0 <= self.passion <= 5.0):
            raise ValueError("passion must be between 0 and 5")
        if not (0.0 <= self.drama <= 1.0):
            raise ValueError("drama must be between 0 and 1")
        if self.tick_rate_hz <= 0:
            raise ValueError("tick_rate_hz must be positive")
        if self.min_vadcc >= self.max_vadcc:
            raise ValueError("min_vadcc must be < max_vadcc")
        if not (0 <= self.log_verbosity <= 3):
            raise ValueError("log_verbosity must be between 0 and 3")
        ensure_nrc_lexicon(self.nrc_lexicon_path)
        if not Path(self.nrc_lexicon_path).exists():
            raise FileNotFoundError(
                f"NRC lexicon not found at {self.nrc_lexicon_path}"
            )

    def apply_logging(self) -> None:
        """Configure logging based on this config. Call once at startup."""
        _apply_logging(self.debug, self.log_verbosity)
        
    def set_log_verbosity(self, verbosity: int) -> None:
        self.log_verbosity = verbosity
        _apply_logging(self.debug, verbosity)
