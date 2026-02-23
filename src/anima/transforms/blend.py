import numpy as np
from typing import List

from ..config import Config
from ..utils.time_source import TimeSource


class Blend:
    """
    Attack–Hold–Decay emotional state system.

    - Bursts are shown at full strength
    - Held for perceptual clarity
    - Then decay back toward baseline
    - Each burst nudges the baseline slightly
    """

    def __init__(self, config: Config, time_source: TimeSource):
        self.config = config
        self.clock = time_source

        # Initial baseline: neutral emotional field
        self.baseline = np.array([0.5, 0.5, 0.5, 0.5, 0.8], dtype=float)
        
        self.current = self.baseline.copy()
        self.velocity = np.zeros_like(self.current)

        self.hold_until = 0.0
        self.dwell_until = 0.0
        self.last_baseline_update = self.clock.now()

    # ---------------------------------------------------------
    # Burst application (fast emotional shift)
    # ---------------------------------------------------------

    def apply_burst(self, burst: List[float], influence: float | None = None):
        burst_array = np.array(burst, dtype=float)

        # Immediately show full emotion
        self.current = burst_array.copy()
        self.velocity[:] = 0.0

        now = self.clock.now()

        # Dwell + hold timing
        self.dwell_until = now + self.config.dwell_seconds
        self.hold_until = self.dwell_until + self.config.hold_seconds

        # Nudge baseline toward burst
        alpha = influence if influence is not None else self.config.default_influence

        self.baseline += (self.current - self.baseline) * alpha
        self.baseline = np.clip(
            self.baseline,
            self.config.min_vadcc,
            self.config.max_vadcc,
        )

    # ---------------------------------------------------------
    # Slow baseline correction (anchor-driven)
    # ---------------------------------------------------------

    def apply_baseline(self, new_baseline: List[float]):
        """
        Apply slow baseline update.
            - This does NOT trigger a burst.
            - Baseline acts as an attractor field.
            - Current state will naturally decay toward it via tick().
        """

        self.baseline = np.array(new_baseline, dtype=float)

        self.baseline = np.clip(
            self.baseline,
            self.config.min_vadcc,
            self.config.max_vadcc,
        )

        self.last_baseline_update = self.clock.now()

    # ---------------------------------------------------------
    # Tick update (~config.tick_rate_hz)
    # ---------------------------------------------------------

    def tick(self) -> List[float]:
        now = self.clock.now()

        # During dwell + hold plateau
        if now < self.hold_until:
            return self.current.tolist()

        # pull toward baseline
        delta = self.baseline - self.current
        self.velocity += delta * self.config.drift_strength

        # damping (friction)
        self.velocity *= self.config.drift_damping

        # safety clamp
        self.velocity = np.clip(
            self.velocity,
            -self.config.max_velocity,
            self.config.max_velocity,
        )

        # move
        self.current += self.velocity

        # Clip for safety
        self.current = np.clip(
            self.current,
            self.config.min_vadcc,
            self.config.max_vadcc,
        )

        return self.current.tolist()

    # ---------------------------------------------------------
    # Debug state inspection
    # ---------------------------------------------------------

    def get_state(self):
        return {
            "baseline": self.baseline.tolist(),
            "current": self.current.tolist(),
            "hold_remaining": max(0.0, self.hold_until - self.clock.now()),
            "time_since_baseline_update": self.clock.elapsed_since(self.last_baseline_update),
        }
