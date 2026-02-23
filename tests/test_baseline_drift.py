import numpy as np

from anima import Config
from anima.transforms.blend import Blend
from anima.utils.time_source import TimeSource


def test_blend_drift_moves_toward_baseline():
    """
    Pure drift test.

    Verifies:
    - drift reduces distance to baseline
    - motion occurs gradually over multiple ticks
    """

    config = Config()

    # make drift obvious + fast for testing
    config.dwell_seconds = 0.0
    config.hold_seconds = 0.0
    config.drift_strength = 0.20
    config.drift_damping = 0.85
    config.max_velocity = 1.0

    clock = TimeSource()
    blend = Blend(config, clock)

    # ---- setup known state ----
    blend.baseline = np.array([0.2, 0.8, 0.7, 0.4, 0.6])
    blend.current = np.array([0.8, 0.2, 0.3, 0.9, 0.9])

    # ensure velocity starts clean
    blend.velocity[:] = 0.0

    dist_start = np.linalg.norm(blend.baseline - blend.current)

    # ---- simulate ticks ----
    history = []

    for _ in range(40):
        blend.tick()
        history.append(blend.current.copy())

    dist_end = np.linalg.norm(blend.baseline - blend.current)

    # must move toward baseline
    assert dist_end < dist_start, (
        "Drift did not reduce distance to baseline"
    )

    # must actually move (not frozen)
    deltas = [
        np.linalg.norm(history[i + 1] - history[i])
        for i in range(len(history) - 1)
    ]

    # bonus: motion should damp over time
    early_motion = np.mean(deltas[:5])
    late_motion = np.mean(deltas[-5:])

    assert late_motion < early_motion, (
        "Drift did not damp over time (motion should decrease)"
    )


    assert any(d > 0 for d in deltas), (
        "Blend state did not move during drift"
    )
