import asyncio
import numpy as np
import pytest

from anima import Anima, Config


def build_anchor_engine():
    config = Config(
        enable_anchor=True,
        warm_baseline_on_start=False,
        debug=True,
    )
    config.apply_logging()
    return Anima(config)


async def wait_for_baseline_task(engine, timeout=180):
    async def _wait():
        while engine._baseline_task is None:
            await asyncio.sleep(0.05)
        await engine._baseline_task

    await asyncio.wait_for(_wait(), timeout=timeout)


@pytest.mark.anyio
async def test_anchor_updates_baseline():
    engine = build_anchor_engine()
    await engine.start()

    try:
        text = (
            "I have been thinking deeply about everything lately. "
            "Some moments feel heavy, others calm and reflective."
        )

        baseline_before = engine.blend.baseline.copy()

        # ---- FIRST PASS (old baseline) ----
        result_before = await engine.process_text(text)
        anima_before = np.array(result_before["anima"])

        # wait for anchor completion
        await wait_for_baseline_task(engine, timeout=180)

        baseline_after = engine.blend.baseline.copy()

        # confirm baseline changed
        assert not np.allclose(
            baseline_before,
            baseline_after,
        ), "Anchor did not update baseline"

        # ---- SECOND PASS (new baseline) ----
        result_after = await engine.process_text(text)
        anima_after = np.array(result_after["anima"])

        # compare outputs
        assert not np.allclose(
            anima_before,
            anima_after,
        ), "Output did not change after baseline update"

        print("\nBEFORE:", anima_before)
        print("AFTER :", anima_after)

    finally:
        await engine.stop()
