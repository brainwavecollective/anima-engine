import logging
import pytest
from anima import Anima, Config


# ---------------------------------------------------------------------------
# Force pytest to show logs without needing -s
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
    force=True,  # IMPORTANT: overrides pytest capture setup
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_engine():
    config = Config(
        enable_anchor=False,
        debug=True,
    )
    config.apply_logging()
    return Anima(config)


async def get_valence(engine, text: str) -> float:
    result = await engine.process_text(text)

    assert "anima" in result
    assert len(result["anima"]) == 5

    return result["anima"][0]


# ---------------------------------------------------------------------------
# Emotional ordering smoke test
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_emotional_gradient_ordering():
    engine = build_engine()
    await engine.start()

    samples = {
        "neutral": "I went to the store and bought groceries today.",
        "sad": "I feel tired, disappointed, and emotionally drained.",
        "despair": "Nothing matters anymore. Everything feels empty and hopeless.",
        "ecstatic": "I feel unstoppable, euphoric, and overflowing with joy.",
        "happy": "Today is a good day. I’m relaxed, smiling, and content.",
    }

    values = {}

    for label, text in samples.items():
        values[label] = await get_valence(engine, text)

    await engine.stop()

    # ------------------------------------------------------------------
    # Print emotion summary (always visible)
    # ------------------------------------------------------------------

    print("\n=== EMOTIONAL VALUES ===")
    for k, v in values.items():
        print(f"{k:10} -> {v:.3f}")

    # ------------------------------------------------------------------
    # Core guarantees
    # ------------------------------------------------------------------

    assert values["happy"] > values["neutral"]
    assert values["ecstatic"] > values["neutral"]

    assert values["sad"] < values["neutral"]
    assert values["despair"] < values["neutral"]

    assert values["despair"] < values["sad"]
