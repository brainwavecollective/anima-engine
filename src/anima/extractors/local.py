"""Nemotron-tuned Ollama baseline extraction (Jetson optimized)"""

import logging
import re
from typing import List, Optional

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

logger = logging.getLogger(__name__)


class LocalExtractor:
    """
    Deterministic numeric baseline extraction using Nemotron.
    """

    def __init__(self, model: str = "nemotron-mini:4b-instruct-q5_K_M"):
        self.model = model
        self.available = OLLAMA_AVAILABLE

        if not self.available:
            logger.warning("Ollama not available - baseline will use fast-only mode")

    def extract_baseline(self, text: str) -> Optional[List[float]]:
        if not self.available:
            return None

        full_prompt = f"""You are a machine-to-machine emotional climate scoring engine.

Evaluate the LONG-TERM emotional climate of the text.

This is not momentary emotion.
This is not expressive intensity.
This represents the stable underlying mood.

You must output five values in this exact order:

1) Valence — negative (0.0) to positive (1.0)
2) Arousal — calm (0.0) to energized (1.0)
3) Dominance — passive/weak (0.0) to powerful/controlled (1.0)
4) Complexity — minimal/simple (0.0) to rich/visually dense (1.0)
5) Coherence — chaotic/disorganized (0.0) to harmonious/ordered (1.0)

Neutral baseline for all dimensions is 0.5.

Only move away from 0.5 if the text shows a clear sustained emotional direction.

If tone is mixed, uncertain, or mild — stay near 0.5.

Weight recent text more heavily than earlier text.
If emotional tone has shifted recently, reflect the shift.

All values must remain between 0.2 and 0.8.
Never exceed this range.

Deviation magnitude should reflect sustained emotional strength:
0.52–0.58 = mild climate
0.58–0.68 = clear climate
0.68–0.8  = dominant sustained climate

Return EXACTLY five floating point numbers separated by spaces.
No words.
No explanations.
No commas.
No brackets.

Example valid output:
0.25 0.48 0.82 0.40 0.70

Text:
{text}

Output:
"""

        try:
            response = ollama.generate(
                model=self.model,
                prompt=full_prompt,
                options={
                    "temperature": 0.0,  
                    "top_k": 1,          
                    "num_predict": 42,   
                    "repeat_penalty": 1.05,
                    "stop": ["\n"],      
                    "num_ctx": 512,       
                    "num_thread": 2,       
                }
            )

            output = response["response"].strip()

            # Defensive float extraction for a chatty model
            numbers = re.findall(r"0\.\d+|1\.0+|[01]", output)
            baseline = [float(n) for n in numbers[:5]]

            if len(baseline) == 5 and all(0.0 <= v <= 1.0 for v in baseline):
                logger.info(
                    f"Baseline updated: "
                    f"V:{baseline[0]:.2f} "
                    f"A:{baseline[1]:.2f} "
                    f"D:{baseline[2]:.2f} "
                    f"Cx:{baseline[3]:.2f} "
                    f"Ch:{baseline[4]:.2f}"
                )
                return baseline

            logger.warning(f"Invalid Nemotron response: {output}")
            return None

        except Exception as e:
            logger.error(f"Baseline extraction failed: {e}")
            return None

