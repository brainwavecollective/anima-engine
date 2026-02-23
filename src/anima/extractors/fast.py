import logging
import numpy as np
from pathlib import Path
from typing import List, Tuple

import re
import nltk
from sentence_transformers import SentenceTransformer
from textstat import flesch_kincaid_grade

from collections import OrderedDict
import threading

from ..utils.nrc_vad_lexicon import ensure_nrc_lexicon
import time


class FastExtractor:
    """
    Extract natural psychological VIBE from text.

    Outputs:
        [Valence, Arousal, Dominance, Complexity, Coherence]
        All scaled 0.0-1.0
    """

    # ---------------------------------------------------------
    # Negation handling
    # ---------------------------------------------------------

    CONTRACTION_MAP = {
        "doesn't": "does not", "don't": "do not", "won't": "will not",
        "isn't": "is not", "aren't": "are not", "wasn't": "was not",
        "weren't": "were not", "can't": "cannot", "couldn't": "could not",
        "wouldn't": "would not", "shouldn't": "should not",
        "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
        "didn't": "did not", "it's": "it is", "i'm": "i am",
        "you're": "you are",
    }

    NEGATION_WORDS = {
        "no", "not", "never", "neither", "nor", "nobody",
        "nothing", "nowhere", "hardly", "scarcely", "barely",
        "without", "cannot",
    }

    NEGATION_WINDOW = 3

    # ---------------------------------------------------------
    # Valence diluters
    # ---------------------------------------------------------

    # Words that score positive in the NRC lexicon but act as
    # connective/contextual tissue in sad or mixed sentences,
    # masking the true emotional content. Weight reduced to 30%.
    #
    # Problems they solve:
    #   "feel" V=0.80 dominates "lonely" V=0.25 in "feel lonely"
    #   "most" V=0.86 dominates in "make the most of this time"
    #   "together" V=0.87 overrides sad content in the same sentence
    VALENCE_DILUTERS = {
        # Connective/contextual positives that mask sad content
        "feel", "feels", "feeling", "felt",
        "most", "best",
        "together", "share",
        "hear", "know", "get", "see",
        # Intensifier adverbs — semantically neutral but lexically loaded
        # e.g. "pretty sad" reads as positive due to "pretty" V=0.875
        "pretty", "really", "quite", "very", "truly", "actually", "right",
    }

    DILUTER_WEIGHT_FACTOR = 0.3

    # ---------------------------------------------------------
    # Construction
    # ---------------------------------------------------------

    def __init__(
        self,
        lexicon_path: str,
        model_name: str,
        debug=False,
    ):
        self.debug = debug

        self.logger = logging.getLogger("anima.extractor")

        self.lexicon_path = Path(lexicon_path)

        ensure_nrc_lexicon(lexicon_path)

        if not self.lexicon_path.exists():
            raise FileNotFoundError(
                f"NRC-VAD lexicon not found at {self.lexicon_path}. "
                "This file is not distributable and must be placed manually."
            )

        self.logger.info(f"Loading NRC-VAD lexicon from {self.lexicon_path}")
        self.vad_lexicon = self._load_nrc_vad(self.lexicon_path)

        self.logger.info(f"Loading sentence model: {model_name}")
        self.embedder = SentenceTransformer(model_name)

        self._embedding_cache = OrderedDict()
        self._embedding_cache_lock = threading.Lock()
        self._embedding_cache_max = 512  # safe upper bound

        # Ensure tokenizer availability
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)

        self.logger.info("FastExtractor initialized")

    # ---------------------------------------------------------
    # NRC-VAD Lexicon Loading
    # ---------------------------------------------------------

    def _load_nrc_vad(self, path: Path) -> dict:
        """Load NRC-VAD lexicon: word -> [v, a, d]"""

        lexicon = {}

        with open(path, encoding="utf-8") as f:
            next(f)  # skip header

            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 4:
                    continue

                word = parts[0].lower()

                try:
                    # Convert from -1..1 to 0..1
                    v = (float(parts[1]) + 1.0) / 2.0
                    a = (float(parts[2]) + 1.0) / 2.0
                    d = (float(parts[3]) + 1.0) / 2.0

                    lexicon[word] = [v, a, d]

                except ValueError:
                    continue

        if not lexicon:
            raise RuntimeError("NRC-VAD lexicon loaded but empty.")

        self.logger.info(f"Loaded {len(lexicon)} NRC-VAD entries")

        return lexicon

    # ---------------------------------------------------------
    # Public Extraction API
    # ---------------------------------------------------------
    def extract(self, text: str) -> List[float]:
        if self.debug:
            t0 = time.perf_counter()
            self.logger.debug("EXTRACT start len=%d", len(text))

        v, a, d = self._get_vad(text)

        if self.debug:
            self.logger.debug("EXTRACT vad dt=%.3fs", time.perf_counter() - t0)

        complexity = self._get_complexity(text)

        if self.debug:
            self.logger.debug("EXTRACT complexity dt=%.3fs", time.perf_counter() - t0)

        coherence = self._get_coherence(text)

        if self.debug:
            self.logger.debug("EXTRACT coherence dt=%.3fs", time.perf_counter() - t0)

        result = [
            float(np.clip(v, 0, 1)),
            float(np.clip(a, 0, 1)),
            float(np.clip(d, 0, 1)),
            float(np.clip(complexity, 0, 1)),
            float(np.clip(coherence, 0, 1)),
        ]

        if self.debug:
            self.logger.debug("EXTRACT total dt=%.3fs", time.perf_counter() - t0)

        return result




    # ---------------------------------------------------------
    # VAD via Lexicon
    # ---------------------------------------------------------

    def _get_vad(self, text: str) -> Tuple[float, float, float]:
        """
        Extract weighted VAD from text.

        Three fixes over the original:

        1. Contraction expansion:
           "doesn't sound great" -> "does not sound great"
           Without this, "doesn't" tokenizes to "doesn" (neutral) + "t"
           (not in lexicon), making the negation completely invisible.

        2. Negation window:
           After any negation token, the next NEGATION_WINDOW words have
           their valence flipped: 1.0 - V. Arousal is softened by 50%
           rather than flipped ("not excited" != "calm").

        3. Weight capping + diluter downweighting:
           Original: weight = (intensity + 0.01)^2
           A single word like "great" (V=0.958) got weight 0.89, overriding
           everything else in the sentence.
           Fixed: weight = min((intensity + 0.01)^0.7, 0.8)
           Additionally, VALENCE_DILUTERS words are reduced to 30% weight
           to prevent connective positives from masking genuine sad content.
        """
        # Expand contractions so negation tokens become visible
        lowered = text.lower()
        for contraction, expansion in self.CONTRACTION_MAP.items():
            lowered = lowered.replace(contraction, expansion)

        words = re.findall(r"[a-z]+", lowered)

        vad_values = []
        weights = []
        negate_window = 0

        for w in words:

            if w in self.NEGATION_WORDS:
                negate_window = self.NEGATION_WINDOW
                continue

            if w in self.vad_lexicon:
                vad = list(self.vad_lexicon[w])

                if negate_window > 0:
                    vad[0] = 1.0 - vad[0]
                    vad[1] = 0.5 + (vad[1] - 0.5) * 0.5

                intensity = sum(abs(v - 0.5) for v in vad)
                wt = min((intensity + 0.01) ** 0.7, 0.8)

                if w in self.VALENCE_DILUTERS:
                    wt *= self.DILUTER_WEIGHT_FACTOR

                vad_values.append(vad)
                weights.append(wt)

            if negate_window > 0:
                negate_window -= 1


        if not vad_values:
            return 0.5, 0.5, 0.5

        vad_array = np.array(vad_values)
        weights_array = np.array(weights)

        weighted_vad = np.average(
            vad_array,
            axis=0,
            weights=weights_array,
        )

        return tuple(weighted_vad)

    # ---------------------------------------------------------
    # Complexity
    # ---------------------------------------------------------

    def _get_complexity(self, text: str) -> float:
        try:
            fk_grade = flesch_kincaid_grade(text)
            return min(1.0, max(0.0, fk_grade / 20.0))
        except Exception:
            return 0.5

    # ---------------------------------------------------------
    # Coherence
    # ---------------------------------------------------------
    def _get_coherence(self, text: str) -> float:
        sentences = nltk.sent_tokenize(text)

        if len(sentences) < 2:
            return 0.8

        embeddings = []

        for sentence in sentences:
            key = sentence.strip()

            with self._embedding_cache_lock:
                if key in self._embedding_cache:
                    emb = self._embedding_cache.pop(key)
                    self._embedding_cache[key] = emb  # maintain LRU order
                    embeddings.append(emb)

                    if self.debug:
                        self.logger.debug("EMBED CACHE HIT: '%s'", key[:60])

                    continue

            # Cache miss
            if self.debug:
                self.logger.debug("EMBED CACHE MISS: '%s'", key[:60])
                t0 = time.perf_counter()

            emb = self.embedder.encode([sentence])[0]

            if self.debug:
                self.logger.debug(
                    "EMBED COMPUTE %.3fs: '%s'",
                    time.perf_counter() - t0,
                    key[:60],
                )

            with self._embedding_cache_lock:
                self._embedding_cache[key] = emb
                if len(self._embedding_cache) > self._embedding_cache_max:
                    self._embedding_cache.popitem(last=False)

            embeddings.append(emb)

        similarities = [
            np.dot(embeddings[i], embeddings[i + 1])
            for i in range(len(embeddings) - 1)
        ]

        coherence = (np.mean(similarities) + 1) / 2
        return float(coherence)

