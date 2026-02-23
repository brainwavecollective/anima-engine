"""anima.py — modified to integrate AnimaTelemetryWriter.

Changes from original:
- Accepts optional `telemetry_writer: AnimaTelemetryWriter` in __init__
- _extract_sentence_states_sync emits sentence rows when writer is present
- process_text emits utterance_final row when writer is present
- utterance_id is a short uuid generated per process_text call
- No behaviour changes when telemetry_writer is None
"""

import asyncio
import inspect
import logging
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Optional

import nltk
import numpy as np

from .transforms.cinematic import CinematicAmplifier
from .extractors.local import OLLAMA_AVAILABLE, LocalExtractor
from .transforms.blend import Blend
from .config import Config
from .extractors.fast import FastExtractor
from .utils.time_source import TimeSource
from .telemetry import (
    AnimaTelemetryWriter,
    make_sentence_record,
    make_utterance_final_record,
)


class Anima:
    """
    Real-time affect generation engine.

    Produces continuous VADCC values from text input.
    Owns its own loop and broadcasts state at fixed tick rate.
    """

    def __init__(self, config: Config, telemetry_writer: Optional[AnimaTelemetryWriter] = None):
        self.config = config
        self.config.validate()

        self.logger = logging.getLogger("anima")
        self.logger.setLevel(logging.DEBUG if self.config.debug else logging.INFO)

        # Components (loaded in start())
        self.extractor: Optional[FastExtractor] = None
        self.amplifier: Optional[CinematicAmplifier] = None
        self.anchor: Optional[LocalExtractor] = None
        self.blend: Optional[Blend] = None

        # Create time_source
        self.clock = TimeSource()

        # Runtime
        self._running = False
        self._loop_task: Optional[asyncio.Task] = None
        self._anchor_executor: Optional[ThreadPoolExecutor] = None

        self._subscribers: List[Callable[[List[float]], None]] = []

        self._context_buffer = deque(maxlen=self.config.context_buffer_words)
        self._transcript_count = 0

        # --- Baseline coordination (race-free) ---
        self._baseline_event = asyncio.Event()
        self._baseline_task: Optional[asyncio.Task] = None
        self._baseline_lock = asyncio.Lock()

        # --- Telemetry ---
        self._telemetry: Optional[AnimaTelemetryWriter] = telemetry_writer
        self._monotonic_start = time.monotonic()

    # ---------------------------------------------------------
    # Telemetry
    # ---------------------------------------------------------

    def set_telemetry_writer(self, writer: AnimaTelemetryWriter) -> None:
        """Attach or replace the telemetry writer after construction."""
        self._telemetry = writer

    def _monotonic_elapsed(self) -> float:
        return time.monotonic() - self._monotonic_start

    # ---------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------

    async def start(self) -> None:
        self.logger.info("Starting Anima...")

        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)

        self.extractor = FastExtractor(
            lexicon_path=self.config.nrc_lexicon_path,
            model_name=self.config.sentence_model_name,
            debug=self.config.debug,
        )
        self.amplifier = CinematicAmplifier()
        self.blend = Blend(self.config, self.clock)

        # ---- FULL PIPELINE WARMUP (direct, no executor) ----
        self._extract_sentence_states_sync(
            "Warmup sentence to initialize extractor and embedder.",
            utterance_id=None,  # suppress telemetry during warmup
        )

        # Anchor (optional)
        if self.config.enable_anchor:
            if not OLLAMA_AVAILABLE:
                raise RuntimeError(
                    "Anchor enabled but ollama is not installed. "
                    "Install affect-engine[anchor]"
                )
            self.logger.info("Anchor enabled")

            self.anchor = LocalExtractor(model=self.config.anchor_model_name)
            self._anchor_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="affect-anchor",
            )
        else:
            self.logger.info("Anchor disabled")
            self.anchor = None
            self._anchor_executor = None

        if self.config.warm_baseline_on_start:
            await self._warm_baseline()

        self._running = True
        self._loop_task = asyncio.create_task(self._loop())

        self.logger.info("Anima started")

    async def stop(self) -> None:
        self.logger.info("Stopping Anima...")

        self._running = False

        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass

        if self._baseline_task and not self._baseline_task.done():
            self._baseline_task.cancel()
            try:
                await self._baseline_task
            except asyncio.CancelledError:
                pass

        if self._anchor_executor:
            self._anchor_executor.shutdown(wait=False)

        self.logger.info("Anima stopped")

    def subscribe(self, callback: Callable[[List[float]], None]) -> None:
        self._subscribers.append(callback)

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------

    async def process_text(
        self,
        text: str,
        influence: Optional[float] = None,
    ) -> dict:
        """
        Process text and apply a single burst driven by the most
        emotionally dominant sentence.
        """

        self.logger.debug(f"[ANIMA] process_text called: '{text[:80]}'")

        if not self.blend or not self.extractor or not self.amplifier:
            raise RuntimeError("Anima not started. Call `await engine.start()` first.")

        t0 = time.perf_counter()
        monotonic_s = self._monotonic_elapsed()

        text = (text or "").strip()
        if not text:
            return {"anima": self.blend.current.tolist()}

        self._transcript_count += 1
        self._context_buffer.extend(text.split())

        # Generate a short utterance ID to group sentence rows with their final row
        utterance_id = uuid.uuid4().hex[:8]

        sentence_states, sentences = self._extract_sentence_states_sync(
            text, utterance_id=utterance_id
        )

        if not sentence_states:
            return {
                "anima": self.blend.current.tolist(),
                "sentences_processed": 0,
                "transcript_count": self._transcript_count,
            }

        def _dominance_score(vibe: np.ndarray) -> float:
            v, a, d = float(vibe[0]), float(vibe[1]), float(vibe[2])
            magnitude = abs(v - 0.5) * 2.0 + abs(a - 0.5) + abs(d - 0.5)
            negative_bias = max(0.0, 0.5 - v) * 3.5
            return magnitude + negative_bias

        dominant = max(sentence_states, key=_dominance_score)
        mean_vibe = np.mean(sentence_states, axis=0)
        final_burst = np.clip(0.85 * dominant + 0.15 * mean_vibe, 0.0, 1.0)

        processing_dt = time.perf_counter() - t0

        self._log_full_text_summary(
            text=text,
            dominant=dominant,
            mean_vibe=mean_vibe,
            final_burst=final_burst,
            total_dt=processing_dt,
        )

        # --- Emit utterance_final telemetry row ---
        if self._telemetry is not None:
            baseline = self.blend.baseline.tolist()
            try:
                rec = make_utterance_final_record(
                    session_id=self._telemetry.session_id,
                    utterance_id=utterance_id,
                    text=text,
                    baseline=baseline,
                    dominant=dominant.tolist(),
                    mean=mean_vibe.tolist(),
                    final_burst=final_burst.tolist(),
                    processing_dt_s=processing_dt,
                    monotonic_s=monotonic_s,
                )
                self._telemetry.push(rec)
            except Exception:
                self.logger.debug("Anima telemetry push failed", exc_info=True)

        self.blend.apply_burst(final_burst, influence=influence)

        if self.anchor:
            self._baseline_event.set()

        return {
            "anima": self.blend.current.tolist(),
            "sentences_processed": len(sentences),
            "transcript_count": self._transcript_count,
        }

    # ---------------------------------------------------------
    # Sync extraction (runs directly now)
    # ---------------------------------------------------------

    def _extract_sentence_states_sync(
        self,
        text: str,
        utterance_id: Optional[str] = None,
    ):
        sentences = nltk.sent_tokenize(text)
        sentence_states: List[np.ndarray] = []
        
        for sentence in sentences:            
            natural = np.array(self.extractor.extract(sentence))
            post_passion = self.amplifier.amplify_passion(natural, self.config.passion)
            post_drama = self.amplifier.snap_drama(post_passion, self.config.drama)

            sentence_states.append(post_drama)

            if self.config.debug:
                self._log_telemetry(sentence, natural, post_passion, post_drama)

            # --- Emit sentence telemetry row ---
            if self._telemetry is not None and utterance_id is not None and self.blend is not None:
                baseline = self.blend.baseline.tolist()
                try:
                    rec = make_sentence_record(
                        session_id=self._telemetry.session_id,
                        utterance_id=utterance_id,
                        text=sentence,
                        natural=natural.tolist(),
                        passion=post_passion.tolist(),
                        drama=post_drama.tolist(),
                        baseline=baseline,
                        monotonic_s=self._monotonic_elapsed(),
                    )
                    self._telemetry.push(rec)
                except Exception:
                    self.logger.debug("Anima sentence telemetry push failed", exc_info=True)

        return sentence_states, sentences

    # ---------------------------------------------------------
    # Internal loop
    # ---------------------------------------------------------

    async def _loop(self) -> None:
        interval = 1.0 / self.config.tick_rate_hz

        while self._running:
            self.clock.tick()
            anima = self.blend.tick()

            for sub in self._subscribers:
                try:
                    result = sub(anima)
                    if inspect.isawaitable(result):
                        asyncio.create_task(result)
                except Exception:
                    self.logger.exception("Subscriber error")

            if self.anchor and self._baseline_event.is_set() and self._has_settled():
                self._ensure_baseline_task()

            await asyncio.sleep(interval)

    # ---------------------------------------------------------
    # Baseline (race-free, coalesced)
    # ---------------------------------------------------------

    def _ensure_baseline_task(self) -> None:
        if self._baseline_task and not self._baseline_task.done():
            return
        self._baseline_task = asyncio.create_task(self._baseline_worker())

    async def _baseline_worker(self) -> None:
        async with self._baseline_lock:
            if not self._baseline_event.is_set():
                return

            self._baseline_event.clear()

            context_text = " ".join(
                list(self._context_buffer)[-self.config.baseline_window_words :]
            )
            if not context_text.strip():
                return

            if not self.anchor or not self._anchor_executor:
                return

            loop = asyncio.get_running_loop()
            baseline = await loop.run_in_executor(
                self._anchor_executor,
                self.anchor.extract_baseline,
                context_text,
            )

            if baseline:
                self.blend.apply_baseline(baseline)
                self.logger.info("New baseline applied: %s", baseline)

    async def _warm_baseline(self) -> None:
        if not self.anchor:
            return

        self._baseline_event.set()
        self._ensure_baseline_task()

        if self._baseline_task:
            await self._baseline_task

    def _has_settled(self, threshold: float = 0.02) -> bool:
        diff = np.abs(self.blend.current - self.blend.baseline)
        return np.all(diff < threshold)

    # ---------------------------------------------------------
    # Debug telemetry
    # ---------------------------------------------------------

    def _log_telemetry(
        self,
        sentence: str,
        natural: np.ndarray,
        post_passion: np.ndarray,
        post_drama: np.ndarray,
        dominant: np.ndarray = None,
        mean_vibe: np.ndarray = None,
        final_burst: np.ndarray = None,
        total_dt: float = None,
    ) -> None:
        if not self.logger.isEnabledFor(logging.DEBUG):
            return

        self.logger.debug("--------------------------------------------------")
        self.logger.debug("SENTENCE: %s", sentence)
        self.logger.debug("NATURAL: %s", natural.tolist())
        self.logger.debug("PASSION: %s", post_passion.tolist())
        self.logger.debug("DRAMA: %s", post_drama.tolist())
        self.logger.debug("BASELINE: %s", self.blend.baseline.tolist())

        if dominant is not None:
            self.logger.debug("DOMINANT: %s", dominant.tolist())
        if mean_vibe is not None:
            self.logger.debug("MEAN: %s", mean_vibe.tolist())
        if final_burst is not None:
            self.logger.debug("FINAL_BURST: %s", final_burst.tolist())
        if total_dt is not None:
            self.logger.debug("TOTAL_DT: %.3fs", total_dt)

        self.logger.debug("--------------------------------------------------")
        
    def _log_full_text_summary(
        self,
        text: str,
        dominant: np.ndarray,
        mean_vibe: np.ndarray,
        final_burst: np.ndarray,
        total_dt: float,
    ) -> None:
        if not self.logger.isEnabledFor(logging.DEBUG):
            return

        self.logger.debug("==================================================")
        self.logger.debug("FULL_TEXT: %s", text)
        self.logger.debug("BASELINE: %s", self.blend.baseline.tolist())
        self.logger.debug("DOMINANT: %s", dominant.tolist())
        self.logger.debug("MEAN: %s", mean_vibe.tolist())
        self.logger.debug("FINAL_BURST: %s", final_burst.tolist())
        self.logger.debug("TOTAL_DT: %.3fs", total_dt)
        self.logger.debug("==================================================")