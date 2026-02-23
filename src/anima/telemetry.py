"""Anima processing telemetry writer.

Captures per-sentence and per-utterance emotional extraction data to CSV.
Completely independent of the conversation app's telemetry infrastructure.

Three row types in a single file:
  sentence        — one row per sentence processed, natural/passion/drama populated
  utterance_final — one row per process_text() call, dominant/mean/final_burst populated

Joining: utterance_id groups all sentence rows + their final row together.

Session ID: 8 hex chars, generated once at writer construction, stable for
the lifetime of the process.
"""

from __future__ import annotations

import csv
import io
import logging
import threading
import time
import uuid
from dataclasses import dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

HEADERS = [
    "session_id",
    "wall_time_iso",
    "monotonic_s",
    "utterance_id",       # groups sentence rows + their final row
    "row_type",           # "sentence" | "utterance_final"
    "text",               # sentence text (sentence rows) or full utterance (final row)
    # Per-sentence scores (blank on utterance_final)
    "natural_v", "natural_a", "natural_d", "natural_cx", "natural_ch",
    "passion_v", "passion_a", "passion_d", "passion_cx", "passion_ch",
    "drama_v",   "drama_a",   "drama_d",   "drama_cx",   "drama_ch",
    # Always populated
    "baseline_v", "baseline_a", "baseline_d", "baseline_cx", "baseline_ch",
    # Utterance-level aggregates (blank on sentence rows)
    "dominant_v", "dominant_a", "dominant_d", "dominant_cx", "dominant_ch",
    "mean_v",     "mean_a",     "mean_d",     "mean_cx",     "mean_ch",
    "final_burst_v", "final_burst_a", "final_burst_d", "final_burst_cx", "final_burst_ch",
    "processing_dt_s",  # blank on sentence rows, populated on utterance_final
]


@dataclass
class AnimaTelemetryRecord:
    """One row in the anima processing CSV."""
    session_id: str
    wall_time_iso: str
    monotonic_s: float
    utterance_id: str
    row_type: str
    text: str
    # Per-sentence (optional)
    natural_v: Optional[float] = None
    natural_a: Optional[float] = None
    natural_d: Optional[float] = None
    natural_cx: Optional[float] = None
    natural_ch: Optional[float] = None
    passion_v: Optional[float] = None
    passion_a: Optional[float] = None
    passion_d: Optional[float] = None
    passion_cx: Optional[float] = None
    passion_ch: Optional[float] = None
    drama_v: Optional[float] = None
    drama_a: Optional[float] = None
    drama_d: Optional[float] = None
    drama_cx: Optional[float] = None
    drama_ch: Optional[float] = None
    # Always populated
    baseline_v: Optional[float] = None
    baseline_a: Optional[float] = None
    baseline_d: Optional[float] = None
    baseline_cx: Optional[float] = None
    baseline_ch: Optional[float] = None
    # Utterance-level aggregates (optional)
    dominant_v: Optional[float] = None
    dominant_a: Optional[float] = None
    dominant_d: Optional[float] = None
    dominant_cx: Optional[float] = None
    dominant_ch: Optional[float] = None
    mean_v: Optional[float] = None
    mean_a: Optional[float] = None
    mean_d: Optional[float] = None
    mean_cx: Optional[float] = None
    mean_ch: Optional[float] = None
    final_burst_v: Optional[float] = None
    final_burst_a: Optional[float] = None
    final_burst_d: Optional[float] = None
    final_burst_cx: Optional[float] = None
    final_burst_ch: Optional[float] = None
    processing_dt_s: Optional[float] = None


assert len(HEADERS) == len(fields(AnimaTelemetryRecord)), (
    f"HEADERS ({len(HEADERS)}) and AnimaTelemetryRecord fields "
    f"({len(fields(AnimaTelemetryRecord))}) are out of sync!"
)

# ---------------------------------------------------------------------------
# Ring buffer
# ---------------------------------------------------------------------------

class _AnimaTelemetryBuffer:
    """Single-writer, single-reader ring buffer. No allocation on push."""

    def __init__(self, capacity: int = 512):
        self._buf: list[Optional[AnimaTelemetryRecord]] = [None] * capacity
        self._capacity = capacity
        self._write_idx = 0
        self._read_idx = 0
        self._lock = threading.Lock()
        self._dropped = 0

    def push(self, record: AnimaTelemetryRecord) -> None:
        next_write = (self._write_idx + 1) % self._capacity
        if next_write == self._read_idx:
            self._dropped += 1
            self._read_idx = (self._read_idx + 1) % self._capacity
        self._buf[self._write_idx] = record
        self._write_idx = next_write

    def drain(self) -> List[AnimaTelemetryRecord]:
        with self._lock:
            result = []
            while self._read_idx != self._write_idx:
                rec = self._buf[self._read_idx]
                if rec is not None:
                    result.append(rec)
                self._read_idx = (self._read_idx + 1) % self._capacity
            return result

    @property
    def dropped(self) -> int:
        return self._dropped


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

class AnimaTelemetryWriter:
    """Owns the ring buffer, drain thread, and CSV file.

    Designed to be instantiated once and shared across the anima engine.
    Thread-safe push from the async extraction path.
    """

    def __init__(
        self,
        output_path: Path,
        session_id: Optional[str] = None,
        drain_interval_s: float = 1.0,
        buffer_capacity: int = 512,
    ):
        self._path = output_path
        self.session_id: str = session_id or uuid.uuid4().hex[:8]
        self._drain_interval = drain_interval_s
        self._buffer = _AnimaTelemetryBuffer(buffer_capacity)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._file: Optional[io.TextIOWrapper] = None
        self._csv_writer = None
        self._rows_written = 0

    def start(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self._path, "w", newline="", encoding="utf-8")
        self._csv_writer = csv.writer(self._file)
        self._csv_writer.writerow(HEADERS)
        self._file.flush()
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._drain_loop,
            name="anima-telemetry-drain",
            daemon=True,
        )
        self._thread.start()
        logger.info("Anima telemetry writer started → %s (session=%s)", self._path, self.session_id)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        self._flush()
        if self._file is not None:
            self._file.close()
            self._file = None
        logger.info(
            "Anima telemetry writer stopped. rows=%d dropped=%d",
            self._rows_written, self._buffer.dropped,
        )

    def push(self, record: AnimaTelemetryRecord) -> None:
        self._buffer.push(record)

    def _drain_loop(self) -> None:
        while not self._stop_event.is_set():
            time.sleep(self._drain_interval)
            self._flush()

    def _flush(self) -> None:
        records = self._buffer.drain()
        if not records or self._csv_writer is None or self._file is None:
            return
        for rec in records:
            self._csv_writer.writerow(_to_row(rec))
        self._rows_written += len(records)
        self._file.flush()
        if self._buffer.dropped:
            logger.warning("Anima telemetry buffer overrun: %d dropped total", self._buffer.dropped)


# ---------------------------------------------------------------------------
# Row helpers
# ---------------------------------------------------------------------------

def _f(v: Optional[float], precision: int = 5) -> str:
    if v is None:
        return ""
    return f"{v:.{precision}f}"


def _to_row(rec: AnimaTelemetryRecord) -> list:
    return [
        rec.session_id,
        rec.wall_time_iso,
        _f(rec.monotonic_s, 4),
        rec.utterance_id,
        rec.row_type,
        rec.text,
        _f(rec.natural_v), _f(rec.natural_a), _f(rec.natural_d), _f(rec.natural_cx), _f(rec.natural_ch),
        _f(rec.passion_v), _f(rec.passion_a), _f(rec.passion_d), _f(rec.passion_cx), _f(rec.passion_ch),
        _f(rec.drama_v),   _f(rec.drama_a),   _f(rec.drama_d),   _f(rec.drama_cx),   _f(rec.drama_ch),
        _f(rec.baseline_v), _f(rec.baseline_a), _f(rec.baseline_d), _f(rec.baseline_cx), _f(rec.baseline_ch),
        _f(rec.dominant_v), _f(rec.dominant_a), _f(rec.dominant_d), _f(rec.dominant_cx), _f(rec.dominant_ch),
        _f(rec.mean_v),     _f(rec.mean_a),     _f(rec.mean_d),     _f(rec.mean_cx),     _f(rec.mean_ch),
        _f(rec.final_burst_v), _f(rec.final_burst_a), _f(rec.final_burst_d), _f(rec.final_burst_cx), _f(rec.final_burst_ch),
        _f(rec.processing_dt_s, 4),
    ]


# ---------------------------------------------------------------------------
# Record constructors
# ---------------------------------------------------------------------------

def make_sentence_record(
    *,
    session_id: str,
    utterance_id: str,
    text: str,
    natural: list,
    passion: list,
    drama: list,
    baseline: list,
    monotonic_s: float,
) -> AnimaTelemetryRecord:
    """Build a sentence-level row."""
    return AnimaTelemetryRecord(
        session_id=session_id,
        wall_time_iso=datetime.now().isoformat(timespec="milliseconds"),
        monotonic_s=monotonic_s,
        utterance_id=utterance_id,
        row_type="sentence",
        text=text,
        natural_v=natural[0], natural_a=natural[1], natural_d=natural[2],
        natural_cx=natural[3], natural_ch=natural[4],
        passion_v=passion[0], passion_a=passion[1], passion_d=passion[2],
        passion_cx=passion[3], passion_ch=passion[4],
        drama_v=drama[0],   drama_a=drama[1],   drama_d=drama[2],
        drama_cx=drama[3],  drama_ch=drama[4],
        baseline_v=baseline[0], baseline_a=baseline[1], baseline_d=baseline[2],
        baseline_cx=baseline[3], baseline_ch=baseline[4],
    )


def make_utterance_final_record(
    *,
    session_id: str,
    utterance_id: str,
    text: str,
    baseline: list,
    dominant: list,
    mean: list,
    final_burst: list,
    processing_dt_s: float,
    monotonic_s: float,
) -> AnimaTelemetryRecord:
    """Build an utterance-final aggregation row."""
    return AnimaTelemetryRecord(
        session_id=session_id,
        wall_time_iso=datetime.now().isoformat(timespec="milliseconds"),
        monotonic_s=monotonic_s,
        utterance_id=utterance_id,
        row_type="utterance_final",
        text=text,
        baseline_v=baseline[0], baseline_a=baseline[1], baseline_d=baseline[2],
        baseline_cx=baseline[3], baseline_ch=baseline[4],
        dominant_v=dominant[0], dominant_a=dominant[1], dominant_d=dominant[2],
        dominant_cx=dominant[3], dominant_ch=dominant[4],
        mean_v=mean[0],     mean_a=mean[1],     mean_d=mean[2],
        mean_cx=mean[3],    mean_ch=mean[4],
        final_burst_v=final_burst[0], final_burst_a=final_burst[1], final_burst_d=final_burst[2],
        final_burst_cx=final_burst[3], final_burst_ch=final_burst[4],
        processing_dt_s=processing_dt_s,
    )