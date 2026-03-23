"""Fatigue-aware gesture smoothing based on hand landmark jitter."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import time

import numpy as np

from config import (
    FATIGUE_ALPHA,
    FATIGUE_ALERT_COOLDOWN_SECONDS,
    FATIGUE_DURATION_SECONDS,
    FATIGUE_THRESHOLD,
    FATIGUE_WINDOW_FRAMES,
    FATIGUE_SCORE_SCALE,
    SMOOTHING_ALPHA,
)


EMPTY_JITTER = 0.0


@dataclass(slots=True)
class FatigueStatus:
    """Current fatigue state for the active hand."""

    jitter: float
    fatigue_level: float
    fatigued: bool
    smoothing_alpha: float
    should_alert: bool


class FatigueDetector:
    """Track fingertip jitter and raise adaptive smoothing when fatigue is sustained."""

    def __init__(self) -> None:
        """Initialize rolling jitter history and fatigue timers."""

        self._index_tip_history: deque[tuple[float, float, float]] = deque(maxlen=FATIGUE_WINDOW_FRAMES)
        self._high_jitter_started_at: float | None = None
        self._fatigued = False
        self._last_alert_time = 0.0

    def update(self, landmarks_norm: list[tuple[float, float, float]] | None) -> FatigueStatus:
        """Update the detector from one frame of normalized hand landmarks."""

        if not landmarks_norm:
            self._index_tip_history.clear()
            self._high_jitter_started_at = None
            self._fatigued = False
            return FatigueStatus(
                jitter=EMPTY_JITTER,
                fatigue_level=EMPTY_JITTER,
                fatigued=False,
                smoothing_alpha=SMOOTHING_ALPHA,
                should_alert=False,
            )

        index_tip = landmarks_norm[8]
        self._index_tip_history.append(index_tip)
        jitter = self._calculate_jitter()
        fatigue_level = self._normalize_fatigue_level(jitter)
        now = time.perf_counter()

        if jitter >= FATIGUE_THRESHOLD:
            if self._high_jitter_started_at is None:
                self._high_jitter_started_at = now
            sustained_for = now - self._high_jitter_started_at
            is_fatigued = sustained_for >= FATIGUE_DURATION_SECONDS
        else:
            self._high_jitter_started_at = None
            is_fatigued = False

        should_alert = False
        if is_fatigued and not self._fatigued and (now - self._last_alert_time) >= FATIGUE_ALERT_COOLDOWN_SECONDS:
            should_alert = True
            self._last_alert_time = now

        self._fatigued = is_fatigued
        return FatigueStatus(
            jitter=jitter,
            fatigue_level=fatigue_level,
            fatigued=is_fatigued,
            smoothing_alpha=FATIGUE_ALPHA if is_fatigued else SMOOTHING_ALPHA,
            should_alert=should_alert,
        )

    def _calculate_jitter(self) -> float:
        """Return the aggregate positional variance over the rolling window."""

        if len(self._index_tip_history) < FATIGUE_WINDOW_FRAMES:
            return EMPTY_JITTER

        points = np.asarray(self._index_tip_history, dtype=np.float32)
        variance_vector = np.var(points, axis=0)
        return float(np.linalg.norm(variance_vector))

    @staticmethod
    def _normalize_fatigue_level(jitter: float) -> float:
        """Map raw jitter to a stable 0-1 fatigue indicator for UI display."""

        return min(1.0, max(0.0, jitter * FATIGUE_SCORE_SCALE))
