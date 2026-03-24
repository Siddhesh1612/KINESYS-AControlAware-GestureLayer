"""Motion-based fatigue detection via fingertip jitter analysis."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import time

import numpy as np

from config import (
    FATIGUE_ALERT_COOLDOWN_SECONDS, FATIGUE_ALPHA, FATIGUE_DURATION_SECONDS,
    FATIGUE_SCORE_SCALE, FATIGUE_THRESHOLD, FATIGUE_WINDOW_FRAMES, SMOOTHING_ALPHA,
)


@dataclass(slots=True)
class FatigueStatus:
    jitter: float
    fatigue_level: float
    fatigued: bool
    smoothing_alpha: float
    should_alert: bool


class FatigueDetector:
    """Raise adaptive smoothing when sustained hand jitter indicates fatigue."""

    def __init__(self) -> None:
        self._history: deque[tuple[float, float, float]] = deque(maxlen=FATIGUE_WINDOW_FRAMES)
        self._high_jitter_start: float | None = None
        self._fatigued = False
        self._last_alert = 0.0

    def update(self, landmarks_norm: list | None) -> FatigueStatus:
        if not landmarks_norm:
            self._history.clear()
            self._high_jitter_start = None
            self._fatigued = False
            return FatigueStatus(0.0, 0.0, False, SMOOTHING_ALPHA, False)

        self._history.append(landmarks_norm[8])  # index tip
        jitter = self._jitter()
        fatigue_level = min(1.0, max(0.0, jitter * FATIGUE_SCORE_SCALE))
        now = time.perf_counter()

        if jitter >= FATIGUE_THRESHOLD:
            if self._high_jitter_start is None:
                self._high_jitter_start = now
            is_fatigued = (now - self._high_jitter_start) >= FATIGUE_DURATION_SECONDS
        else:
            self._high_jitter_start = None
            is_fatigued = False

        should_alert = (is_fatigued and not self._fatigued
                        and (now - self._last_alert) >= FATIGUE_ALERT_COOLDOWN_SECONDS)
        if should_alert:
            self._last_alert = now
        self._fatigued = is_fatigued

        return FatigueStatus(
            jitter=jitter, fatigue_level=fatigue_level, fatigued=is_fatigued,
            smoothing_alpha=FATIGUE_ALPHA if is_fatigued else SMOOTHING_ALPHA,
            should_alert=should_alert,
        )

    def _jitter(self) -> float:
        if len(self._history) < FATIGUE_WINDOW_FRAMES:
            return 0.0
        pts = np.asarray(self._history, dtype=np.float32)
        return float(np.linalg.norm(np.var(pts, axis=0)))
