"""Gesture analytics logging and session statistics for KINESYS."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import time

from config import ANALYTICS_LOG


LOGGER = logging.getLogger(__name__)

EMPTY_CONFIDENCE = 0.0


@dataclass(slots=True)
class GestureEvent:
    """One logged gesture event for analytics and dashboard visualizations."""

    timestamp_utc: str
    app_name: str
    profile_name: str
    gesture_name: str
    confidence: float
    state_name: str
    modifier_active: str | None


class AnalyticsTracker:
    """Persist gesture events and expose lightweight session statistics."""

    def __init__(self, log_path: str = ANALYTICS_LOG) -> None:
        """Initialize the tracker and load any existing event log."""

        self._log_path = Path(log_path)
        self._session_started_at = time.perf_counter()
        self._events: list[GestureEvent] = []
        self._load_events()

    def record_gesture(
        self,
        gesture_name: str,
        app_name: str,
        profile_name: str,
        confidence: float,
        state_name: str,
        modifier_active: str | None,
    ) -> None:
        """Append one gesture event to the in-memory and persisted analytics log."""

        event = GestureEvent(
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            app_name=app_name,
            profile_name=profile_name,
            gesture_name=gesture_name,
            confidence=confidence,
            state_name=state_name,
            modifier_active=modifier_active,
        )
        self._events.append(event)
        self._save_events()

    def get_counts_by_app(self) -> dict[str, dict[str, int]]:
        """Return per-app gesture counts for the current log."""

        counts_by_app: dict[str, Counter[str]] = {}
        for event in self._events:
            app_counter = counts_by_app.setdefault(event.app_name, Counter())
            app_counter[event.gesture_name] += 1
        return {app_name: dict(counter) for app_name, counter in counts_by_app.items()}

    def get_confidence_history(self) -> list[dict[str, float | str]]:
        """Return timestamped confidence points for later line charts."""

        return [
            {"timestamp_utc": event.timestamp_utc, "confidence": event.confidence}
            for event in self._events
        ]

    def get_session_stats(self) -> dict[str, float | int | str | None]:
        """Return a compact summary of session activity."""

        total_events = len(self._events)
        most_common_gesture = None
        if self._events:
            most_common_gesture = Counter(event.gesture_name for event in self._events).most_common(1)[0][0]

        return {
            "session_duration_seconds": time.perf_counter() - self._session_started_at,
            "total_gestures": total_events,
            "most_used_gesture": most_common_gesture,
            "average_confidence": (
                sum(event.confidence for event in self._events) / float(total_events)
                if total_events
                else EMPTY_CONFIDENCE
            ),
        }

    def _load_events(self) -> None:
        """Load persisted events if the analytics log already exists."""

        if not self._log_path.exists():
            return

        try:
            with self._log_path.open("r", encoding="utf-8") as log_file:
                raw_events = json.load(log_file)
        except Exception as exc:
            LOGGER.exception("Failed to load analytics log: %s", exc)
            return

        if not isinstance(raw_events, list):
            return

        loaded_events: list[GestureEvent] = []
        for raw_event in raw_events:
            if not isinstance(raw_event, dict):
                continue
            try:
                loaded_events.append(GestureEvent(**raw_event))
            except Exception:
                continue
        self._events = loaded_events

    def _save_events(self) -> None:
        """Persist the current event list to disk."""

        try:
            with self._log_path.open("w", encoding="utf-8") as log_file:
                json.dump([asdict(event) for event in self._events], log_file, indent=2)
        except Exception as exc:
            LOGGER.exception("Failed to save analytics log: %s", exc)

