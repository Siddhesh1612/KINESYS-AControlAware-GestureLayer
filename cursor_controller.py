"""OS cursor movement and click helpers for the KINESYS foundation build."""

from __future__ import annotations

import logging
import time

import pyautogui

from config import (
    CLICK_DEBOUNCE_MS,
    PYAUTOGUI_FAILSAFE,
    PYAUTOGUI_MINIMUM_DURATION,
    PYAUTOGUI_MINIMUM_SLEEP,
    PYAUTOGUI_MOVE_DURATION,
    PYAUTOGUI_PAUSE,
    SMOOTHING_ALPHA,
)


LOGGER = logging.getLogger(__name__)

MILLISECONDS_PER_SECOND = 1000.0
LEFT_MOUSE_BUTTON = "left"


class CursorController:
    """Map webcam coordinates to the desktop cursor with smoothing and click debounce."""

    def __init__(self) -> None:
        """Initialize the controller and configure PyAutoGUI safety settings."""

        pyautogui.FAILSAFE = PYAUTOGUI_FAILSAFE
        pyautogui.PAUSE = PYAUTOGUI_PAUSE
        pyautogui.MINIMUM_DURATION = PYAUTOGUI_MINIMUM_DURATION
        pyautogui.MINIMUM_SLEEP = PYAUTOGUI_MINIMUM_SLEEP

        self._screen_width, self._screen_height = pyautogui.size()
        self._previous_position: tuple[float, float] | None = None
        self._last_click_timestamp_ms = 0.0

    def move_cursor(
        self,
        landmark_point: tuple[int, int],
        frame_size: tuple[int, int],
        smoothing_alpha: float = SMOOTHING_ALPHA,
    ) -> tuple[int, int]:
        """Move the OS cursor using a smoothed webcam-to-screen coordinate mapping."""

        frame_width, frame_height = frame_size
        landmark_x, landmark_y = landmark_point

        target_x = self._clamp_coordinate(
            (landmark_x * self._screen_width) / frame_width,
            self._screen_width,
        )
        target_y = self._clamp_coordinate(
            (landmark_y * self._screen_height) / frame_height,
            self._screen_height,
        )

        if self._previous_position is None:
            smoothed_x = target_x
            smoothed_y = target_y
        else:
            previous_x, previous_y = self._previous_position
            smoothed_x = (smoothing_alpha * target_x) + ((1.0 - smoothing_alpha) * previous_x)
            smoothed_y = (smoothing_alpha * target_y) + ((1.0 - smoothing_alpha) * previous_y)

        self._previous_position = (smoothed_x, smoothed_y)
        cursor_x = int(smoothed_x)
        cursor_y = int(smoothed_y)

        try:
            pyautogui.moveTo(cursor_x, cursor_y, duration=PYAUTOGUI_MOVE_DURATION)
        except Exception as exc:  # pragma: no cover - OS automation dependent
            LOGGER.exception("Cursor move failed: %s", exc)

        return cursor_x, cursor_y

    def click(self) -> bool:
        """Perform a debounced left click and report whether it fired."""

        current_time_ms = time.perf_counter() * MILLISECONDS_PER_SECOND
        if current_time_ms - self._last_click_timestamp_ms < CLICK_DEBOUNCE_MS:
            return False

        try:
            pyautogui.click(button=LEFT_MOUSE_BUTTON)
            self._last_click_timestamp_ms = current_time_ms
            return True
        except Exception as exc:  # pragma: no cover - OS automation dependent
            LOGGER.exception("Mouse click failed: %s", exc)
            return False

    def reset(self) -> None:
        """Clear the smoothing buffer so the next movement starts fresh."""

        self._previous_position = None

    @staticmethod
    def _clamp_coordinate(value: float, upper_bound: int) -> float:
        """Clamp a coordinate to the valid screen bounds."""

        return max(0.0, min(value, float(upper_bound - 1)))
