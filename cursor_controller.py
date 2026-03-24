"""OS cursor movement and click helpers for the KINESYS foundation build."""

from __future__ import annotations

from collections import deque
import ctypes
import logging
import math
import time

import pyautogui

from config import (
    CLICK_DEBOUNCE_MS,
    CURSOR_ALPHA_MAX,
    CURSOR_ALPHA_MIN,
    CURSOR_DEADZONE_PIXELS,
    CURSOR_FAST_MOVEMENT_PIXELS,
    CURSOR_HISTORY_LENGTH,
    CURSOR_MAX_STEP_PIXELS,
    CURSOR_PREDICTION_FACTOR,
    CURSOR_SLOW_MOVEMENT_PIXELS,
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

# Win32 mouse_event flags
_MOUSEEVENTF_LEFTDOWN = 0x0002
_MOUSEEVENTF_LEFTUP = 0x0004
_MOUSEEVENTF_RIGHTDOWN = 0x0008
_MOUSEEVENTF_RIGHTUP = 0x0010

# Try to use Win32 SetCursorPos for zero-latency cursor movement
try:
    _user32 = ctypes.windll.user32
    _WIN32_CURSOR = True
except Exception:
    _user32 = None
    _WIN32_CURSOR = False


def _set_cursor_pos_win32(x: int, y: int) -> None:
    """Move cursor via Win32 SetCursorPos — no pyautogui overhead."""
    _user32.SetCursorPos(x, y)


def _click_win32(x: int, y: int) -> None:
    """Click via Win32 mouse_event — fastest possible click."""
    _user32.SetCursorPos(x, y)
    _user32.mouse_event(_MOUSEEVENTF_LEFTDOWN, x, y, 0, 0)
    _user32.mouse_event(_MOUSEEVENTF_LEFTUP, x, y, 0, 0)


def _right_click_win32(x: int, y: int) -> None:
    """Right-click via Win32 mouse_event."""
    _user32.SetCursorPos(x, y)
    _user32.mouse_event(_MOUSEEVENTF_RIGHTDOWN, x, y, 0, 0)
    _user32.mouse_event(_MOUSEEVENTF_RIGHTUP, x, y, 0, 0)


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
        self._target_history: deque[tuple[float, float]] = deque(maxlen=CURSOR_HISTORY_LENGTH)
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

        # Map with edge margins so full screen is reachable without going to frame edge
        margin_x = frame_width * 0.10
        margin_y = frame_height * 0.10
        raw_x = (landmark_x - margin_x) / max(frame_width - 2 * margin_x, 1)
        raw_y = (landmark_y - margin_y) / max(frame_height - 2 * margin_y, 1)

        target_x = self._clamp_coordinate(raw_x * self._screen_width, self._screen_width)
        target_y = self._clamp_coordinate(raw_y * self._screen_height, self._screen_height)

        self._target_history.append((target_x, target_y))
        filtered_x, filtered_y = self._weighted_average_target()
        predicted_x, predicted_y = self._predict_target(filtered_x, filtered_y)
        predicted_x = self._clamp_coordinate(predicted_x, self._screen_width)
        predicted_y = self._clamp_coordinate(predicted_y, self._screen_height)

        if self._previous_position is None:
            smoothed_x = predicted_x
            smoothed_y = predicted_y
        else:
            previous_x, previous_y = self._previous_position
            movement_distance = math.hypot(predicted_x - previous_x, predicted_y - previous_y)
            if movement_distance <= CURSOR_DEADZONE_PIXELS:
                smoothed_x = previous_x
                smoothed_y = previous_y
            else:
                adaptive_alpha = self._adaptive_alpha(
                    movement_distance=movement_distance,
                    requested_alpha=smoothing_alpha,
                )
                smoothed_x = (adaptive_alpha * predicted_x) + ((1.0 - adaptive_alpha) * previous_x)
                smoothed_y = (adaptive_alpha * predicted_y) + ((1.0 - adaptive_alpha) * previous_y)
                smoothed_x, smoothed_y = self._limit_step(
                    previous_position=(previous_x, previous_y),
                    next_position=(smoothed_x, smoothed_y),
                )

        self._previous_position = (smoothed_x, smoothed_y)
        cursor_x = int(round(smoothed_x))
        cursor_y = int(round(smoothed_y))

        try:
            if _WIN32_CURSOR:
                _set_cursor_pos_win32(cursor_x, cursor_y)
            else:
                pyautogui.moveTo(cursor_x, cursor_y, duration=PYAUTOGUI_MOVE_DURATION)
        except Exception as exc:
            LOGGER.exception("Cursor move failed: %s", exc)

        return cursor_x, cursor_y

    def click(self) -> bool:
        """Perform a debounced left click and report whether it fired."""

        current_time_ms = time.perf_counter() * MILLISECONDS_PER_SECOND
        if current_time_ms - self._last_click_timestamp_ms < CLICK_DEBOUNCE_MS:
            return False

        try:
            if _WIN32_CURSOR and self._previous_position is not None:
                cx = int(round(self._previous_position[0]))
                cy = int(round(self._previous_position[1]))
                _click_win32(cx, cy)
            else:
                pyautogui.click(button=LEFT_MOUSE_BUTTON)
            self._last_click_timestamp_ms = current_time_ms
            return True
        except Exception as exc:
            LOGGER.exception("Mouse click failed: %s", exc)
            return False

    def right_click(self) -> bool:
        """Perform a debounced right click and report whether it fired."""

        current_time_ms = time.perf_counter() * MILLISECONDS_PER_SECOND
        if current_time_ms - self._last_click_timestamp_ms < CLICK_DEBOUNCE_MS:
            return False

        try:
            if _WIN32_CURSOR and self._previous_position is not None:
                cx = int(round(self._previous_position[0]))
                cy = int(round(self._previous_position[1]))
                _right_click_win32(cx, cy)
            else:
                pyautogui.click(button="right")
            self._last_click_timestamp_ms = current_time_ms
            return True
        except Exception as exc:
            LOGGER.exception("Right-click failed: %s", exc)
            return False

    def reset(self) -> None:
        """Clear the smoothing buffer so the next movement starts fresh."""

        self._previous_position = None
        self._target_history.clear()

    @staticmethod
    def _clamp_coordinate(value: float, upper_bound: int) -> float:
        """Clamp a coordinate to the valid screen bounds."""

        return max(0.0, min(value, float(upper_bound - 1)))

    def _weighted_average_target(self) -> tuple[float, float]:
        """Blend recent cursor targets to suppress frame-to-frame fingertip jitter."""

        if not self._target_history:
            return 0.0, 0.0

        weighted_sum_x = 0.0
        weighted_sum_y = 0.0
        total_weight = 0.0
        for index, point in enumerate(self._target_history, start=1):
            weight = float(index)
            weighted_sum_x += point[0] * weight
            weighted_sum_y += point[1] * weight
            total_weight += weight

        return weighted_sum_x / total_weight, weighted_sum_y / total_weight

    def _predict_target(self, filtered_x: float, filtered_y: float) -> tuple[float, float]:
        """Apply a small forward projection so smoothing does not feel sluggish."""

        if len(self._target_history) < 2:
            return filtered_x, filtered_y

        last_x, last_y = self._target_history[-1]
        previous_x, previous_y = self._target_history[-2]
        velocity_x = last_x - previous_x
        velocity_y = last_y - previous_y
        return (
            filtered_x + (velocity_x * CURSOR_PREDICTION_FACTOR),
            filtered_y + (velocity_y * CURSOR_PREDICTION_FACTOR),
        )

    @staticmethod
    def _adaptive_alpha(movement_distance: float, requested_alpha: float) -> float:
        """Raise responsiveness for fast movement while stabilizing slow movement."""

        if movement_distance <= CURSOR_SLOW_MOVEMENT_PIXELS:
            dynamic_alpha = CURSOR_ALPHA_MIN
        elif movement_distance >= CURSOR_FAST_MOVEMENT_PIXELS:
            dynamic_alpha = CURSOR_ALPHA_MAX
        else:
            progress = (
                (movement_distance - CURSOR_SLOW_MOVEMENT_PIXELS)
                / (CURSOR_FAST_MOVEMENT_PIXELS - CURSOR_SLOW_MOVEMENT_PIXELS)
            )
            dynamic_alpha = CURSOR_ALPHA_MIN + ((CURSOR_ALPHA_MAX - CURSOR_ALPHA_MIN) * progress)

        return max(CURSOR_ALPHA_MIN, min(dynamic_alpha, max(requested_alpha, CURSOR_ALPHA_MIN)))

    @staticmethod
    def _limit_step(
        previous_position: tuple[float, float],
        next_position: tuple[float, float],
    ) -> tuple[float, float]:
        """Clamp sudden cursor jumps so reacquisition does not snap across the screen."""

        previous_x, previous_y = previous_position
        next_x, next_y = next_position
        delta_x = next_x - previous_x
        delta_y = next_y - previous_y
        distance = math.hypot(delta_x, delta_y)
        if distance <= CURSOR_MAX_STEP_PIXELS or distance == 0.0:
            return next_x, next_y

        scale = CURSOR_MAX_STEP_PIXELS / distance
        return previous_x + (delta_x * scale), previous_y + (delta_y * scale)
