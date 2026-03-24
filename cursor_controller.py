"""Advanced cursor controller with adaptive smoothing, velocity prediction, and Win32 support."""
from __future__ import annotations

from collections import deque
import ctypes
import math
import time

import pyautogui

from config import (
    CLICK_DEBOUNCE_MS, CURSOR_ALPHA_MAX, CURSOR_ALPHA_MIN,
    CURSOR_DEADZONE_PIXELS, CURSOR_FAST_MOVEMENT_PIXELS,
    CURSOR_HISTORY_LENGTH, CURSOR_MAX_STEP_PIXELS,
    CURSOR_PREDICTION_FACTOR, CURSOR_SLOW_MOVEMENT_PIXELS,
    PYAUTOGUI_FAILSAFE, PYAUTOGUI_MINIMUM_DURATION,
    PYAUTOGUI_MINIMUM_SLEEP, PYAUTOGUI_MOVE_DURATION, PYAUTOGUI_PAUSE,
    SMOOTHING_ALPHA,
)

_MS = 1000.0

try:
    _user32 = ctypes.windll.user32
    _WIN32 = True
except Exception:
    _user32 = None
    _WIN32 = False


class CursorController:
    """Map webcam coordinates to desktop cursor with adaptive smoothing and debounced clicks."""

    def __init__(self) -> None:
        pyautogui.FAILSAFE = PYAUTOGUI_FAILSAFE
        pyautogui.PAUSE = PYAUTOGUI_PAUSE
        pyautogui.MINIMUM_DURATION = PYAUTOGUI_MINIMUM_DURATION
        pyautogui.MINIMUM_SLEEP = PYAUTOGUI_MINIMUM_SLEEP
        self._sw, self._sh = pyautogui.size()
        self._prev: tuple[float, float] | None = None
        self._history: deque[tuple[float, float]] = deque(maxlen=CURSOR_HISTORY_LENGTH)
        self._last_click_ms = 0.0

    def move_cursor(self, landmark_point: tuple[int, int],
                    frame_size: tuple[int, int],
                    smoothing_alpha: float = SMOOTHING_ALPHA) -> tuple[int, int]:
        fw, fh = frame_size
        lx, ly = landmark_point
        mx = fw * 0.10
        my = fh * 0.10
        tx = (lx - mx) / (fw - 2 * mx) * self._sw
        ty = (ly - my) / (fh - 2 * my) * self._sh
        tx = max(0.0, min(float(self._sw - 1), tx))
        ty = max(0.0, min(float(self._sh - 1), ty))

        self._history.append((tx, ty))
        if len(self._history) > 1:
            avg_x = sum(p[0] for p in self._history) / len(self._history)
            avg_y = sum(p[1] for p in self._history) / len(self._history)
        else:
            avg_x, avg_y = tx, ty

        if self._prev is None:
            self._prev = (avg_x, avg_y)

        dx = avg_x - self._prev[0]
        dy = avg_y - self._prev[1]
        dist = math.hypot(dx, dy)

        if dist < CURSOR_DEADZONE_PIXELS:
            return int(self._prev[0]), int(self._prev[1])

        # Adaptive alpha based on movement speed
        t = max(0.0, min(1.0, (dist - CURSOR_SLOW_MOVEMENT_PIXELS) /
                         (CURSOR_FAST_MOVEMENT_PIXELS - CURSOR_SLOW_MOVEMENT_PIXELS)))
        alpha = CURSOR_ALPHA_MIN + t * (CURSOR_ALPHA_MAX - CURSOR_ALPHA_MIN)

        # Velocity prediction
        pred_x = avg_x + dx * CURSOR_PREDICTION_FACTOR
        pred_y = avg_y + dy * CURSOR_PREDICTION_FACTOR

        nx = self._prev[0] + alpha * (pred_x - self._prev[0])
        ny = self._prev[1] + alpha * (pred_y - self._prev[1])

        # Step limit
        step = math.hypot(nx - self._prev[0], ny - self._prev[1])
        if step > CURSOR_MAX_STEP_PIXELS:
            scale = CURSOR_MAX_STEP_PIXELS / step
            nx = self._prev[0] + (nx - self._prev[0]) * scale
            ny = self._prev[1] + (ny - self._prev[1]) * scale

        nx = max(0.0, min(float(self._sw - 1), nx))
        ny = max(0.0, min(float(self._sh - 1), ny))
        self._prev = (nx, ny)

        ix, iy = int(nx), int(ny)
        if _WIN32 and _user32:
            _user32.SetCursorPos(ix, iy)
        else:
            pyautogui.moveTo(ix, iy, duration=PYAUTOGUI_MOVE_DURATION)
        return ix, iy

    def click(self) -> bool:
        now = time.time() * _MS
        if now - self._last_click_ms < CLICK_DEBOUNCE_MS:
            return False
        self._last_click_ms = now
        if _WIN32 and _user32:
            x, y = pyautogui.position()
            _user32.mouse_event(0x0002, x, y, 0, 0)
            _user32.mouse_event(0x0004, x, y, 0, 0)
        else:
            pyautogui.click()
        return True

    def right_click(self) -> bool:
        now = time.time() * _MS
        if now - self._last_click_ms < CLICK_DEBOUNCE_MS:
            return False
        self._last_click_ms = now
        if _WIN32 and _user32:
            x, y = pyautogui.position()
            _user32.mouse_event(0x0008, x, y, 0, 0)
            _user32.mouse_event(0x0010, x, y, 0, 0)
        else:
            pyautogui.rightClick()
        return True

    def scroll(self, dy: float, speed: float = 5.0) -> None:
        pyautogui.scroll(int(dy * speed))
