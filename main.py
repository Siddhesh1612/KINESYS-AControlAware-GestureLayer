"""Main KINESYS gesture-engine runtime for COMMIT 2."""

from __future__ import annotations

import argparse
import base64
import logging
from multiprocessing import Manager
import time
from typing import Any

from check_setup import run_checks
from config import (
    ACTION_CONFIDENCE_THRESHOLD,
    ACTION_RESET_GRACE_SECONDS,
    CLICK_DEBOUNCE_MS,
    CURSOR_INDICATOR_COLOR,
    EXIT_KEY,
    FRAME_ENCODE_EXTENSION,
    FRAME_FLIP_CODE,
    FRAME_JPEG_QUALITY,
    FRAME_WAIT_KEY_MS,
    FOUNDATION_CAMERA_BACKEND,
    FOUNDATION_CAMERA_INDEX,
    GESTURE_CIRCLE,
    GESTURE_CLOSED_FIST,
    GESTURE_FOUR_FINGER_SWIPE,
    GESTURE_HOLD_FRAMES,
    GESTURE_HISTORY_FRAMES,
    GESTURE_INDEX_POINT,
    GESTURE_OPEN_PALM,
    GESTURE_PEACE_SIGN,
    GESTURE_PINCH,
    GESTURE_PINCH_ZOOM_IN,
    GESTURE_PINCH_ZOOM_OUT,
    GESTURE_THREE_FINGER_LEFT,
    GESTURE_THREE_FINGER_RIGHT,
    GESTURE_TWO_FINGER_SWIPE,
    GESTURE_UNKNOWN,
    HUD_FONT_SCALE,
    HUD_FONT_THICKNESS,
    HUD_LINE_HEIGHT,
    HUD_MARGIN_X,
    HUD_MARGIN_Y,
    HUD_PANEL_ALPHA,
    HUD_PANEL_COLOR,
    HUD_STATE_COLOR,
    HUD_TEXT_COLOR,
    HUD_WARNING_COLOR,
    INDEX_FINGER_TIP_ID,
    KEY_ALT,
    KEY_CTRL,
    KEY_LEFT_ARROW,
    KEY_RIGHT_ARROW,
    KEY_SHIFT,
    KEY_TAB,
    LOG_FORMAT,
    LOG_LEVEL,
    MACRO_STATE_DURATION_SECONDS,
    MAIN_WINDOW_NAME,
    MODIFIER_ALT,
    MODIFIER_CTRL,
    MODIFIER_NONE,
    MODIFIER_SHIFT,
    SCROLL_DELTA_DIVISOR,
    SCROLL_MIN_STEP,
    SCROLL_SPEED,
    SMOOTHING_ALPHA,
    STATE_CURSOR,
    STATE_IDLE,
    STATE_LOCK,
    STATE_MACRO,
    STATE_SCROLL,
    STATE_TERMINATED,
    STATE_WRITE,
    TERMINATION_HOLD_FRAMES,
    WEBCAM_FOURCC,
    WEBCAM_HEIGHT,
    WEBCAM_WIDTH,
)


LOGGER = logging.getLogger(__name__)

FONT_FACE = "FONT_HERSHEY_SIMPLEX"
JPEG_QUALITY_PARAM = "IMWRITE_JPEG_QUALITY"
PANEL_PADDING_TOP = 12
PANEL_PADDING_BOTTOM = 18
PANEL_WIDTH = 520
HUD_LINE_COUNT = 7
TEXT_SHADOW_OFFSET = 2
KEY_MASK = 0xFF
LINE_INDEX_OFFSET = 1
TERMINATION_READY_VALUE = True
CLICK_MESSAGE_SECONDS = CLICK_DEBOUNCE_MS / 1000.0

SHARED_STATE_DEFAULT_PROFILE = "default"
SHARED_STATE_DEFAULT_APP = ""
SHARED_STATE_DEFAULT_TEXT = ""
SHARED_STATE_DEFAULT_URL = ""
SHARED_STATE_DEFAULT_CONFIDENCE = 0.0
SHARED_STATE_DEFAULT_FATIGUE = 0.0
SHARED_STATE_DEFAULT_MODIFIER = None

DISPLAY_STATE_COLORS = {
    STATE_IDLE: (136, 135, 128),
    STATE_CURSOR: (15, 110, 86),
    STATE_WRITE: (60, 52, 137),
    STATE_SCROLL: (99, 56, 6),
    STATE_MACRO: (26, 74, 107),
    STATE_LOCK: (113, 43, 19),
    STATE_TERMINATED: (121, 31, 31),
}


class StabilizedValueTracker:
    """Emit a value only after it remains unchanged for a configured number of frames."""

    def __init__(self, hold_frames: int, initial_value: object) -> None:
        """Initialize the tracker for a single stream of candidate values."""

        self._hold_frames = hold_frames
        self._initial_value = initial_value
        self._candidate = initial_value
        self._consecutive_frames = 0
        self._emitted = False

    def update(self, value: object) -> object | None:
        """Advance the tracker and return a newly stabilized value when ready."""

        if value != self._candidate:
            self._candidate = value
            self._consecutive_frames = 1
            self._emitted = False
            return None

        self._consecutive_frames += 1
        if self._consecutive_frames >= self._hold_frames and not self._emitted:
            self._emitted = True
            return value

        return None

    def reset(self) -> None:
        """Clear the tracker so the next value starts a new hold window."""

        self._candidate = self._initial_value
        self._consecutive_frames = 0
        self._emitted = False

    def progress(self, matching_value: object | None = None) -> int:
        """Return the current hold progress, optionally only for one target value."""

        if matching_value is not None and self._candidate != matching_value:
            return 0
        return self._consecutive_frames


class KinesysGestureEngineApp:
    """Run the COMMIT 2 gesture engine with full gesture-state dispatch."""

    def __init__(self) -> None:
        """Initialize state that is independent from external runtime dependencies."""

        self._state = STATE_IDLE
        self._action_hold_tracker = StabilizedValueTracker(
            hold_frames=GESTURE_HOLD_FRAMES,
            initial_value=GESTURE_UNKNOWN,
        )
        self._termination_hold_tracker = StabilizedValueTracker(
            hold_frames=TERMINATION_HOLD_FRAMES,
            initial_value=False,
        )
        self._last_click_message_time = 0.0
        self._last_frame_time = time.perf_counter()
        self._macro_started_at = 0.0
        self._state_entered_at = time.perf_counter()
        self._shared_state_manager = None
        self._shared_state = None

    def run(self) -> int:
        """Start the runtime after validating setup and runtime dependencies."""

        configure_logging()
        if not run_checks(strict=False, skip_webcam=False):
            return 1

        try:
            import cv2
            import pyautogui
        except Exception as exc:
            LOGGER.exception("Runtime dependency import failed after setup checks: %s", exc)
            return 1

        try:
            from cursor_controller import CursorController
            from hand_tracker import HandTracker
        except Exception as exc:
            LOGGER.exception("Runtime module import failed: %s", exc)
            return 1

        capture = None
        tracker = None
        cursor = None

        try:
            self._initialize_shared_state()
            capture = self._open_capture(cv2)
            if not capture.isOpened():
                LOGGER.error("Unable to open the default webcam.")
                return 1

            tracker = HandTracker()
            cursor = CursorController()
            font_face = getattr(cv2, FONT_FACE)
            jpeg_quality_param = getattr(cv2, JPEG_QUALITY_PARAM)

            while True:
                frame_ok, frame = capture.read()
                if not frame_ok:
                    LOGGER.error("Failed to read a frame from the webcam.")
                    break

                frame = cv2.flip(frame, FRAME_FLIP_CODE)
                analysis = tracker.process(frame)

                if self._handle_termination(analysis):
                    self._state = STATE_TERMINATED
                    break

                stabilized_gesture = self._action_hold_tracker.update(analysis.action_gesture)
                if stabilized_gesture is not None:
                    self._handle_stabilized_gesture(
                        stabilized_gesture=stabilized_gesture,
                        analysis=analysis,
                        pyautogui_module=pyautogui,
                        cursor=cursor,
                    )

                self._refresh_runtime_state(analysis)
                self._dispatch_continuous_actions(
                    analysis=analysis,
                    pyautogui_module=pyautogui,
                    cursor=cursor,
                    frame_width=frame.shape[1],
                    frame_height=frame.shape[0],
                )

                fps = self._calculate_fps()
                annotated_frame = tracker.draw_annotations(frame)
                self._draw_hud(
                    cv2_module=cv2,
                    frame=annotated_frame,
                    font_face=font_face,
                    analysis=analysis,
                    fps=fps,
                )
                self._update_shared_state(
                    cv2_module=cv2,
                    frame=annotated_frame,
                    analysis=analysis,
                    fps=fps,
                    jpeg_quality_param=jpeg_quality_param,
                )

                cv2.imshow(MAIN_WINDOW_NAME, annotated_frame)
                key_code = cv2.waitKey(FRAME_WAIT_KEY_MS) & KEY_MASK
                if key_code == ord(EXIT_KEY):
                    break

            return 0
        except Exception as exc:
            LOGGER.exception("Main loop crashed: %s", exc)
            return 1
        finally:
            if capture is not None:
                capture.release()
            if tracker is not None:
                tracker.close()
            if "cv2" in locals():
                cv2.destroyAllWindows()
            if self._shared_state_manager is not None:
                self._shared_state_manager.shutdown()

    def _handle_termination(self, analysis: Any) -> bool:
        """Return whether the two-hand termination gesture has stabilized."""

        stabilized_value = self._termination_hold_tracker.update(analysis.termination_detected)
        return bool(stabilized_value is TERMINATION_READY_VALUE)

    def _handle_stabilized_gesture(
        self,
        stabilized_gesture: str,
        analysis: Any,
        pyautogui_module: Any,
        cursor: Any,
    ) -> None:
        """Apply one-shot actions and state transitions for stabilized gestures."""

        if stabilized_gesture == GESTURE_OPEN_PALM:
            self._set_state(STATE_IDLE)
            cursor.reset()
            return

        if stabilized_gesture == GESTURE_CLOSED_FIST:
            self._set_state(STATE_LOCK)
            cursor.reset()
            return

        if self._state == STATE_LOCK:
            return

        if stabilized_gesture == GESTURE_INDEX_POINT:
            self._set_state(STATE_CURSOR)
            return

        if stabilized_gesture == GESTURE_PINCH:
            self._set_state(STATE_CURSOR)
            self._perform_click(
                pyautogui_module=pyautogui_module,
                cursor=cursor,
                modifier_active=analysis.modifier_active,
            )
            return

        if stabilized_gesture == GESTURE_TWO_FINGER_SWIPE:
            self._set_state(STATE_SCROLL)
            return

        if stabilized_gesture == GESTURE_PEACE_SIGN:
            if analysis.modifier_active == MODIFIER_ALT:
                self._perform_hotkey(pyautogui_module, [KEY_ALT, KEY_TAB])
                self._set_state(STATE_CURSOR)
                return

            self._set_state(STATE_WRITE)
            return

        if stabilized_gesture == GESTURE_CIRCLE:
            self._set_state(STATE_MACRO)
            self._macro_started_at = time.perf_counter()
            return

        if stabilized_gesture == GESTURE_THREE_FINGER_LEFT:
            self._perform_hotkey(pyautogui_module, [KEY_ALT, KEY_LEFT_ARROW])
            return

        if stabilized_gesture == GESTURE_THREE_FINGER_RIGHT:
            self._perform_hotkey(pyautogui_module, [KEY_ALT, KEY_RIGHT_ARROW])
            return

        if stabilized_gesture == GESTURE_FOUR_FINGER_SWIPE:
            self._perform_hotkey(pyautogui_module, [KEY_ALT, KEY_TAB])
            self._set_state(STATE_CURSOR)
            return

        if stabilized_gesture == GESTURE_PINCH_ZOOM_IN:
            self._perform_zoom(pyautogui_module=pyautogui_module, direction=SCROLL_SPEED)
            return

        if stabilized_gesture == GESTURE_PINCH_ZOOM_OUT:
            self._perform_zoom(pyautogui_module=pyautogui_module, direction=-SCROLL_SPEED)

    def _refresh_runtime_state(self, analysis: Any) -> None:
        """Apply non-blocking state updates driven by raw gesture continuity."""

        if self._state == STATE_LOCK or self._state == STATE_TERMINATED:
            return

        if self._state == STATE_MACRO:
            if time.perf_counter() - self._macro_started_at >= MACRO_STATE_DURATION_SECONDS:
                self._set_state(STATE_CURSOR)
            return

        if self._state == STATE_SCROLL and analysis.action_gesture != GESTURE_TWO_FINGER_SWIPE:
            if analysis.action_gesture in {GESTURE_INDEX_POINT, GESTURE_PINCH}:
                self._set_state(STATE_CURSOR)
            elif time.perf_counter() - self._state_entered_at >= ACTION_RESET_GRACE_SECONDS:
                self._set_state(STATE_IDLE)
            return

        if self._state == STATE_CURSOR and analysis.action_gesture == GESTURE_UNKNOWN:
            if time.perf_counter() - self._state_entered_at >= ACTION_RESET_GRACE_SECONDS:
                self._set_state(STATE_IDLE)

    def _dispatch_continuous_actions(
        self,
        analysis: Any,
        pyautogui_module: Any,
        cursor: Any,
        frame_width: int,
        frame_height: int,
    ) -> None:
        """Run continuous cursor and scroll actions for the current state."""

        if self._state == STATE_LOCK or analysis.action_hand is None:
            return

        if self._state == STATE_CURSOR and analysis.action_gesture in {
            GESTURE_INDEX_POINT,
            GESTURE_PINCH,
            GESTURE_PINCH_ZOOM_IN,
            GESTURE_PINCH_ZOOM_OUT,
        }:
            cursor.move_cursor(
                landmark_point=analysis.action_hand.landmarks_px[INDEX_FINGER_TIP_ID],
                frame_size=(frame_width, frame_height),
                smoothing_alpha=SMOOTHING_ALPHA,
            )
            return

        if self._state == STATE_SCROLL and analysis.action_gesture == GESTURE_TWO_FINGER_SWIPE:
            scroll_units = self._calculate_scroll_units(analysis.action_hand.motion_features.palm_dy)
            if scroll_units != 0:
                self._perform_scroll(
                    pyautogui_module=pyautogui_module,
                    scroll_units=scroll_units,
                    modifier_active=analysis.modifier_active,
                )

    def _perform_click(
        self,
        pyautogui_module: Any,
        cursor: Any,
        modifier_active: str | None,
    ) -> None:
        """Execute a click, optionally wrapped in a keyboard modifier."""

        modifier_key = self._modifier_to_key(modifier_active)
        if modifier_key is None:
            if cursor.click():
                self._last_click_message_time = time.perf_counter()
            return

        try:
            pyautogui_module.keyDown(modifier_key)
            if cursor.click():
                self._last_click_message_time = time.perf_counter()
        except Exception as exc:
            LOGGER.exception("Modified click failed: %s", exc)
        finally:
            try:
                pyautogui_module.keyUp(modifier_key)
            except Exception as exc:
                LOGGER.exception("Modifier release failed after click: %s", exc)

    def _perform_scroll(
        self,
        pyautogui_module: Any,
        scroll_units: int,
        modifier_active: str | None,
    ) -> None:
        """Execute a scroll step, optionally wrapped in a modifier key."""

        modifier_key = self._modifier_to_key(modifier_active)
        try:
            if modifier_key is not None:
                pyautogui_module.keyDown(modifier_key)
            pyautogui_module.scroll(scroll_units)
        except Exception as exc:
            LOGGER.exception("Scroll dispatch failed: %s", exc)
        finally:
            if modifier_key is not None:
                try:
                    pyautogui_module.keyUp(modifier_key)
                except Exception as exc:
                    LOGGER.exception("Modifier release failed after scroll: %s", exc)

    def _perform_zoom(self, pyautogui_module: Any, direction: int) -> None:
        """Execute a Ctrl+scroll zoom step."""

        try:
            pyautogui_module.keyDown(KEY_CTRL)
            pyautogui_module.scroll(direction)
        except Exception as exc:
            LOGGER.exception("Zoom dispatch failed: %s", exc)
        finally:
            try:
                pyautogui_module.keyUp(KEY_CTRL)
            except Exception as exc:
                LOGGER.exception("Ctrl release failed after zoom: %s", exc)

    def _perform_hotkey(self, pyautogui_module: Any, keys: list[str]) -> None:
        """Execute a hotkey sequence for one-shot system actions."""

        try:
            pyautogui_module.hotkey(*keys)
        except Exception as exc:
            LOGGER.exception("Hotkey dispatch failed for %s: %s", keys, exc)

    def _calculate_scroll_units(self, palm_dy: float) -> int:
        """Convert vertical swipe motion into a bounded scroll command."""

        average_delta = -palm_dy / float(GESTURE_HISTORY_FRAMES)
        if abs(average_delta) < SCROLL_MIN_STEP:
            return 0
        return int(average_delta / SCROLL_DELTA_DIVISOR)

    @staticmethod
    def _modifier_to_key(modifier_active: str | None) -> str | None:
        """Map the modifier label exposed by the tracker to a PyAutoGUI key name."""

        if modifier_active == MODIFIER_CTRL:
            return KEY_CTRL
        if modifier_active == MODIFIER_SHIFT:
            return KEY_SHIFT
        if modifier_active == MODIFIER_ALT:
            return KEY_ALT
        return None

    @staticmethod
    def _open_capture(cv2_module: Any) -> Any:
        """Open the webcam using the configured backend and resolution."""

        backend = getattr(cv2_module, FOUNDATION_CAMERA_BACKEND, FOUNDATION_CAMERA_INDEX)
        capture = cv2_module.VideoCapture(FOUNDATION_CAMERA_INDEX, backend)
        capture.set(cv2_module.CAP_PROP_FRAME_WIDTH, WEBCAM_WIDTH)
        capture.set(cv2_module.CAP_PROP_FRAME_HEIGHT, WEBCAM_HEIGHT)
        capture.set(cv2_module.CAP_PROP_FOURCC, cv2_module.VideoWriter_fourcc(*WEBCAM_FOURCC))
        return capture

    def _initialize_shared_state(self) -> None:
        """Create the multiprocessing-backed shared dashboard state."""

        self._shared_state_manager = Manager()
        self._shared_state = self._shared_state_manager.dict(
            {
                "frame_b64": SHARED_STATE_DEFAULT_TEXT,
                "gesture_state": STATE_IDLE,
                "active_app": SHARED_STATE_DEFAULT_APP,
                "active_profile": SHARED_STATE_DEFAULT_PROFILE,
                "recognized_text": SHARED_STATE_DEFAULT_TEXT,
                "confidence": SHARED_STATE_DEFAULT_CONFIDENCE,
                "fps": SHARED_STATE_DEFAULT_CONFIDENCE,
                "fatigue_level": SHARED_STATE_DEFAULT_FATIGUE,
                "last_chars": [],
                "modifier_active": SHARED_STATE_DEFAULT_MODIFIER,
                "ngrok_url": SHARED_STATE_DEFAULT_URL,
            }
        )

    def _set_state(self, next_state: str) -> None:
        """Transition the runtime state while recording entry time."""

        self._state = next_state
        self._state_entered_at = time.perf_counter()

    def _calculate_fps(self) -> float:
        """Return the current approximate frame rate."""

        now = time.perf_counter()
        delta = now - self._last_frame_time
        self._last_frame_time = now
        if delta <= 0.0:
            return 0.0
        return 1.0 / delta

    def _draw_hud(
        self,
        cv2_module: Any,
        frame: Any,
        font_face: int,
        analysis: Any,
        fps: float,
    ) -> None:
        """Render the gesture-engine status overlay."""

        panel_height = PANEL_PADDING_TOP + PANEL_PADDING_BOTTOM + (HUD_LINE_HEIGHT * HUD_LINE_COUNT)
        panel = frame.copy()
        cv2_module.rectangle(
            panel,
            (0, 0),
            (PANEL_WIDTH, panel_height),
            HUD_PANEL_COLOR,
            cv2_module.FILLED,
        )
        cv2_module.addWeighted(
            panel,
            HUD_PANEL_ALPHA,
            frame,
            1.0 - HUD_PANEL_ALPHA,
            0.0,
            frame,
        )

        state_color = DISPLAY_STATE_COLORS.get(self._state, HUD_STATE_COLOR)
        self._put_hud_text(
            cv2_module=cv2_module,
            frame=frame,
            font_face=font_face,
            line_index=1,
            text=f"State: {self._state}",
            color=state_color if self._state != STATE_LOCK else HUD_WARNING_COLOR,
        )
        self._put_hud_text(
            cv2_module=cv2_module,
            frame=frame,
            font_face=font_face,
            line_index=2,
            text=f"Action: {analysis.action_gesture} ({analysis.action_confidence:.2f})",
            color=HUD_TEXT_COLOR,
        )
        self._put_hud_text(
            cv2_module=cv2_module,
            frame=frame,
            font_face=font_face,
            line_index=3,
            text=f"Modifier: {analysis.modifier_active or 'none'}",
            color=CURSOR_INDICATOR_COLOR,
        )
        self._put_hud_text(
            cv2_module=cv2_module,
            frame=frame,
            font_face=font_face,
            line_index=4,
            text=(
                "Termination hold: "
                f"{self._termination_hold_tracker.progress(TERMINATION_READY_VALUE)}/"
                f"{TERMINATION_HOLD_FRAMES}"
            ),
            color=HUD_TEXT_COLOR,
        )
        self._put_hud_text(
            cv2_module=cv2_module,
            frame=frame,
            font_face=font_face,
            line_index=5,
            text=f"FPS: {fps:.1f}",
            color=HUD_TEXT_COLOR,
        )
        self._put_hud_text(
            cv2_module=cv2_module,
            frame=frame,
            font_face=font_face,
            line_index=6,
            text=f"Hold frames: {GESTURE_HOLD_FRAMES}",
            color=HUD_TEXT_COLOR,
        )
        click_age = time.perf_counter() - self._last_click_message_time
        click_text = "Last click: fired" if click_age <= CLICK_MESSAGE_SECONDS else "Last click: waiting"
        self._put_hud_text(
            cv2_module=cv2_module,
            frame=frame,
            font_face=font_face,
            line_index=7,
            text=click_text,
            color=CURSOR_INDICATOR_COLOR,
        )

    def _put_hud_text(
        self,
        cv2_module: Any,
        frame: Any,
        font_face: int,
        line_index: int,
        text: str,
        color: tuple[int, int, int],
    ) -> None:
        """Render one HUD line with a shadow for contrast."""

        text_x = HUD_MARGIN_X
        text_y = HUD_MARGIN_Y + (HUD_LINE_HEIGHT * (line_index - LINE_INDEX_OFFSET))
        cv2_module.putText(
            frame,
            text,
            (text_x + TEXT_SHADOW_OFFSET, text_y + TEXT_SHADOW_OFFSET),
            font_face,
            HUD_FONT_SCALE,
            HUD_PANEL_COLOR,
            HUD_FONT_THICKNESS,
            cv2_module.LINE_AA,
        )
        cv2_module.putText(
            frame,
            text,
            (text_x, text_y),
            font_face,
            HUD_FONT_SCALE,
            color,
            HUD_FONT_THICKNESS,
            cv2_module.LINE_AA,
        )

    def _update_shared_state(
        self,
        cv2_module: Any,
        frame: Any,
        analysis: Any,
        fps: float,
        jpeg_quality_param: int,
    ) -> None:
        """Refresh the shared state used by the later dashboard commit."""

        if self._shared_state is None:
            return

        encoded_ok, encoded_frame = cv2_module.imencode(
            FRAME_ENCODE_EXTENSION,
            frame,
            [jpeg_quality_param, FRAME_JPEG_QUALITY],
        )
        if encoded_ok:
            self._shared_state["frame_b64"] = base64.b64encode(encoded_frame.tobytes()).decode("utf-8")

        self._shared_state["gesture_state"] = self._state
        self._shared_state["active_app"] = SHARED_STATE_DEFAULT_APP
        self._shared_state["active_profile"] = SHARED_STATE_DEFAULT_PROFILE
        self._shared_state["recognized_text"] = SHARED_STATE_DEFAULT_TEXT
        self._shared_state["confidence"] = analysis.action_confidence
        self._shared_state["fps"] = fps
        self._shared_state["fatigue_level"] = SHARED_STATE_DEFAULT_FATIGUE
        self._shared_state["last_chars"] = []
        self._shared_state["modifier_active"] = analysis.modifier_active or MODIFIER_NONE
        self._shared_state["ngrok_url"] = SHARED_STATE_DEFAULT_URL


def configure_logging() -> None:
    """Configure logging for the main runtime."""

    logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the runtime entry point."""

    parser = argparse.ArgumentParser(description="Run the KINESYS gesture engine.")
    return parser.parse_args()


def main() -> int:
    """Run the COMMIT 2 KINESYS application."""

    parse_args()
    application = KinesysGestureEngineApp()
    return application.run()


if __name__ == "__main__":
    raise SystemExit(main())
