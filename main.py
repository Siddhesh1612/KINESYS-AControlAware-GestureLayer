"""Foundation runtime for KINESYS with cursor tracking and click control."""

from __future__ import annotations

import argparse
import base64
import logging
import time
from multiprocessing import Manager
from typing import Any

from config import (
    ACTION_CONFIDENCE_THRESHOLD,
    CURSOR_INDICATOR_COLOR,
    EXIT_KEY,
    FRAME_ENCODE_EXTENSION,
    FRAME_FLIP_CODE,
    FRAME_JPEG_QUALITY,
    FRAME_WAIT_KEY_MS,
    FOUNDATION_CAMERA_BACKEND,
    FOUNDATION_CAMERA_INDEX,
    GESTURE_CLOSED_FIST,
    GESTURE_HOLD_FRAMES,
    GESTURE_INDEX_POINT,
    GESTURE_OPEN_PALM,
    GESTURE_PINCH,
    GESTURE_UNKNOWN,
    HAND_RIGHT,
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
    LOG_FORMAT,
    LOG_LEVEL,
    MAIN_WINDOW_NAME,
    SMOOTHING_ALPHA,
    STATE_CURSOR,
    STATE_IDLE,
    STATE_LOCK,
    WEBCAM_FOURCC,
    WEBCAM_HEIGHT,
    WEBCAM_WIDTH,
)
from check_setup import run_checks


LOGGER = logging.getLogger(__name__)

FONT_FACE = "FONT_HERSHEY_SIMPLEX"
TEXT_SHADOW_OFFSET = 2
TEXT_LINE_1 = 1
TEXT_LINE_2 = 2
TEXT_LINE_3 = 3
TEXT_LINE_4 = 4
TEXT_LINE_5 = 5
TEXT_LINE_6 = 6
PANEL_PADDING_TOP = 12
PANEL_PADDING_BOTTOM = 18
PANEL_WIDTH = 420
JPEG_QUALITY_PARAM = "IMWRITE_JPEG_QUALITY"
CLICK_MESSAGE_SECONDS = 0.4
SHARED_STATE_DEFAULT_PROFILE = "default"
SHARED_STATE_DEFAULT_APP = ""
SHARED_STATE_DEFAULT_TEXT = ""
SHARED_STATE_DEFAULT_URL = ""
SHARED_STATE_DEFAULT_CONFIDENCE = 0.0
SHARED_STATE_DEFAULT_FATIGUE = 0.0
SHARED_STATE_DEFAULT_MODIFIER = None
KEY_MASK = 0xFF
LINE_INDEX_OFFSET = 1


class GestureHoldTracker:
    """Emit a gesture only after it has been observed for the required hold duration."""

    def __init__(self, hold_frames: int) -> None:
        """Initialize the hold tracker with the required consecutive frame count."""

        self._hold_frames = hold_frames
        self._candidate = GESTURE_UNKNOWN
        self._consecutive_frames = 0
        self._emitted = False

    def update(self, gesture: str) -> str | None:
        """Update the tracker with the latest gesture and return a newly stabilized gesture."""

        if gesture != self._candidate:
            self._candidate = gesture
            self._consecutive_frames = 1
            self._emitted = False
            return None

        self._consecutive_frames += 1
        if self._consecutive_frames >= self._hold_frames and not self._emitted:
            self._emitted = True
            return gesture

        return None


class KinesysFoundationApp:
    """Own the foundation runtime loop, shared state, and cursor-only state machine."""

    def __init__(self) -> None:
        """Initialize app state that does not require heavy runtime dependencies."""

        self._state = STATE_IDLE
        self._hold_tracker = GestureHoldTracker(hold_frames=GESTURE_HOLD_FRAMES)
        self._last_click_message_time = 0.0
        self._last_frame_time = time.perf_counter()
        self._shared_state_manager = None
        self._shared_state = None

    def run(self) -> int:
        """Start the foundation cursor loop after validating local setup."""

        configure_logging()
        if not run_checks(strict=False, skip_webcam=False):
            return 1

        try:
            import cv2
        except Exception as exc:
            LOGGER.exception("OpenCV import failed after setup checks: %s", exc)
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
                frame_height, frame_width = frame.shape[:2]
                hands = tracker.process(frame)
                primary_hand = self._select_primary_hand(hands)
                current_gesture = (
                    primary_hand.gesture
                    if primary_hand and primary_hand.confidence >= ACTION_CONFIDENCE_THRESHOLD
                    else GESTURE_UNKNOWN
                )
                stabilized_gesture = self._hold_tracker.update(current_gesture)

                if stabilized_gesture:
                    self._handle_state_transition(stabilized_gesture, cursor)

                if self._state == STATE_CURSOR and primary_hand and current_gesture in {
                    GESTURE_INDEX_POINT,
                    GESTURE_PINCH,
                }:
                    cursor.move_cursor(
                        landmark_point=primary_hand.landmarks_px[INDEX_FINGER_TIP_ID],
                        frame_size=(frame_width, frame_height),
                        smoothing_alpha=SMOOTHING_ALPHA,
                    )

                fps = self._calculate_fps()
                annotated_frame = tracker.draw_annotations(frame)
                self._draw_hud(
                    cv2=cv2,
                    frame=annotated_frame,
                    font_face=font_face,
                    gesture=current_gesture,
                    confidence=primary_hand.confidence if primary_hand else SHARED_STATE_DEFAULT_CONFIDENCE,
                    fps=fps,
                )
                self._update_shared_state(
                    cv2=cv2,
                    frame=annotated_frame,
                    confidence=primary_hand.confidence if primary_hand else SHARED_STATE_DEFAULT_CONFIDENCE,
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

    @staticmethod
    def _open_capture(cv2_module: Any) -> Any:
        """Open the default webcam using the preferred backend and configured properties."""

        backend = getattr(cv2_module, FOUNDATION_CAMERA_BACKEND, FOUNDATION_CAMERA_INDEX)
        capture = cv2_module.VideoCapture(FOUNDATION_CAMERA_INDEX, backend)
        capture.set(cv2_module.CAP_PROP_FRAME_WIDTH, WEBCAM_WIDTH)
        capture.set(cv2_module.CAP_PROP_FRAME_HEIGHT, WEBCAM_HEIGHT)
        capture.set(cv2_module.CAP_PROP_FOURCC, cv2_module.VideoWriter_fourcc(*WEBCAM_FOURCC))
        return capture

    @staticmethod
    def _select_primary_hand(hands: list[Any]) -> Any | None:
        """Prefer the right hand when present, otherwise use the first detected hand."""

        if not hands:
            return None

        for hand in hands:
            if hand.handedness == HAND_RIGHT:
                return hand
        return hands[0]

    def _handle_state_transition(self, stabilized_gesture: str, cursor: Any) -> None:
        """Apply the cursor-only state machine for the foundation milestone."""

        if stabilized_gesture == GESTURE_CLOSED_FIST:
            self._state = STATE_LOCK
            cursor.reset()
            return

        if stabilized_gesture == GESTURE_OPEN_PALM:
            self._state = STATE_IDLE
            cursor.reset()
            return

        if stabilized_gesture == GESTURE_INDEX_POINT:
            self._state = STATE_CURSOR
            return

        if stabilized_gesture == GESTURE_PINCH and self._state == STATE_CURSOR:
            if cursor.click():
                self._last_click_message_time = time.perf_counter()

    def _initialize_shared_state(self) -> None:
        """Create the multiprocessing-backed shared state dictionary for future UI integration."""

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

    def _calculate_fps(self) -> float:
        """Compute the current approximate frame rate from successive frame timestamps."""

        now = time.perf_counter()
        delta = now - self._last_frame_time
        self._last_frame_time = now
        if delta <= 0.0:
            return 0.0
        return 1.0 / delta

    def _draw_hud(
        self,
        cv2: Any,
        frame: Any,
        font_face: int,
        gesture: str,
        confidence: float,
        fps: float,
    ) -> None:
        """Draw a compact diagnostic HUD for the foundation milestone."""

        panel_height = PANEL_PADDING_TOP + PANEL_PADDING_BOTTOM + (HUD_LINE_HEIGHT * TEXT_LINE_6)
        panel = frame.copy()
        cv2.rectangle(
            panel,
            (0, 0),
            (PANEL_WIDTH, panel_height),
            HUD_PANEL_COLOR,
            cv2.FILLED,
        )
        cv2.addWeighted(panel, HUD_PANEL_ALPHA, frame, 1.0 - HUD_PANEL_ALPHA, 0.0, frame)

        self._put_hud_text(
            cv2=cv2,
            frame=frame,
            font_face=font_face,
            text=f"State: {self._state}",
            line_index=TEXT_LINE_1,
            color=HUD_STATE_COLOR if self._state != STATE_LOCK else HUD_WARNING_COLOR,
        )
        self._put_hud_text(
            cv2=cv2,
            frame=frame,
            font_face=font_face,
            text=f"Gesture: {gesture}",
            line_index=TEXT_LINE_2,
            color=HUD_TEXT_COLOR,
        )
        self._put_hud_text(
            cv2=cv2,
            frame=frame,
            font_face=font_face,
            text=f"Confidence: {confidence:.2f}",
            line_index=TEXT_LINE_3,
            color=HUD_TEXT_COLOR,
        )
        self._put_hud_text(
            cv2=cv2,
            frame=frame,
            font_face=font_face,
            text=f"FPS: {fps:.1f}",
            line_index=TEXT_LINE_4,
            color=HUD_TEXT_COLOR,
        )
        self._put_hud_text(
            cv2=cv2,
            frame=frame,
            font_face=font_face,
            text=f"Hold frames: {GESTURE_HOLD_FRAMES}",
            line_index=TEXT_LINE_5,
            color=HUD_TEXT_COLOR,
        )

        click_age = time.perf_counter() - self._last_click_message_time
        click_text = "Last click: fired" if click_age <= CLICK_MESSAGE_SECONDS else "Last click: waiting"
        self._put_hud_text(
            cv2=cv2,
            frame=frame,
            font_face=font_face,
            text=click_text,
            line_index=TEXT_LINE_6,
            color=CURSOR_INDICATOR_COLOR,
        )

    def _put_hud_text(
        self,
        cv2: Any,
        frame: Any,
        font_face: int,
        text: str,
        line_index: int,
        color: tuple[int, int, int],
    ) -> None:
        """Render one HUD text line with a lightweight shadow for contrast."""

        text_x = HUD_MARGIN_X
        text_y = HUD_MARGIN_Y + (HUD_LINE_HEIGHT * (line_index - LINE_INDEX_OFFSET))
        cv2.putText(
            frame,
            text,
            (text_x + TEXT_SHADOW_OFFSET, text_y + TEXT_SHADOW_OFFSET),
            font_face,
            HUD_FONT_SCALE,
            HUD_PANEL_COLOR,
            HUD_FONT_THICKNESS,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            text,
            (text_x, text_y),
            font_face,
            HUD_FONT_SCALE,
            color,
            HUD_FONT_THICKNESS,
            cv2.LINE_AA,
        )

    def _update_shared_state(
        self,
        cv2: Any,
        frame: Any,
        confidence: float,
        fps: float,
        jpeg_quality_param: int,
    ) -> None:
        """Refresh the shared dashboard state for the current frame."""

        encoded_ok, encoded_frame = cv2.imencode(
            FRAME_ENCODE_EXTENSION,
            frame,
            [jpeg_quality_param, FRAME_JPEG_QUALITY],
        )
        if self._shared_state is None:
            return

        if encoded_ok:
            self._shared_state["frame_b64"] = base64.b64encode(encoded_frame.tobytes()).decode("utf-8")

        self._shared_state["gesture_state"] = self._state
        self._shared_state["active_app"] = SHARED_STATE_DEFAULT_APP
        self._shared_state["active_profile"] = SHARED_STATE_DEFAULT_PROFILE
        self._shared_state["recognized_text"] = SHARED_STATE_DEFAULT_TEXT
        self._shared_state["confidence"] = confidence
        self._shared_state["fps"] = fps
        self._shared_state["fatigue_level"] = SHARED_STATE_DEFAULT_FATIGUE
        self._shared_state["last_chars"] = []
        self._shared_state["modifier_active"] = SHARED_STATE_DEFAULT_MODIFIER
        self._shared_state["ngrok_url"] = SHARED_STATE_DEFAULT_URL


def configure_logging() -> None:
    """Configure application logging for the foundation runtime."""

    logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)


def parse_args() -> argparse.Namespace:
    """Parse the command-line interface for the foundation build."""

    parser = argparse.ArgumentParser(description="Run the KINESYS foundation build.")
    return parser.parse_args()


def main() -> int:
    """Run the KINESYS foundation application."""

    parse_args()
    application = KinesysFoundationApp()
    return application.run()


if __name__ == "__main__":
    raise SystemExit(main())
