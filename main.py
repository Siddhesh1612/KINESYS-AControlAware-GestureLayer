"""Main KINESYS adaptive runtime for COMMIT 6."""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
from multiprocessing import Manager
from multiprocessing.managers import BaseManager
import subprocess
import sys
import threading
import time
from typing import Any

from check_setup import run_checks
from config import (
    ACTION_CONFIDENCE_THRESHOLD,
    ACTION_RESET_GRACE_SECONDS,
    CLICK_DEBOUNCE_MS,
    CURSOR_INDICATOR_COLOR,
    DASHBOARD_HEADLESS,
    DASHBOARD_HOST,
    DASHBOARD_PORT,
    DASHBOARD_REFRESH_MS,
    DASHBOARD_SCRIPT,
    DASHBOARD_STATE_AUTHKEY,
    DASHBOARD_STATE_ENV_AUTHKEY,
    DASHBOARD_STATE_ENV_HOST,
    DASHBOARD_STATE_ENV_PORT,
    DASHBOARD_STATE_HOST,
    DASHBOARD_STATE_PORT,
    DASHBOARD_STATE_SNAPSHOT,
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
    GESTURE_TWO_HANDS_X,
    GESTURE_UNKNOWN,
    FATIGUE_ALPHA,
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
    KNN_SAMPLES_REQUIRED,
    LOG_FORMAT,
    LOG_LEVEL,
    MACRO_STATE_DURATION_SECONDS,
    MAIN_WINDOW_NAME,
    MODIFIER_ALT,
    MODIFIER_CTRL,
    MODIFIER_NONE,
    MODIFIER_SHIFT,
    NGROK_AUTH_TOKEN_ENV,
    RECENT_CHARS_LIMIT,
    RECOGNIZED_TEXT_PREVIEW_LENGTH,
    SMOOTHING_ALPHA,
    SCROLL_DELTA_DIVISOR,
    SCROLL_MIN_STEP,
    SCROLL_SPEED,
    STATE_CURSOR,
    STATE_IDLE,
    STATE_LOCK,
    STATE_MACRO,
    STATE_SCROLL,
    STATE_TERMINATED,
    STATE_WRITE,
    TERMINATION_HOLD_FRAMES,
    TRAINER_RECORD_INTERVAL_SECONDS,
    WEBCAM_FOURCC,
    WEBCAM_HEIGHT,
    WEBCAM_WIDTH,
)


LOGGER = logging.getLogger(__name__)

SHARED_STATE_PROXY: Any = None


def get_shared_state_proxy() -> Any:
    """Return the shared dashboard state proxy exposed to the Streamlit process."""

    return SHARED_STATE_PROXY


class DashboardStateBridge(BaseManager):
    """Expose the shared dashboard state to the Streamlit subprocess."""


DashboardStateBridge.register("get_shared_state", callable=get_shared_state_proxy)

FONT_FACE = "FONT_HERSHEY_SIMPLEX"
JPEG_QUALITY_PARAM = "IMWRITE_JPEG_QUALITY"
PANEL_PADDING_TOP = 12
PANEL_PADDING_BOTTOM = 18
PANEL_WIDTH = 520
HUD_LINE_COUNT = 13
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

ACTION_NEW_TAB = "new_tab"
ACTION_RUN_CODE = "run_code"
ACTION_RAISE_HAND = "raise_hand"
ACTION_SWITCH_WINDOW = "switch_window"
ACTION_BROWSER_BACK = "browser_back"
ACTION_BROWSER_FORWARD = "browser_forward"
ACTION_BACK = "back"
ACTION_FORWARD = "forward"
ACTION_ZOOM_IN = "zoom_in"
ACTION_ZOOM_OUT = "zoom_out"
ACTION_BOOKMARK_PAGE = "bookmark_page"
ACTION_SAVE_FILE = "save_file"
ACTION_SCROLL_UP_EDITOR = "scroll_up_editor"
ACTION_SWITCH_TAB = "switch_tab"
ACTION_SWITCH_EDITOR_TAB = "switch_editor_tab"
ACTION_TOGGLE_MUTE = "toggle_mute"
ACTION_TOGGLE_CAMERA = "toggle_camera"
ACTION_SWITCH_PARTICIPANT_VIEW = "switch_participant_view"
ACTION_ALT_TAB = "alt_tab"
ACTION_SCREENSHOT = "screenshot"
ACTION_COMMENT_LINE = "comment_line"
ACTION_UNCOMMENT_LINE = "uncomment_line"

VOICE_FATIGUE = "Please take a break"
VOICE_LOCK = "Kinesys locked"
VOICE_UNLOCK = "Kinesys unlocked"
VOICE_WRITE_MODE = "Write mode"
VOICE_TERMINATED = "Kinesys off"
WRITE_BACKSPACE_TOKEN = "<BACKSPACE>"

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
    """Run the COMMIT 6 gesture engine with dashboard and shared-state services."""

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
        self._current_context = None
        self._context_engine = None
        self._gesture_trainer = None
        self._fatigue_detector = None
        self._analytics_tracker = None
        self._macro_engine = None
        self._air_writer = None
        self._fatigue_level = SHARED_STATE_DEFAULT_FATIGUE
        self._fatigue_jitter = SHARED_STATE_DEFAULT_FATIGUE
        self._active_smoothing_alpha = SMOOTHING_ALPHA
        self._recognized_text = SHARED_STATE_DEFAULT_TEXT
        self._last_chars: list[str] = []
        self._last_write_confidence = SHARED_STATE_DEFAULT_CONFIDENCE
        self._dashboard_process = None
        self._dashboard_state_server = None
        self._dashboard_state_thread = None
        self._ngrok_tunnel = None
        self._ngrok_url = SHARED_STATE_DEFAULT_URL
        self._last_handled_record_request = 0
        self._last_handled_train_request = 0
        self._last_handled_delete_request = 0
        self._last_trainer_sample_time = 0.0

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
            from air_writer import AirWriter
            from analytics import AnalyticsTracker
            from context_engine import ContextEngine
            from cursor_controller import CursorController
            from fatigue_detector import FatigueDetector
            from gesture_trainer import GestureTrainer
            from hand_tracker import HandTracker
            from macro_engine import MacroEngine
            from voice_feedback import VoiceFeedback
        except Exception as exc:
            LOGGER.exception("Runtime module import failed: %s", exc)
            return 1

        capture = None
        air_writer = None
        analytics_tracker = None
        context_engine = None
        tracker = None
        cursor = None
        fatigue_detector = None
        gesture_trainer = None
        macro_engine = None
        voice_feedback = None

        try:
            self._initialize_shared_state()
            self._start_dashboard_state_server()
            self._launch_dashboard_subprocess()
            self._start_ngrok_tunnel()
            capture = self._open_capture(cv2)
            if not capture.isOpened():
                LOGGER.error("Unable to open the default webcam.")
                return 1

            context_engine = ContextEngine()
            self._context_engine = context_engine
            gesture_trainer = GestureTrainer()
            self._gesture_trainer = gesture_trainer
            tracker = HandTracker()
            cursor = CursorController()
            fatigue_detector = FatigueDetector()
            self._fatigue_detector = fatigue_detector
            analytics_tracker = AnalyticsTracker()
            self._analytics_tracker = analytics_tracker
            macro_engine = MacroEngine()
            self._macro_engine = macro_engine
            air_writer = AirWriter()
            self._air_writer = air_writer
            voice_feedback = VoiceFeedback()
            font_face = getattr(cv2, FONT_FACE)
            jpeg_quality_param = getattr(cv2, JPEG_QUALITY_PARAM)

            while True:
                frame_ok, frame = capture.read()
                if not frame_ok:
                    LOGGER.error("Failed to read a frame from the webcam.")
                    break

                frame = cv2.flip(frame, FRAME_FLIP_CODE)
                analysis = tracker.process(frame)
                self._apply_personal_gesture_override(analysis=analysis)
                fatigue_status = self._update_fatigue_status(
                    analysis=analysis,
                    voice_feedback=voice_feedback,
                )
                context_snapshot = context_engine.get_context()
                self._current_context = context_snapshot
                self._handle_app_switch(context_snapshot=context_snapshot, voice_feedback=voice_feedback)
                self._process_dashboard_commands(analysis=analysis, voice_feedback=voice_feedback)

                if self._handle_termination(analysis=analysis, context_snapshot=context_snapshot):
                    voice_feedback.speak(VOICE_TERMINATED)
                    self._set_state(STATE_TERMINATED)
                    break

                stabilized_gesture = self._action_hold_tracker.update(analysis.action_gesture)
                if stabilized_gesture is not None:
                    self._handle_stabilized_gesture(
                        stabilized_gesture=stabilized_gesture,
                        analysis=analysis,
                        context_snapshot=context_snapshot,
                        pyautogui_module=pyautogui,
                        cursor=cursor,
                        voice_feedback=voice_feedback,
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
                    context_snapshot=context_snapshot,
                    fatigue_status=fatigue_status,
                    fps=fps,
                )
                self._update_shared_state(
                    cv2_module=cv2,
                    frame=annotated_frame,
                    analysis=analysis,
                    context_snapshot=context_snapshot,
                    fatigue_status=fatigue_status,
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
            if air_writer is not None:
                air_writer.shutdown()
            if voice_feedback is not None:
                voice_feedback.shutdown()
            self._stop_background_services()
            self._air_writer = None
            self._analytics_tracker = None
            self._context_engine = None
            self._gesture_trainer = None
            self._fatigue_detector = None
            self._macro_engine = None
            if self._shared_state_manager is not None:
                self._shared_state_manager.shutdown()

    def _handle_termination(self, analysis: Any, context_snapshot: Any) -> bool:
        """Return whether the two-hand termination gesture has stabilized."""

        stabilized_value = self._termination_hold_tracker.update(analysis.termination_detected)
        termination_ready = bool(stabilized_value is TERMINATION_READY_VALUE)
        if termination_ready:
            self._record_gesture_event(
                gesture_name=GESTURE_TWO_HANDS_X,
                analysis=analysis,
                context_snapshot=context_snapshot,
            )
        return termination_ready

    def _apply_personal_gesture_override(self, analysis: Any) -> None:
        """Override the default action gesture when the personal KNN model is confident."""

        if self._gesture_trainer is None or analysis.action_hand is None:
            return

        try:
            prediction = self._gesture_trainer.predict_gesture(
                analysis.action_hand.landmarks_norm,
                minimum_confidence=ACTION_CONFIDENCE_THRESHOLD,
            )
        except Exception as exc:
            LOGGER.exception("Personal gesture override failed: %s", exc)
            return

        if prediction.gesture_name is None:
            return

        analysis.action_hand.gesture = prediction.gesture_name
        analysis.action_hand.confidence = prediction.confidence
        analysis.action_gesture = prediction.gesture_name
        analysis.action_confidence = prediction.confidence

    def _update_fatigue_status(self, analysis: Any, voice_feedback: Any) -> Any:
        """Update fatigue state from the active hand and trigger one-shot alerts."""

        if self._fatigue_detector is None:
            self._fatigue_level = SHARED_STATE_DEFAULT_FATIGUE
            self._fatigue_jitter = SHARED_STATE_DEFAULT_FATIGUE
            self._active_smoothing_alpha = SMOOTHING_ALPHA
            return None

        landmarks_norm = analysis.action_hand.landmarks_norm if analysis.action_hand is not None else None
        fatigue_status = self._fatigue_detector.update(landmarks_norm)
        self._fatigue_level = fatigue_status.fatigue_level
        self._fatigue_jitter = fatigue_status.jitter
        self._active_smoothing_alpha = fatigue_status.smoothing_alpha

        if fatigue_status.should_alert:
            voice_feedback.speak(VOICE_FATIGUE)

        return fatigue_status

    def _start_dashboard_state_server(self) -> None:
        """Expose the multiprocessing shared state to the Streamlit subprocess."""

        if self._shared_state is None:
            return

        global SHARED_STATE_PROXY
        SHARED_STATE_PROXY = self._shared_state

        try:
            bridge = DashboardStateBridge(
                address=(DASHBOARD_STATE_HOST, DASHBOARD_STATE_PORT),
                authkey=DASHBOARD_STATE_AUTHKEY.encode("utf-8"),
            )
            self._dashboard_state_server = bridge.get_server()
            self._dashboard_state_thread = threading.Thread(
                target=self._dashboard_state_server.serve_forever,
                name="KinesysDashboardStateServer",
                daemon=True,
            )
            self._dashboard_state_thread.start()
        except Exception as exc:
            LOGGER.exception("Failed to start dashboard shared-state server: %s", exc)
            self._dashboard_state_server = None
            self._dashboard_state_thread = None

    def _launch_dashboard_subprocess(self) -> None:
        """Launch the Streamlit dashboard as a non-blocking subprocess."""

        if not os.path.exists(DASHBOARD_SCRIPT):
            LOGGER.warning("Dashboard script not found at %s", DASHBOARD_SCRIPT)
            return

        dashboard_environment = os.environ.copy()
        dashboard_environment[DASHBOARD_STATE_ENV_HOST] = DASHBOARD_STATE_HOST
        dashboard_environment[DASHBOARD_STATE_ENV_PORT] = str(DASHBOARD_STATE_PORT)
        dashboard_environment[DASHBOARD_STATE_ENV_AUTHKEY] = DASHBOARD_STATE_AUTHKEY

        command = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            DASHBOARD_SCRIPT,
            "--server.headless",
            DASHBOARD_HEADLESS,
            "--server.address",
            DASHBOARD_HOST,
            "--server.port",
            str(DASHBOARD_PORT),
        ]

        try:
            self._dashboard_process = subprocess.Popen(
                command,
                cwd=os.path.dirname(DASHBOARD_SCRIPT) or None,
                env=dashboard_environment,
            )
        except Exception as exc:
            LOGGER.exception("Failed to launch Streamlit dashboard: %s", exc)
            self._dashboard_process = None

    def _start_ngrok_tunnel(self) -> None:
        """Start an ngrok tunnel for the dashboard if a token is configured."""

        if self._shared_state is None:
            return

        try:
            from dotenv import load_dotenv
            from pyngrok import ngrok
        except Exception as exc:
            LOGGER.exception("Unable to import ngrok runtime dependencies: %s", exc)
            return

        try:
            load_dotenv()
            auth_token = os.getenv(NGROK_AUTH_TOKEN_ENV, "").strip()
            if not auth_token:
                LOGGER.info("No ngrok token configured; dashboard will remain local-only.")
                return

            ngrok.set_auth_token(auth_token)
            self._ngrok_tunnel = ngrok.connect(addr=DASHBOARD_PORT, bind_tls=True)
            self._ngrok_url = self._ngrok_tunnel.public_url
            self._shared_state["ngrok_url"] = self._ngrok_url
            LOGGER.info("Dashboard ngrok URL: %s", self._ngrok_url)
            print(f"KINESYS dashboard URL: {self._ngrok_url}")
        except Exception as exc:
            LOGGER.exception("Failed to start ngrok tunnel: %s", exc)

    def _stop_background_services(self) -> None:
        """Stop the dashboard subprocess, ngrok tunnel, and shared-state bridge."""

        global SHARED_STATE_PROXY

        if self._ngrok_tunnel is not None:
            try:
                from pyngrok import ngrok

                ngrok.disconnect(self._ngrok_tunnel.public_url)
            except Exception as exc:
                LOGGER.exception("Failed to stop ngrok tunnel: %s", exc)
            self._ngrok_tunnel = None
            self._ngrok_url = SHARED_STATE_DEFAULT_URL

        if self._dashboard_process is not None:
            try:
                if self._dashboard_process.poll() is None:
                    self._dashboard_process.terminate()
                    self._dashboard_process.wait(timeout=5.0)
            except Exception:
                try:
                    self._dashboard_process.kill()
                except Exception as exc:
                    LOGGER.exception("Failed to kill dashboard subprocess: %s", exc)
            self._dashboard_process = None

        if self._dashboard_state_server is not None:
            try:
                self._dashboard_state_server.stop_event.set()
            except Exception as exc:
                LOGGER.exception("Failed to stop dashboard shared-state server: %s", exc)
            self._dashboard_state_server = None

        if self._dashboard_state_thread is not None:
            self._dashboard_state_thread.join(timeout=1.0)
            self._dashboard_state_thread = None

        SHARED_STATE_PROXY = None

    def _write_dashboard_snapshot(self) -> None:
        """Persist a JSON snapshot so the dashboard can fall back to file polling."""

        if self._shared_state is None:
            return

        try:
            with open(DASHBOARD_STATE_SNAPSHOT, "w", encoding="utf-8") as snapshot_file:
                json.dump(dict(self._shared_state), snapshot_file, indent=2)
        except Exception as exc:
            LOGGER.exception("Failed to write dashboard snapshot: %s", exc)

    def _process_dashboard_commands(self, analysis: Any, voice_feedback: Any) -> None:
        """Handle dashboard-triggered trainer commands from the shared state."""

        if self._shared_state is None or self._gesture_trainer is None:
            return

        self._shared_state["trainer_progress"] = self._gesture_trainer.get_training_progress()
        self._shared_state["trained_gestures"] = self._gesture_trainer.list_trained_gestures()
        self._shared_state["two_hand_mode"] = analysis.modifier_hand is not None

        record_request_id = int(self._shared_state.get("trainer_record_request_id", 0))
        if record_request_id != self._last_handled_record_request:
            self._last_handled_record_request = record_request_id
            self._shared_state["trainer_recording_active"] = True
            self._shared_state["trainer_status"] = "Show the selected gesture to the camera."
            self._last_trainer_sample_time = 0.0

        if bool(self._shared_state.get("trainer_recording_active", False)):
            self._capture_trainer_sample(analysis=analysis)

        train_request_id = int(self._shared_state.get("trainer_train_request_id", 0))
        if train_request_id != self._last_handled_train_request:
            self._last_handled_train_request = train_request_id
            self._handle_train_request(voice_feedback=voice_feedback)

        delete_request_id = int(self._shared_state.get("trainer_delete_request_id", 0))
        if delete_request_id != self._last_handled_delete_request:
            self._last_handled_delete_request = delete_request_id
            self._handle_delete_request()

    def _capture_trainer_sample(self, analysis: Any) -> None:
        """Capture one timed trainer sample from the current action hand."""

        if self._shared_state is None or self._gesture_trainer is None:
            return

        target_gesture = str(self._shared_state.get("trainer_target_gesture", "")).strip()
        if not target_gesture:
            self._shared_state["trainer_recording_active"] = False
            self._shared_state["trainer_status"] = "Select a gesture before recording."
            return

        if analysis.action_hand is None:
            return

        now = time.perf_counter()
        if now - self._last_trainer_sample_time < TRAINER_RECORD_INTERVAL_SECONDS:
            return

        try:
            sample_count = self._gesture_trainer.record_sample(
                gesture_name=target_gesture,
                landmarks_norm=analysis.action_hand.landmarks_norm,
            )
        except Exception as exc:
            LOGGER.exception("Failed to record trainer sample: %s", exc)
            self._shared_state["trainer_recording_active"] = False
            self._shared_state["trainer_status"] = f"Recording failed: {exc}"
            return

        self._last_trainer_sample_time = now
        self._shared_state["trainer_progress"] = self._gesture_trainer.get_training_progress()
        self._shared_state["trainer_status"] = f"Collected {sample_count}/{KNN_SAMPLES_REQUIRED} samples."

        if sample_count >= KNN_SAMPLES_REQUIRED:
            self._shared_state["trainer_recording_active"] = False
            self._shared_state["trainer_status"] = "Samples collected. Press Train."

    def _handle_train_request(self, voice_feedback: Any) -> None:
        """Train the selected gesture from the samples already collected."""

        if self._shared_state is None or self._gesture_trainer is None:
            return

        target_gesture = str(self._shared_state.get("trainer_target_gesture", "")).strip()
        if not target_gesture:
            self._shared_state["trainer_status"] = "Select a gesture before training."
            return

        trained_ok = self._gesture_trainer.train_gesture(target_gesture)
        self._shared_state["trainer_progress"] = self._gesture_trainer.get_training_progress()
        self._shared_state["trained_gestures"] = self._gesture_trainer.list_trained_gestures()
        if trained_ok:
            self._shared_state["trainer_status"] = f"Trained {target_gesture}."
            voice_feedback.speak("Custom gesture trained")
            return

        self._shared_state["trainer_status"] = "Need at least 5 samples before training."

    def _handle_delete_request(self) -> None:
        """Delete one selected trained gesture from the persisted trainer state."""

        if self._shared_state is None or self._gesture_trainer is None:
            return

        target_gesture = str(self._shared_state.get("trainer_delete_gesture", "")).strip()
        if not target_gesture:
            self._shared_state["trainer_status"] = "Select a trained gesture to delete."
            return

        deleted_ok = self._gesture_trainer.delete_gesture(target_gesture)
        self._shared_state["trainer_progress"] = self._gesture_trainer.get_training_progress()
        self._shared_state["trained_gestures"] = self._gesture_trainer.list_trained_gestures()
        if deleted_ok:
            self._shared_state["trainer_status"] = f"Deleted {target_gesture}."
            return

        self._shared_state["trainer_status"] = f"No trained gesture named {target_gesture}."

    @staticmethod
    def _handle_app_switch(context_snapshot: Any, voice_feedback: Any) -> None:
        """Announce active app switches without blocking the main loop."""

        if context_snapshot.app_changed:
            voice_feedback.speak(context_snapshot.voice_label)

    def _handle_stabilized_gesture(
        self,
        stabilized_gesture: str,
        analysis: Any,
        context_snapshot: Any,
        pyautogui_module: Any,
        cursor: Any,
        voice_feedback: Any,
    ) -> None:
        """Apply one-shot actions and state transitions for stabilized gestures."""

        if stabilized_gesture == GESTURE_OPEN_PALM:
            if self._state == STATE_LOCK:
                voice_feedback.speak(VOICE_UNLOCK)
            self._set_state(STATE_IDLE)
            cursor.reset()
            self._record_gesture_event(stabilized_gesture, analysis, context_snapshot)
            return

        if stabilized_gesture == GESTURE_CLOSED_FIST:
            self._set_state(STATE_LOCK)
            cursor.reset()
            voice_feedback.speak(VOICE_LOCK)
            self._record_gesture_event(stabilized_gesture, analysis, context_snapshot)
            return

        if self._state == STATE_LOCK:
            return

        if self._state == STATE_WRITE:
            return

        if stabilized_gesture == GESTURE_INDEX_POINT:
            self._set_state(STATE_CURSOR)
            self._record_gesture_event(stabilized_gesture, analysis, context_snapshot)
            return

        if stabilized_gesture == GESTURE_PINCH:
            self._set_state(STATE_CURSOR)
            self._perform_click(
                pyautogui_module=pyautogui_module,
                cursor=cursor,
                modifier_active=analysis.modifier_active,
            )
            self._record_gesture_event(stabilized_gesture, analysis, context_snapshot)
            return

        if stabilized_gesture == GESTURE_TWO_FINGER_SWIPE:
            self._set_state(STATE_SCROLL)
            self._record_gesture_event(stabilized_gesture, analysis, context_snapshot)
            return

        if stabilized_gesture == GESTURE_PEACE_SIGN:
            if analysis.modifier_active == MODIFIER_SHIFT:
                self._set_state(STATE_WRITE)
                voice_feedback.speak(VOICE_WRITE_MODE)
                self._record_gesture_event(stabilized_gesture, analysis, context_snapshot)
                return

            if self._dispatch_context_action(
                gesture_name=stabilized_gesture,
                analysis=analysis,
                context_snapshot=context_snapshot,
                pyautogui_module=pyautogui_module,
            ):
                self._set_state(STATE_CURSOR)
                self._record_gesture_event(stabilized_gesture, analysis, context_snapshot)
                return

            if analysis.modifier_active == MODIFIER_ALT:
                self._perform_hotkey(pyautogui_module, [KEY_ALT, KEY_TAB])
                self._set_state(STATE_CURSOR)
                self._record_gesture_event(stabilized_gesture, analysis, context_snapshot)
                return

            self._set_state(STATE_WRITE)
            voice_feedback.speak(VOICE_WRITE_MODE)
            self._record_gesture_event(stabilized_gesture, analysis, context_snapshot)
            return

        if stabilized_gesture == GESTURE_CIRCLE:
            macro_name = self._play_macro_for_context(
                context_snapshot=context_snapshot,
                pyautogui_module=pyautogui_module,
            )
            if macro_name is not None:
                self._set_state(STATE_MACRO)
                self._macro_started_at = time.perf_counter()
                voice_feedback.speak(f"Macro {macro_name}")
                self._record_gesture_event(stabilized_gesture, analysis, context_snapshot)
                return

            if self._dispatch_context_action(
                gesture_name=stabilized_gesture,
                analysis=analysis,
                context_snapshot=context_snapshot,
                pyautogui_module=pyautogui_module,
            ):
                self._set_state(STATE_CURSOR)
                self._record_gesture_event(stabilized_gesture, analysis, context_snapshot)
                return

            self._set_state(STATE_MACRO)
            self._macro_started_at = time.perf_counter()
            self._record_gesture_event(stabilized_gesture, analysis, context_snapshot)
            return

        if stabilized_gesture == GESTURE_THREE_FINGER_LEFT:
            if self._dispatch_context_action(
                gesture_name=stabilized_gesture,
                analysis=analysis,
                context_snapshot=context_snapshot,
                pyautogui_module=pyautogui_module,
            ):
                self._set_state(STATE_CURSOR)
                self._record_gesture_event(stabilized_gesture, analysis, context_snapshot)
                return
            self._perform_hotkey(pyautogui_module, [KEY_ALT, KEY_LEFT_ARROW])
            self._record_gesture_event(stabilized_gesture, analysis, context_snapshot)
            return

        if stabilized_gesture == GESTURE_THREE_FINGER_RIGHT:
            if self._dispatch_context_action(
                gesture_name=stabilized_gesture,
                analysis=analysis,
                context_snapshot=context_snapshot,
                pyautogui_module=pyautogui_module,
            ):
                self._set_state(STATE_CURSOR)
                self._record_gesture_event(stabilized_gesture, analysis, context_snapshot)
                return
            self._perform_hotkey(pyautogui_module, [KEY_ALT, KEY_RIGHT_ARROW])
            self._record_gesture_event(stabilized_gesture, analysis, context_snapshot)
            return

        if stabilized_gesture == GESTURE_FOUR_FINGER_SWIPE:
            if self._dispatch_context_action(
                gesture_name=stabilized_gesture,
                analysis=analysis,
                context_snapshot=context_snapshot,
                pyautogui_module=pyautogui_module,
            ):
                self._set_state(STATE_CURSOR)
                self._record_gesture_event(stabilized_gesture, analysis, context_snapshot)
                return
            self._perform_hotkey(pyautogui_module, [KEY_ALT, KEY_TAB])
            self._set_state(STATE_CURSOR)
            self._record_gesture_event(stabilized_gesture, analysis, context_snapshot)
            return

        if stabilized_gesture == GESTURE_PINCH_ZOOM_IN:
            if self._dispatch_context_action(
                gesture_name=stabilized_gesture,
                analysis=analysis,
                context_snapshot=context_snapshot,
                pyautogui_module=pyautogui_module,
            ):
                self._set_state(STATE_CURSOR)
                self._record_gesture_event(stabilized_gesture, analysis, context_snapshot)
                return
            self._perform_zoom(pyautogui_module=pyautogui_module, direction=SCROLL_SPEED)
            self._record_gesture_event(stabilized_gesture, analysis, context_snapshot)
            return

        if stabilized_gesture == GESTURE_PINCH_ZOOM_OUT:
            if self._dispatch_context_action(
                gesture_name=stabilized_gesture,
                analysis=analysis,
                context_snapshot=context_snapshot,
                pyautogui_module=pyautogui_module,
            ):
                self._set_state(STATE_CURSOR)
                self._record_gesture_event(stabilized_gesture, analysis, context_snapshot)
                return
            self._perform_zoom(pyautogui_module=pyautogui_module, direction=-SCROLL_SPEED)
            self._record_gesture_event(stabilized_gesture, analysis, context_snapshot)

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

        if self._state == STATE_WRITE and self._air_writer is not None:
            index_point = (
                analysis.action_hand.landmarks_px[INDEX_FINGER_TIP_ID]
                if analysis.action_hand is not None
                else None
            )
            write_update = self._air_writer.update(
                index_point=index_point,
                frame_size=(frame_width, frame_height),
                allow_stroke_drawing=analysis.action_gesture == GESTURE_PEACE_SIGN,
            )
            self._last_write_confidence = write_update.confidence
            if write_update.character is not None:
                self._handle_write_update(
                    write_update=write_update,
                    pyautogui_module=pyautogui_module,
                )

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
                smoothing_alpha=self._active_smoothing_alpha,
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

    def _play_macro_for_context(self, context_snapshot: Any, pyautogui_module: Any) -> str | None:
        """Play the best matching macro for the current app and return its name."""

        if self._macro_engine is None:
            return None

        try:
            return self._macro_engine.play_macro_for_context(
                trigger_gesture=GESTURE_CIRCLE,
                active_app=context_snapshot.active_app,
                profile_name=context_snapshot.profile_name,
                pyautogui_module=pyautogui_module,
            )
        except Exception as exc:
            LOGGER.exception("Macro playback failed: %s", exc)
            return None

    def _handle_write_update(self, write_update: Any, pyautogui_module: Any) -> None:
        """Type one recognized or hovered character and update shared write state."""

        typed_token = write_update.character
        if typed_token is None:
            return

        try:
            if typed_token == WRITE_BACKSPACE_TOKEN:
                pyautogui_module.press("backspace")
                self._recognized_text = self._recognized_text[:-1]
                self._remember_character("BACKSPACE")
                return

            if typed_token == " ":
                pyautogui_module.press("space")
            else:
                pyautogui_module.write(typed_token)

            self._recognized_text += typed_token
            self._remember_character(typed_token)
        except Exception as exc:
            LOGGER.exception("Write-mode typing failed: %s", exc)

    def _remember_character(self, character: str) -> None:
        """Keep a bounded list of the most recent typed characters."""

        self._last_chars.append(character)
        self._last_chars = self._last_chars[-RECENT_CHARS_LIMIT:]

    def _record_gesture_event(
        self,
        gesture_name: str,
        analysis: Any,
        context_snapshot: Any,
    ) -> None:
        """Persist one fired gesture event for the analytics pipeline."""

        if self._analytics_tracker is None:
            return

        try:
            self._analytics_tracker.record_gesture(
                gesture_name=gesture_name,
                app_name=context_snapshot.active_app,
                profile_name=context_snapshot.profile_name,
                confidence=analysis.action_confidence,
                state_name=self._state,
                modifier_active=analysis.modifier_active,
            )
        except Exception as exc:
            LOGGER.exception("Analytics logging failed: %s", exc)

    def _dispatch_context_action(
        self,
        gesture_name: str,
        analysis: Any,
        context_snapshot: Any,
        pyautogui_module: Any,
    ) -> bool:
        """Resolve and execute the current profile action for one stabilized gesture."""

        if analysis.action_hand is None:
            return False

        try:
            if self._context_engine is None:
                return False
            vertical_motion = analysis.action_hand.motion_features.palm_dy
            action_name = self._context_engine.resolve_action(
                gesture_name=gesture_name,
                profile=context_snapshot.profile,
                vertical_motion=vertical_motion,
            )
        except Exception as exc:
            LOGGER.exception("Context action resolution failed: %s", exc)
            return False

        if action_name is None:
            return False

        return self._execute_profile_action(
            action_name=action_name,
            pyautogui_module=pyautogui_module,
        )

    def _execute_profile_action(
        self,
        action_name: str,
        pyautogui_module: Any,
    ) -> bool:
        """Execute one resolved profile action through PyAutoGUI."""

        try:
            if action_name == ACTION_NEW_TAB:
                pyautogui_module.hotkey(KEY_CTRL, "t")
                return True
            if action_name == ACTION_RUN_CODE:
                pyautogui_module.press("f5")
                return True
            if action_name == ACTION_RAISE_HAND:
                pyautogui_module.hotkey(KEY_ALT, "y")
                return True
            if action_name in {ACTION_SWITCH_WINDOW, ACTION_ALT_TAB}:
                pyautogui_module.hotkey(KEY_ALT, KEY_TAB)
                return True
            if action_name in {ACTION_BROWSER_BACK, ACTION_BACK}:
                pyautogui_module.hotkey(KEY_ALT, KEY_LEFT_ARROW)
                return True
            if action_name in {ACTION_BROWSER_FORWARD, ACTION_FORWARD}:
                pyautogui_module.hotkey(KEY_ALT, KEY_RIGHT_ARROW)
                return True
            if action_name == ACTION_ZOOM_IN:
                self._perform_zoom(pyautogui_module=pyautogui_module, direction=SCROLL_SPEED)
                return True
            if action_name == ACTION_ZOOM_OUT:
                self._perform_zoom(pyautogui_module=pyautogui_module, direction=-SCROLL_SPEED)
                return True
            if action_name == ACTION_BOOKMARK_PAGE:
                pyautogui_module.hotkey(KEY_CTRL, "d")
                return True
            if action_name == ACTION_SAVE_FILE:
                pyautogui_module.hotkey(KEY_CTRL, "s")
                return True
            if action_name == ACTION_SCROLL_UP_EDITOR:
                pyautogui_module.scroll(SCROLL_SPEED)
                return True
            if action_name in {ACTION_SWITCH_TAB, ACTION_SWITCH_EDITOR_TAB}:
                pyautogui_module.hotkey(KEY_CTRL, KEY_TAB)
                return True
            if action_name == ACTION_TOGGLE_MUTE:
                pyautogui_module.hotkey(KEY_ALT, "a")
                return True
            if action_name == ACTION_TOGGLE_CAMERA:
                pyautogui_module.hotkey(KEY_ALT, "v")
                return True
            if action_name == ACTION_SWITCH_PARTICIPANT_VIEW:
                pyautogui_module.hotkey(KEY_ALT, "u")
                return True
            if action_name == ACTION_SCREENSHOT:
                pyautogui_module.hotkey("win", KEY_SHIFT, "s")
                return True
            if action_name == ACTION_COMMENT_LINE:
                pyautogui_module.hotkey(KEY_CTRL, "/")
                return True
            if action_name == ACTION_UNCOMMENT_LINE:
                pyautogui_module.hotkey(KEY_CTRL, "/")
                return True
        except Exception as exc:
            LOGGER.exception("Profile action execution failed for %s: %s", action_name, exc)
            return False

        LOGGER.warning("No executor implemented for profile action: %s", action_name)
        return False

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
                "two_hand_mode": False,
                "trainer_target_gesture": SHARED_STATE_DEFAULT_TEXT,
                "trainer_record_request_id": 0,
                "trainer_train_request_id": 0,
                "trainer_delete_request_id": 0,
                "trainer_delete_gesture": SHARED_STATE_DEFAULT_TEXT,
                "trainer_recording_active": False,
                "trainer_status": "Idle",
                "trainer_progress": {},
                "trained_gestures": [],
            }
        )
        self._write_dashboard_snapshot()

    def _set_state(self, next_state: str) -> None:
        """Transition the runtime state while recording entry time."""

        if self._state == next_state:
            self._state_entered_at = time.perf_counter()
            return

        if self._state == STATE_WRITE and next_state != STATE_WRITE and self._air_writer is not None:
            self._air_writer.stop_session()
            self._last_write_confidence = SHARED_STATE_DEFAULT_CONFIDENCE

        if self._state != STATE_WRITE and next_state == STATE_WRITE and self._air_writer is not None:
            self._air_writer.start_session()
            self._last_write_confidence = SHARED_STATE_DEFAULT_CONFIDENCE

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
        context_snapshot: Any,
        fatigue_status: Any,
        fps: float,
    ) -> None:
        """Render the gesture-engine status overlay."""

        recognized_preview = self._recognized_text[-RECOGNIZED_TEXT_PREVIEW_LENGTH:] or "-"
        recent_characters = " ".join(self._last_chars) if self._last_chars else "-"
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
            text=f"App: {context_snapshot.voice_label}",
            color=HUD_TEXT_COLOR,
        )
        self._put_hud_text(
            cv2_module=cv2_module,
            frame=frame,
            font_face=font_face,
            line_index=6,
            text=f"Profile: {context_snapshot.profile_name}",
            color=HUD_TEXT_COLOR,
        )
        self._put_hud_text(
            cv2_module=cv2_module,
            frame=frame,
            font_face=font_face,
            line_index=7,
            text=f"Fatigue: {self._fatigue_level:.2f}",
            color=HUD_WARNING_COLOR if self._fatigue_level > 0.0 else HUD_TEXT_COLOR,
        )
        self._put_hud_text(
            cv2_module=cv2_module,
            frame=frame,
            font_face=font_face,
            line_index=8,
            text=f"Smoothing: {self._active_smoothing_alpha:.2f}",
            color=HUD_WARNING_COLOR if self._active_smoothing_alpha == FATIGUE_ALPHA else HUD_TEXT_COLOR,
        )
        self._put_hud_text(
            cv2_module=cv2_module,
            frame=frame,
            font_face=font_face,
            line_index=9,
            text=f"Text: {recognized_preview}",
            color=HUD_TEXT_COLOR,
        )
        self._put_hud_text(
            cv2_module=cv2_module,
            frame=frame,
            font_face=font_face,
            line_index=10,
            text=f"Recent: {recent_characters}",
            color=HUD_TEXT_COLOR,
        )
        self._put_hud_text(
            cv2_module=cv2_module,
            frame=frame,
            font_face=font_face,
            line_index=11,
            text=f"FPS: {fps:.1f}",
            color=HUD_TEXT_COLOR,
        )
        self._put_hud_text(
            cv2_module=cv2_module,
            frame=frame,
            font_face=font_face,
            line_index=12,
            text=f"Hold frames: {GESTURE_HOLD_FRAMES}",
            color=HUD_TEXT_COLOR,
        )
        click_age = time.perf_counter() - self._last_click_message_time
        click_text = "Last click: fired" if click_age <= CLICK_MESSAGE_SECONDS else "Last click: waiting"
        self._put_hud_text(
            cv2_module=cv2_module,
            frame=frame,
            font_face=font_face,
            line_index=13,
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
        context_snapshot: Any,
        fatigue_status: Any,
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
        self._shared_state["active_app"] = context_snapshot.voice_label
        self._shared_state["active_profile"] = context_snapshot.profile_name
        self._shared_state["recognized_text"] = self._recognized_text
        self._shared_state["confidence"] = max(analysis.action_confidence, self._last_write_confidence)
        self._shared_state["fps"] = fps
        self._shared_state["fatigue_level"] = self._fatigue_level
        self._shared_state["last_chars"] = list(self._last_chars)
        self._shared_state["modifier_active"] = analysis.modifier_active or MODIFIER_NONE
        self._shared_state["ngrok_url"] = self._ngrok_url
        self._shared_state["two_hand_mode"] = analysis.modifier_hand is not None
        self._write_dashboard_snapshot()


def configure_logging() -> None:
    """Configure logging for the main runtime."""

    logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the runtime entry point."""

    parser = argparse.ArgumentParser(description="Run the KINESYS gesture engine.")
    return parser.parse_args()


def main() -> int:
    """Run the COMMIT 6 KINESYS application."""

    parse_args()
    application = KinesysGestureEngineApp()
    return application.run()


if __name__ == "__main__":
    raise SystemExit(main())
