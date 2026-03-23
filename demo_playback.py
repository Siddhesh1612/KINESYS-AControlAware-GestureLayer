"""Backup prerecorded demo playback for KINESYS stage demos."""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import cv2

from config import (
    DASHBOARD_HEADLESS,
    DASHBOARD_HOST,
    DASHBOARD_PORT,
    DASHBOARD_REFRESH_MS,
    DASHBOARD_SCRIPT,
    DASHBOARD_STATE_SNAPSHOT,
    DEMO_FPS_FALLBACK,
    DEMO_METADATA_FILE,
    DEMO_VIDEO_FILE,
    DEMO_WINDOW_NAME,
    EXIT_KEY,
    FRAME_ENCODE_EXTENSION,
    FRAME_JPEG_QUALITY,
    HUD_FONT_SCALE,
    HUD_FONT_THICKNESS,
    HUD_LINE_HEIGHT,
    HUD_MARGIN_X,
    HUD_MARGIN_Y,
    HUD_PANEL_ALPHA,
    HUD_PANEL_COLOR,
    HUD_TEXT_COLOR,
    LOG_FORMAT,
    LOG_LEVEL,
    MAIN_WINDOW_NAME,
    MODIFIER_NONE,
    STATE_CURSOR,
    STATE_IDLE,
    STATE_MACRO,
    STATE_SCROLL,
    STATE_WRITE,
)
from voice_feedback import VoiceFeedback


LOGGER = logging.getLogger(__name__)

SNAPSHOT_TEMP_SUFFIX = ".tmp"
DEFAULT_NGROK_URL = ""
DEFAULT_TEXT = ""
DEFAULT_TWO_HAND_MODE = False
DEFAULT_LOOP = False
WINDOW_CLOSE_WAIT_MS = 1
OVERLAY_PANEL_WIDTH = 520
PROCESS_WAIT_TIMEOUT_SECONDS = 2.0
PLAYBACK_START_MESSAGE = "Kinesys demo playback"
PLAYBACK_STOP_MESSAGE = "Playback stopped"
PLAYBACK_EXIT_CODE_SUCCESS = 0
PLAYBACK_EXIT_CODE_FAILURE = 1
NO_DASHBOARD_FLAG = "--no-dashboard"


@dataclass(slots=True)
class PlaybackEvent:
    """One scripted state update applied during demo playback."""

    timestamp_seconds: float
    gesture_state: str
    active_app: str
    active_profile: str
    recognized_text: str
    confidence: float
    fatigue_level: float
    last_chars: list[str]
    modifier_active: str | None


class DemoPlaybackApp:
    """Play a prerecorded session and keep the dashboard fed with snapshot updates."""

    def __init__(
        self,
        video_path: str,
        metadata_path: str,
        launch_dashboard: bool,
        loop_video: bool,
    ) -> None:
        """Initialize playback paths and runtime options."""

        self._video_path = Path(video_path)
        self._metadata_path = Path(metadata_path)
        self._launch_dashboard = launch_dashboard
        self._loop_video = loop_video
        self._dashboard_process: subprocess.Popen[str] | None = None
        self._voice_feedback = VoiceFeedback()

    def run(self) -> int:
        """Start the backup demo playback and return a process exit code."""

        if not self._video_path.exists():
            LOGGER.error("Demo video not found: %s", self._video_path)
            print(f"Demo video not found: {self._video_path}")
            self._shutdown()
            return PLAYBACK_EXIT_CODE_FAILURE

        metadata_events = self._load_metadata_events()
        if self._launch_dashboard:
            self._dashboard_process = self._start_dashboard()

        capture = cv2.VideoCapture(str(self._video_path))
        if not capture.isOpened():
            LOGGER.error("Unable to open demo video: %s", self._video_path)
            print(f"Unable to open demo video: {self._video_path}")
            self._shutdown()
            return PLAYBACK_EXIT_CODE_FAILURE

        fps = capture.get(cv2.CAP_PROP_FPS) or DEMO_FPS_FALLBACK
        frame_delay_ms = max(int(1000.0 / float(fps)), WINDOW_CLOSE_WAIT_MS)
        playback_started_at = time.perf_counter()
        self._voice_feedback.speak(PLAYBACK_START_MESSAGE)

        try:
            while True:
                frame_ok, frame = capture.read()
                if not frame_ok:
                    if self._loop_video:
                        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        playback_started_at = time.perf_counter()
                        continue
                    break

                elapsed_seconds = time.perf_counter() - playback_started_at
                active_event = self._select_event(metadata_events, elapsed_seconds)
                annotated_frame = self._annotate_frame(frame=frame, event=active_event, fps=float(fps))
                self._write_snapshot(frame=annotated_frame, event=active_event, fps=float(fps))

                cv2.imshow(DEMO_WINDOW_NAME, annotated_frame)
                pressed_key = cv2.waitKey(frame_delay_ms) & 0xFF
                if pressed_key == ord(EXIT_KEY):
                    self._voice_feedback.speak(PLAYBACK_STOP_MESSAGE)
                    break
        except KeyboardInterrupt:
            self._voice_feedback.speak(PLAYBACK_STOP_MESSAGE)
        finally:
            capture.release()
            cv2.destroyAllWindows()
            self._shutdown()

        return PLAYBACK_EXIT_CODE_SUCCESS

    def _load_metadata_events(self) -> list[PlaybackEvent]:
        """Load scripted playback events from disk or fall back to a default timeline."""

        if not self._metadata_path.exists():
            return self._default_metadata_events()

        try:
            with self._metadata_path.open("r", encoding="utf-8") as metadata_file:
                payload = json.load(metadata_file)
        except Exception as exc:
            LOGGER.exception("Failed to load demo metadata: %s", exc)
            return self._default_metadata_events()

        raw_events = payload.get("events", []) if isinstance(payload, dict) else []
        if not isinstance(raw_events, list):
            return self._default_metadata_events()

        events: list[PlaybackEvent] = []
        for raw_event in raw_events:
            if not isinstance(raw_event, dict):
                continue
            try:
                events.append(PlaybackEvent(**raw_event))
            except Exception:
                continue

        return events or self._default_metadata_events()

    def _default_metadata_events(self) -> list[PlaybackEvent]:
        """Return a simple scripted timeline used when no metadata file exists."""

        return [
            PlaybackEvent(0.0, STATE_IDLE, "chrome.exe", "chrome", DEFAULT_TEXT, 0.0, 0.0, [], MODIFIER_NONE),
            PlaybackEvent(3.0, STATE_CURSOR, "chrome.exe", "chrome", DEFAULT_TEXT, 0.91, 0.0, [], MODIFIER_NONE),
            PlaybackEvent(6.0, STATE_CURSOR, "Code.exe", "code", DEFAULT_TEXT, 0.93, 0.0, [], "CTRL"),
            PlaybackEvent(9.0, STATE_WRITE, "Code.exe", "code", "KIN", 0.88, 0.0, ["K", "I", "N"], MODIFIER_NONE),
            PlaybackEvent(12.0, STATE_MACRO, "Zoom.exe", "zoom", "KINESYS", 0.90, 0.0, ["E", "S", "Y", "S"], MODIFIER_NONE),
            PlaybackEvent(15.0, STATE_SCROLL, "Zoom.exe", "zoom", "KINESYS", 0.86, 0.1, ["E", "S", "Y", "S"], "SHIFT"),
        ]

    def _select_event(self, events: list[PlaybackEvent], elapsed_seconds: float) -> PlaybackEvent:
        """Select the latest event whose timestamp is not greater than elapsed playback time."""

        selected_event = events[0]
        for event in events:
            if event.timestamp_seconds <= elapsed_seconds:
                selected_event = event
            else:
                break
        return selected_event

    def _annotate_frame(self, frame: Any, event: PlaybackEvent, fps: float) -> Any:
        """Overlay lightweight status text on the playback frame."""

        panel = frame.copy()
        panel_height = HUD_MARGIN_Y + (HUD_LINE_HEIGHT * 6)
        cv2.rectangle(panel, (0, 0), (OVERLAY_PANEL_WIDTH, panel_height), HUD_PANEL_COLOR, cv2.FILLED)
        cv2.addWeighted(panel, HUD_PANEL_ALPHA, frame, 1.0 - HUD_PANEL_ALPHA, 0.0, frame)

        overlay_lines = [
            f"Mode: backup playback",
            f"State: {event.gesture_state}",
            f"App: {event.active_app}",
            f"Profile: {event.active_profile}",
            f"Text: {event.recognized_text or '-'}",
            f"FPS: {fps:.1f}",
        ]

        for index, text in enumerate(overlay_lines):
            text_y = HUD_MARGIN_Y + (index * HUD_LINE_HEIGHT)
            cv2.putText(
                frame,
                text,
                (HUD_MARGIN_X, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                HUD_FONT_SCALE,
                HUD_TEXT_COLOR,
                HUD_FONT_THICKNESS,
                cv2.LINE_AA,
            )

        return frame

    def _write_snapshot(self, frame: Any, event: PlaybackEvent, fps: float) -> None:
        """Write dashboard-compatible snapshot data for the fallback playback flow."""

        encoded_ok, encoded_frame = cv2.imencode(
            FRAME_ENCODE_EXTENSION,
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, FRAME_JPEG_QUALITY],
        )
        frame_b64 = ""
        if encoded_ok:
            frame_b64 = base64.b64encode(encoded_frame.tobytes()).decode("utf-8")

        payload = {
            "frame_b64": frame_b64,
            "gesture_state": event.gesture_state,
            "active_app": event.active_app,
            "active_profile": event.active_profile,
            "recognized_text": event.recognized_text,
            "confidence": event.confidence,
            "fps": fps,
            "fatigue_level": event.fatigue_level,
            "last_chars": list(event.last_chars),
            "modifier_active": event.modifier_active,
            "ngrok_url": DEFAULT_NGROK_URL,
            "two_hand_mode": DEFAULT_TWO_HAND_MODE,
        }

        snapshot_path = Path(DASHBOARD_STATE_SNAPSHOT)
        temp_path = snapshot_path.with_name(snapshot_path.name + SNAPSHOT_TEMP_SUFFIX)

        try:
            with temp_path.open("w", encoding="utf-8") as snapshot_file:
                json.dump(payload, snapshot_file)
            os.replace(temp_path, snapshot_path)
        except Exception as exc:
            LOGGER.exception("Failed to write demo snapshot: %s", exc)

    def _start_dashboard(self) -> subprocess.Popen[str] | None:
        """Launch the Streamlit dashboard for the fallback playback flow."""

        environment = os.environ.copy()
        environment.setdefault("STREAMLIT_SERVER_HEADLESS", DASHBOARD_HEADLESS)

        command = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            DASHBOARD_SCRIPT,
            "--server.port",
            str(DASHBOARD_PORT),
            "--server.address",
            DASHBOARD_HOST,
        ]

        try:
            process = subprocess.Popen(command, env=environment)
        except Exception as exc:
            LOGGER.exception("Failed to launch dashboard for playback: %s", exc)
            return None

        LOGGER.info("Playback dashboard available at http://%s:%s", DASHBOARD_HOST, DASHBOARD_PORT)
        return process

    def _shutdown(self) -> None:
        """Terminate background helpers started by the playback app."""

        if self._dashboard_process is not None and self._dashboard_process.poll() is None:
            try:
                self._dashboard_process.terminate()
                self._dashboard_process.wait(timeout=PROCESS_WAIT_TIMEOUT_SECONDS)
            except Exception:
                try:
                    self._dashboard_process.kill()
                except Exception:
                    pass

        self._voice_feedback.shutdown()


def configure_logging() -> None:
    """Configure process-wide logging for the playback entry point."""

    logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the backup demo runner."""

    parser = argparse.ArgumentParser(description="Play a prerecorded KINESYS demo session.")
    parser.add_argument("--video", default=DEMO_VIDEO_FILE, help="Path to the prerecorded demo video.")
    parser.add_argument(
        "--metadata",
        default=DEMO_METADATA_FILE,
        help="Optional JSON file with scripted dashboard state events.",
    )
    parser.add_argument("--loop", action="store_true", default=DEFAULT_LOOP, help="Loop the video.")
    parser.add_argument(
        NO_DASHBOARD_FLAG,
        action="store_true",
        help="Skip launching the Streamlit dashboard during playback.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the backup playback command-line entry point."""

    configure_logging()
    args = parse_args()
    LOGGER.info("Starting %s playback mode", MAIN_WINDOW_NAME)
    application = DemoPlaybackApp(
        video_path=args.video,
        metadata_path=args.metadata,
        launch_dashboard=not args.no_dashboard,
        loop_video=args.loop,
    )
    return application.run()


if __name__ == "__main__":
    raise SystemExit(main())
