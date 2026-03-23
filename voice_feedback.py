"""Threaded offline voice feedback helpers for KINESYS."""

from __future__ import annotations

import logging
import queue
import threading
from typing import Any

import pyttsx3


LOGGER = logging.getLogger(__name__)

QUEUE_TIMEOUT_SECONDS = 0.2
STOP_SENTINEL = "__KINESYS_VOICE_STOP__"


class VoiceFeedback:
    """Background text-to-speech dispatcher that never blocks the main loop."""

    def __init__(self) -> None:
        """Initialize the speech queue and start the worker thread."""

        self._queue: queue.Queue[str] = queue.Queue()
        self._shutdown_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run_worker,
            name="kinesys-voice-feedback",
            daemon=True,
        )
        self._thread.start()

    def speak(self, text: str) -> None:
        """Queue one phrase for asynchronous playback."""

        if not text:
            return
        try:
            self._queue.put_nowait(text)
        except Exception as exc:
            LOGGER.exception("Voice queueing failed: %s", exc)

    def shutdown(self) -> None:
        """Stop the worker thread after it drains the queued utterances."""

        self._shutdown_event.set()
        self.speak(STOP_SENTINEL)
        self._thread.join(timeout=None)

    def _run_worker(self) -> None:
        """Run the background speech loop and recover from engine errors."""

        engine = self._initialize_engine()
        while not self._shutdown_event.is_set() or not self._queue.empty():
            try:
                text = self._queue.get(timeout=QUEUE_TIMEOUT_SECONDS)
            except queue.Empty:
                continue
            except Exception as exc:
                LOGGER.exception("Voice queue read failed: %s", exc)
                continue

            if text == STOP_SENTINEL:
                self._queue.task_done()
                continue

            if engine is None:
                engine = self._initialize_engine()
                if engine is None:
                    self._queue.task_done()
                    continue

            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as exc:
                LOGGER.exception("Voice playback failed; reinitializing engine: %s", exc)
                engine = self._recover_engine(engine)
            finally:
                self._queue.task_done()

        self._shutdown_engine(engine)

    def _initialize_engine(self) -> Any | None:
        """Create the pyttsx3 engine instance inside the worker thread."""

        try:
            return pyttsx3.init()
        except Exception as exc:
            LOGGER.exception("Voice engine initialization failed: %s", exc)
            return None

    def _recover_engine(self, engine: Any | None) -> Any | None:
        """Stop the current engine and try to build a fresh one."""

        self._shutdown_engine(engine)
        return self._initialize_engine()

    @staticmethod
    def _shutdown_engine(engine: Any | None) -> None:
        """Stop the pyttsx3 engine safely if it exists."""

        if engine is None:
            return
        try:
            engine.stop()
        except Exception as exc:
            LOGGER.exception("Voice engine shutdown failed: %s", exc)
