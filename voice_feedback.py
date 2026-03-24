"""Non-blocking threaded TTS voice feedback."""
from __future__ import annotations

import queue
import threading

import pyttsx3

_STOP = "__STOP__"


class VoiceFeedback:
    def __init__(self) -> None:
        self._q: queue.Queue[str] = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="kinesys-voice")
        self._thread.start()

    def speak(self, text: str) -> None:
        if text:
            try:
                self._q.put_nowait(text)
            except Exception:
                pass

    def shutdown(self) -> None:
        self._stop.set()
        self.speak(_STOP)
        self._thread.join()

    def _run(self) -> None:
        engine = self._init_engine()
        while not self._stop.is_set() or not self._q.empty():
            try:
                text = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            if text == _STOP:
                self._q.task_done()
                continue
            if engine is None:
                engine = self._init_engine()
            if engine:
                try:
                    engine.say(text)
                    engine.runAndWait()
                except Exception:
                    engine = self._init_engine()
            self._q.task_done()

    @staticmethod
    def _init_engine():
        try:
            return pyttsx3.init()
        except Exception:
            return None


# Module-level singleton for simple speak() calls
_feedback: VoiceFeedback | None = None


def speak(text: str) -> None:
    global _feedback
    if _feedback is None:
        _feedback = VoiceFeedback()
    _feedback.speak(text)
