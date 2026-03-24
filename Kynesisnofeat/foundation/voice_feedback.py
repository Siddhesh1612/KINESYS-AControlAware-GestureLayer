import threading
import sys

def _get_engine():
    """Lazy-load pyttsx3 so import errors don't crash the whole app."""
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 180)   # words per minute
        engine.setProperty("volume", 1.0)
        return engine
    except Exception as e:
        print(f"[VOICE] pyttsx3 unavailable: {e}")
        return None

_engine = None
_lock   = threading.Lock()

def speak(text: str, blocking: bool = False):
    """
    Speak text via TTS.
    Non-blocking by default so it never stalls the camera loop.
    Falls back to print if pyttsx3 is not installed.
    """
    global _engine
    print(f"[VOICE]: {text}")

    def _run():
        global _engine
        with _lock:
            try:
                if _engine is None:
                    _engine = _get_engine()
                if _engine:
                    _engine.say(text)
                    _engine.runAndWait()
            except Exception as e:
                print(f"[VOICE] speak error: {e}")
                _engine = None   # reset so next call retries init

    if blocking:
        _run()
    else:
        t = threading.Thread(target=_run, daemon=True)
        t.start()
