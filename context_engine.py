"""Active application context detection with JSON profile loading."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from config import PROFILES_DIR

try:
    import psutil
    import win32gui
    import win32process
    _WIN32 = True
except ImportError:
    _WIN32 = False

LOGGER = logging.getLogger(__name__)

PROFILE_ALIASES = {
    "chrome":  ("chrome", "msedge", "brave", "opera", "firefox"),
    "code":    ("code", "vscodium"),
    "zoom":    ("zoom", "zoomus"),
    "spotify": ("spotify",),
    "youtube": ("youtube",),
}


@dataclass(slots=True)
class ContextSnapshot:
    active_app: str
    window_title: str
    profile_name: str
    profile: dict
    app_changed: bool


class ContextEngine:
    def __init__(self) -> None:
        self._profiles_dir = Path(PROFILES_DIR)
        self._cache: dict[str, dict] = {}
        self._last_app = ""
        self._last_profile = "default"
        self._default_profile = self._load_profile("default")

    def get_context(self) -> ContextSnapshot:
        app, title = self._detect_active_app()
        profile_name = self._resolve_profile(app)
        profile = self._load_profile(profile_name)
        changed = app != self._last_app
        self._last_app = app
        self._last_profile = profile_name
        return ContextSnapshot(
            active_app=app, window_title=title,
            profile_name=profile_name, profile=profile, app_changed=changed,
        )

    def _detect_active_app(self) -> tuple[str, str]:
        if _WIN32:
            try:
                hwnd = win32gui.GetForegroundWindow()
                title = win32gui.GetWindowText(hwnd)
                _, pid = win32process.GetWindowThreadProcessId(hwnd)
                proc = psutil.Process(pid)
                name = proc.name().lower().replace(".exe", "")
                return name, title
            except Exception:
                pass
        return "default", ""

    def _resolve_profile(self, app_name: str) -> str:
        for profile_key, aliases in PROFILE_ALIASES.items():
            if any(alias in app_name for alias in aliases):
                return profile_key
        return "default"

    def _load_profile(self, name: str) -> dict:
        if name in self._cache:
            return self._cache[name]
        path = self._profiles_dir / f"{name}.json"
        if path.exists():
            try:
                data = json.loads(path.read_text())
                self._cache[name] = data
                return data
            except Exception as exc:
                LOGGER.warning("Failed to load profile %s: %s", name, exc)
        self._cache[name] = {}
        return {}
