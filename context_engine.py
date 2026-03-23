"""Application context detection and profile loading for KINESYS."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path

import psutil
import win32gui
import win32process

from config import PROFILES_DIR


LOGGER = logging.getLogger(__name__)

DEFAULT_PROFILE_NAME = "default"
DEFAULT_PROFILE_FILE = "default.json"
EMPTY_WINDOW_TITLE = ""
UNKNOWN_APP_NAME = "unknown"
PROFILE_SUFFIX = ".json"

PROFILE_ALIASES = {
    "chrome": ("chrome", "msedge", "brave", "opera", "vivaldi"),
    "code": ("code", "code - insiders"),
    "zoom": ("zoom", "zoomus", "zoom.us"),
}

DISPLAY_NAMES = {
    "chrome": "Chrome",
    "code": "VS Code",
    "zoom": "Zoom",
    "default": "Default",
}

GESTURE_PROFILE_KEYS = {
    "PEACE_SIGN": ("peace_sign",),
    "THREE_FINGER_LEFT": ("three_finger_left", "three_finger_swipe"),
    "THREE_FINGER_RIGHT": ("three_finger_right", "three_finger_swipe"),
    "FOUR_FINGER_SWIPE": ("four_finger_swipe",),
    "CIRCLE": ("circle",),
    "PINCH_ZOOM_IN": ("pinch_zoom_in",),
    "PINCH_ZOOM_OUT": ("pinch_zoom_out",),
}

SWIPE_UP_GESTURE = "TWO_FINGER_SWIPE"
SWIPE_UP_PROFILE_KEY = "swipe_up"
SWIPE_DOWN_PROFILE_KEY = "swipe_down"


@dataclass(slots=True)
class ContextSnapshot:
    """Current active application context and resolved action profile."""

    active_app: str
    window_title: str
    profile_name: str
    profile: dict[str, str]
    app_changed: bool
    profile_changed: bool
    voice_label: str


class ContextEngine:
    """Detect the active Windows app and load the matching JSON profile."""

    def __init__(self) -> None:
        """Initialize profile caches and current app tracking."""

        self._profiles_dir = Path(PROFILES_DIR)
        self._profile_cache: dict[str, dict[str, str]] = {}
        self._last_active_app = EMPTY_WINDOW_TITLE
        self._last_profile_name = DEFAULT_PROFILE_NAME
        self._default_profile = self._load_profile(DEFAULT_PROFILE_NAME)

    def get_context(self) -> ContextSnapshot:
        """Return the current active app and its resolved profile snapshot."""

        active_app, window_title = self._detect_active_app()
        profile_name = self._resolve_profile_name(active_app)
        profile = self._load_profile(profile_name)

        app_changed = active_app != self._last_active_app
        profile_changed = profile_name != self._last_profile_name
        voice_label = self._build_voice_label(active_app, profile_name)

        self._last_active_app = active_app
        self._last_profile_name = profile_name

        return ContextSnapshot(
            active_app=active_app,
            window_title=window_title,
            profile_name=profile_name,
            profile=profile,
            app_changed=app_changed,
            profile_changed=profile_changed,
            voice_label=voice_label,
        )

    def resolve_action(
        self,
        gesture_name: str,
        profile: dict[str, str],
        vertical_motion: float = 0.0,
    ) -> str | None:
        """Resolve a gesture name to the current profile action name."""

        if gesture_name == SWIPE_UP_GESTURE:
            if vertical_motion < 0.0 and SWIPE_UP_PROFILE_KEY in profile:
                return profile[SWIPE_UP_PROFILE_KEY]
            if vertical_motion > 0.0 and SWIPE_DOWN_PROFILE_KEY in profile:
                return profile[SWIPE_DOWN_PROFILE_KEY]

        for profile_key in GESTURE_PROFILE_KEYS.get(gesture_name, ()):
            if profile_key in profile:
                return profile[profile_key]

        return None

    def _detect_active_app(self) -> tuple[str, str]:
        """Detect the active foreground app with a safe default fallback."""

        try:
            hwnd = win32gui.GetForegroundWindow()
            window_title = win32gui.GetWindowText(hwnd).strip()
            _, process_id = win32process.GetWindowThreadProcessId(hwnd)
            process_name = psutil.Process(process_id).name().lower()
            normalized_name = self._normalize_app_name(process_name=process_name, window_title=window_title)
            return normalized_name, window_title
        except Exception as exc:
            LOGGER.exception("Active app detection failed; falling back to default profile: %s", exc)
            return DEFAULT_PROFILE_NAME, EMPTY_WINDOW_TITLE

    def _resolve_profile_name(self, active_app: str) -> str:
        """Resolve the preferred profile name for the active app."""

        profile_path = self._profiles_dir / f"{active_app}{PROFILE_SUFFIX}"
        if profile_path.exists():
            return active_app
        return DEFAULT_PROFILE_NAME

    def _normalize_app_name(self, process_name: str, window_title: str) -> str:
        """Normalize the process name to one of the known profile keys when possible."""

        normalized_process_name = Path(process_name).stem.lower().strip()
        normalized_title = window_title.lower().strip()

        for profile_name, aliases in PROFILE_ALIASES.items():
            if normalized_process_name in aliases:
                return profile_name
            if any(alias in normalized_title for alias in aliases):
                return profile_name

        return normalized_process_name or UNKNOWN_APP_NAME

    def _load_profile(self, profile_name: str) -> dict[str, str]:
        """Load one profile file and cache the parsed JSON content."""

        if profile_name in self._profile_cache:
            return self._profile_cache[profile_name]

        profile_path = self._profiles_dir / f"{profile_name}{PROFILE_SUFFIX}"
        if not profile_path.exists():
            if profile_name != DEFAULT_PROFILE_NAME:
                return dict(self._default_profile)
            LOGGER.error("Default profile is missing at %s", profile_path)
            self._profile_cache[profile_name] = {}
            return self._profile_cache[profile_name]

        try:
            with profile_path.open("r", encoding="utf-8") as profile_file:
                loaded_profile = json.load(profile_file)
        except Exception as exc:
            LOGGER.exception("Profile load failed for %s: %s", profile_name, exc)
            loaded_profile = {} if profile_name == DEFAULT_PROFILE_NAME else dict(self._default_profile)

        self._profile_cache[profile_name] = loaded_profile
        return self._profile_cache[profile_name]

    def _build_voice_label(self, active_app: str, profile_name: str) -> str:
        """Return a human-readable label for app-switch voice feedback."""

        if active_app in DISPLAY_NAMES:
            return DISPLAY_NAMES[active_app]
        if profile_name in DISPLAY_NAMES and profile_name != DEFAULT_PROFILE_NAME:
            return DISPLAY_NAMES[profile_name]
        if not active_app or active_app == DEFAULT_PROFILE_NAME:
            return DISPLAY_NAMES[DEFAULT_PROFILE_NAME]
        return active_app.replace("_", " ").title()
