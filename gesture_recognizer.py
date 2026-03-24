"""Context-aware gesture-to-action dispatcher and cross-platform app launcher.

Ported from Kynesisnofeat/gesture_engine/gesture_recognizer.py and integrated
into the root KINESYS project.
"""

from __future__ import annotations

import platform
import subprocess

import pyautogui

# ── Landmark indices ──────────────────────────────────────────────────────────
WRIST      = 0
THUMB_TIP  = 4;  THUMB_MCP  = 2
INDEX_TIP  = 8;  INDEX_PIP  = 6
MIDDLE_TIP = 12; MIDDLE_PIP = 10
RING_TIP   = 16; RING_PIP   = 14
PINKY_TIP  = 20; PINKY_PIP  = 18

# ── Context-aware gesture → action map ───────────────────────────────────────
GESTURE_MAP: dict[str, dict[str, tuple]] = {
    "default": {
        "pointing":  ("click",    [],                        "click"),
        "peace":     ("shortcut", ["ctrl", "t"],             "new tab"),
        "open_hand": ("shortcut", ["win"],                   "start menu"),
        "fist":      ("shortcut", ["ctrl", "z"],             "undo"),
        "rock_on":   ("shortcut", ["ctrl", "shift", "t"],    "reopen tab"),
        "thumbs_up": ("shortcut", ["ctrl", "s"],             "save"),
    },
    "browser": {
        "peace":     ("shortcut", ["ctrl", "t"],             "new tab"),
        "fist":      ("shortcut", ["ctrl", "w"],             "close tab"),
        "pointing":  ("shortcut", ["ctrl", "l"],             "address bar"),
        "rock_on":   ("shortcut", ["ctrl", "shift", "t"],    "reopen tab"),
        "thumbs_up": ("shortcut", ["ctrl", "r"],             "reload"),
        "open_hand": ("shortcut", ["ctrl", "n"],             "new window"),
    },
    "editor": {
        "peace":     ("shortcut", ["ctrl", "shift", "p"],    "command palette"),
        "fist":      ("shortcut", ["ctrl", "`"],             "terminal"),
        "open_hand": ("shortcut", ["f5"],                    "run/debug"),
        "thumbs_up": ("shortcut", ["ctrl", "s"],             "save"),
        "rock_on":   ("shortcut", ["ctrl", "shift", "k"],    "delete line"),
        "pointing":  ("shortcut", ["ctrl", "p"],             "quick open"),
    },
    "zoom": {
        "peace":     ("shortcut", ["alt", "y"],              "raise hand"),
        "fist":      ("shortcut", ["alt", "a"],              "mute/unmute"),
        "open_hand": ("shortcut", ["alt", "v"],              "video on/off"),
        "thumbs_up": ("shortcut", ["alt", "r"],              "react"),
    },
    "youtube": {
        "peace":     ("shortcut", ["f"],                     "fullscreen"),
        "pointing":  ("shortcut", ["k"],                     "play/pause"),
        "open_hand": ("shortcut", ["m"],                     "mute"),
        "thumbs_up": ("shortcut", ["shift", "."],            "speed up"),
        "fist":      ("shortcut", ["shift", ","],            "slow down"),
    },
    "spotify": {
        "pointing":  ("shortcut", ["space"],                 "play/pause"),
        "peace":     ("shortcut", ["ctrl", "right"],         "next track"),
        "fist":      ("shortcut", ["ctrl", "left"],          "prev track"),
        "thumbs_up": ("shortcut", ["ctrl", "up"],            "volume up"),
        "open_hand": ("shortcut", ["ctrl", "down"],          "volume down"),
    },
    "terminal": {
        "peace":     ("shortcut", ["ctrl", "shift", "t"],    "new tab"),
        "fist":      ("shortcut", ["ctrl", "c"],             "interrupt"),
        "open_hand": ("shortcut", ["ctrl", "shift", "v"],    "paste"),
        "thumbs_up": ("shortcut", ["ctrl", "shift", "c"],    "copy"),
    },
}

# ── App registry for launcher mode ───────────────────────────────────────────
APP_REGISTRY: dict[str, list[str]] = {
    "chrome":      ["chrome", "chro", "chr", "ch"],
    "firefox":     ["firefox", "fire", "fox"],
    "brave":       ["brave", "brav"],
    "vscode":      ["vscode", "code", "vs"],
    "zoom":        ["zoom", "zoo", "zo"],
    "spotify":     ["spotify", "spot", "spo", "sp"],
    "terminal":    ["terminal", "term", "ter", "bash"],
    "vlc":         ["vlc", "vl"],
    "notepad":     ["notepad", "note"],
    "discord":     ["discord", "disc", "dis"],
    "whatsapp":    ["whatsapp", "what", "wha", "wa"],
    "calculator":  ["calculator", "calc", "cal"],
}

LAUNCH_CMDS: dict[str, dict[str, str]] = {
    "Windows": {
        "chrome":     "start chrome",
        "firefox":    "start firefox",
        "brave":      "start brave",
        "vscode":     "code",
        "zoom":       "start zoom",
        "spotify":    "start spotify",
        "terminal":   "start cmd",
        "notepad":    "notepad",
        "calculator": "calc",
        "discord":    "start discord",
        "whatsapp":   "start whatsapp",
    },
    "Darwin": {
        "chrome":     "open -a 'Google Chrome'",
        "firefox":    "open -a Firefox",
        "brave":      "open -a 'Brave Browser'",
        "vscode":     "open -a 'Visual Studio Code'",
        "zoom":       "open -a zoom.us",
        "spotify":    "open -a Spotify",
        "terminal":   "open -a Terminal",
        "calculator": "open -a Calculator",
        "discord":    "open -a Discord",
    },
    "Linux": {
        "chrome":     "google-chrome",
        "firefox":    "firefox",
        "brave":      "brave-browser",
        "vscode":     "code",
        "zoom":       "zoom",
        "spotify":    "spotify",
        "terminal":   "gnome-terminal",
        "vlc":        "vlc",
        "notepad":    "gedit",
        "calculator": "gnome-calculator",
        "discord":    "discord",
    },
}


def classify_gesture(landmarks: list[tuple[int, int]], hand_label: str) -> str:
    """Classify a hand pose from pixel landmarks into a named gesture.

    Args:
        landmarks: List of (x, y) pixel tuples from HandTracker (21 points).
        hand_label: "Right" or "Left" as reported by MediaPipe.

    Returns:
        One of: pointing, peace, open_hand, fist, rock_on, thumbs_up, tracking.
    """
    lm = landmarks

    # Thumb: compare x-axis (mirrored frame)
    if hand_label == "Right":
        thumb_up = lm[THUMB_TIP][0] < lm[THUMB_MCP][0]
    else:
        thumb_up = lm[THUMB_TIP][0] > lm[THUMB_MCP][0]

    index_up  = lm[INDEX_TIP][1]  < lm[INDEX_PIP][1]
    middle_up = lm[MIDDLE_TIP][1] < lm[MIDDLE_PIP][1]
    ring_up   = lm[RING_TIP][1]   < lm[RING_PIP][1]
    pinky_up  = lm[PINKY_TIP][1]  < lm[PINKY_PIP][1]

    if not any([index_up, middle_up, ring_up, pinky_up]):
        return "thumbs_up" if thumb_up else "fist"

    if index_up and not middle_up and not ring_up and not pinky_up:
        return "pointing"

    if index_up and middle_up and not ring_up and not pinky_up:
        return "peace"

    if index_up and not middle_up and not ring_up and pinky_up:
        return "rock_on"

    if index_up and middle_up and ring_up and pinky_up:
        return "open_hand"

    return "tracking"


def dispatch_action(gesture: str, context: str) -> str:
    """Execute the OS action for a gesture in the given context.

    Args:
        gesture: Gesture name from classify_gesture().
        context: Active app context key (e.g. "browser", "editor").

    Returns:
        Human-readable description of the action taken, or empty string.
    """
    mapping = GESTURE_MAP.get(context, GESTURE_MAP["default"])
    entry = mapping.get(gesture) or GESTURE_MAP["default"].get(gesture)
    if not entry:
        return ""

    action_type, keys, description = entry
    try:
        if action_type == "click":
            pyautogui.click()
        elif action_type == "shortcut":
            mapped = []
            for k in keys:
                if k == "win":
                    os_name = platform.system()
                    mapped.append(
                        "win" if os_name == "Windows"
                        else "command" if os_name == "Darwin"
                        else "super"
                    )
                else:
                    mapped.append(k)
            pyautogui.hotkey(*mapped)
    except Exception as exc:
        return f"ERR: {exc}"

    return f"{gesture} → {description}"


def match_app(word: str) -> tuple[str | None, str | None]:
    """Fuzzy-match a spelled word to a registered app.

    Args:
        word: Partial or full app name spelled via ISL/air-writing.

    Returns:
        (app_key, launch_command) tuple, or (None, None) if no match.
    """
    word_lower = word.lower().strip()
    if not word_lower:
        return None, None

    os_name = platform.system()
    launch_map = LAUNCH_CMDS.get(os_name, LAUNCH_CMDS["Linux"])

    best_key: str | None = None
    best_len = 0
    for app_key, aliases in APP_REGISTRY.items():
        for alias in aliases:
            if word_lower.startswith(alias) or alias.startswith(word_lower):
                if len(alias) > best_len:
                    best_len = len(alias)
                    best_key = app_key

    if best_key and best_key in launch_map:
        return best_key, launch_map[best_key]
    return None, None


def launch_app(command: str) -> bool:
    """Launch an application via a shell command.

    Args:
        command: Shell command string to execute.

    Returns:
        True if the process was spawned successfully.
    """
    try:
        subprocess.Popen(command, shell=True)
        return True
    except Exception as exc:
        print(f"[LAUNCHER] Launch failed: {exc}")
        return False
