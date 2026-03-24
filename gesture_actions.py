"""Context-aware gesture → OS action dispatcher with app launcher support."""
from __future__ import annotations

import platform
import subprocess

import pyautogui

# Context-aware gesture → action map
# Uses kinesisv3 gesture names: pointing, peace, open_hand, fist, rock_on, thumbs_up
GESTURE_MAP: dict[str, dict[str, tuple]] = {
    "default": {
        "pointing":  ("click",    [],                        "left click"),
        "peace":     ("shortcut", ["ctrl", "t"],             "new tab"),
        "open_hand": ("shortcut", ["win"],                   "start menu"),
        "fist":      ("shortcut", ["ctrl", "z"],             "undo"),
        "rock_on":   ("shortcut", ["ctrl", "shift", "t"],    "reopen tab"),
    },
    "browser": {
        "peace":     ("shortcut", ["ctrl", "t"],             "new tab"),
        "fist":      ("shortcut", ["ctrl", "w"],             "close tab"),
        "pointing":  ("shortcut", ["ctrl", "l"],             "address bar"),
        "rock_on":   ("shortcut", ["ctrl", "shift", "t"],    "reopen tab"),
        "open_hand": ("shortcut", ["ctrl", "n"],             "new window"),
    },
    "editor": {
        "peace":     ("shortcut", ["ctrl", "shift", "p"],    "command palette"),
        "fist":      ("shortcut", ["ctrl", "`"],             "terminal"),
        "open_hand": ("shortcut", ["f5"],                    "run/debug"),
        "rock_on":   ("shortcut", ["ctrl", "shift", "k"],    "delete line"),
        "pointing":  ("shortcut", ["ctrl", "p"],             "quick open"),
    },
    "zoom": {
        "peace":     ("shortcut", ["alt", "y"],              "raise hand"),
        "fist":      ("shortcut", ["alt", "a"],              "mute/unmute"),
        "open_hand": ("shortcut", ["alt", "v"],              "video on/off"),
    },
    "youtube": {
        "peace":     ("shortcut", ["f"],                     "fullscreen"),
        "pointing":  ("shortcut", ["k"],                     "play/pause"),
        "open_hand": ("shortcut", ["m"],                     "mute"),
        "fist":      ("shortcut", ["shift", ","],            "slow down"),
    },
    "spotify": {
        "pointing":  ("shortcut", ["space"],                 "play/pause"),
        "peace":     ("shortcut", ["ctrl", "right"],         "next track"),
        "fist":      ("shortcut", ["ctrl", "left"],          "prev track"),
        "open_hand": ("shortcut", ["ctrl", "down"],          "volume down"),
    },
    "terminal": {
        "peace":     ("shortcut", ["ctrl", "shift", "t"],    "new tab"),
        "fist":      ("shortcut", ["ctrl", "c"],             "interrupt"),
        "open_hand": ("shortcut", ["ctrl", "shift", "v"],    "paste"),
    },
}

APP_REGISTRY: dict[str, list[str]] = {
    "chrome":     ["chrome", "chro", "chr"],
    "firefox":    ["firefox", "fire", "fox"],
    "brave":      ["brave", "brav"],
    "vscode":     ["vscode", "code", "vs"],
    "zoom":       ["zoom", "zoo"],
    "spotify":    ["spotify", "spot", "spo"],
    "terminal":   ["terminal", "term", "bash"],
    "notepad":    ["notepad", "note"],
    "discord":    ["discord", "disc"],
    "calculator": ["calculator", "calc"],
}

LAUNCH_CMDS: dict[str, dict[str, str]] = {
    "Windows": {
        "chrome": "start chrome", "firefox": "start firefox",
        "brave": "start brave", "vscode": "code", "zoom": "start zoom",
        "spotify": "start spotify", "terminal": "start cmd",
        "notepad": "notepad", "calculator": "calc",
        "discord": "start discord",
    },
    "Darwin": {
        "chrome": "open -a 'Google Chrome'", "firefox": "open -a Firefox",
        "brave": "open -a 'Brave Browser'", "vscode": "open -a 'Visual Studio Code'",
        "zoom": "open -a zoom.us", "spotify": "open -a Spotify",
        "terminal": "open -a Terminal", "calculator": "open -a Calculator",
        "discord": "open -a Discord",
    },
    "Linux": {
        "chrome": "google-chrome", "firefox": "firefox",
        "brave": "brave-browser", "vscode": "code", "zoom": "zoom",
        "spotify": "spotify", "terminal": "gnome-terminal",
        "notepad": "gedit", "calculator": "gnome-calculator",
        "discord": "discord",
    },
}


def dispatch_action(gesture: str, context: str, cursor_ctrl=None) -> str:
    """Execute the OS action for a gesture in the given context."""
    mapping = GESTURE_MAP.get(context, GESTURE_MAP["default"])
    entry = mapping.get(gesture) or GESTURE_MAP["default"].get(gesture)
    if not entry:
        return ""
    action_type, keys, description = entry
    try:
        if action_type == "click":
            if cursor_ctrl:
                cursor_ctrl.click()
            else:
                pyautogui.click()
        elif action_type == "shortcut":
            mapped = []
            for k in keys:
                if k == "win":
                    os_name = platform.system()
                    mapped.append("win" if os_name == "Windows"
                                  else "command" if os_name == "Darwin" else "super")
                else:
                    mapped.append(k)
            pyautogui.hotkey(*mapped)
    except Exception as exc:
        return f"ERR: {exc}"
    return f"{gesture} → {description}"


def match_app(word: str) -> tuple[str | None, str | None]:
    word_lower = word.lower().strip()
    if not word_lower:
        return None, None
    os_name = platform.system()
    launch_map = LAUNCH_CMDS.get(os_name, LAUNCH_CMDS["Linux"])
    best_key, best_len = None, 0
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
    try:
        subprocess.Popen(command, shell=True)
        return True
    except Exception as exc:
        print(f"[LAUNCHER] {exc}")
        return False
