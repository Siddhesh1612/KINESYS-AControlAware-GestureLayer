import ctypes
import platform
import subprocess

def get_active_window_title() -> str:
    try:
        os_name = platform.system()
        if os_name == "Windows":
            hwnd = ctypes.windll.user32.GetForegroundWindow()
            length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
            buf = ctypes.create_unicode_buffer(length + 1)
            ctypes.windll.user32.GetWindowTextW(hwnd, buf, length + 1)
            return buf.value
        elif os_name == "Darwin":
            return subprocess.check_output(
                ["osascript", "-e",
                 'tell application "System Events" to get name of first '
                 'application process whose frontmost is true'],
                text=True).strip()
        elif os_name == "Linux":
            wid = subprocess.check_output(["xdotool", "getactivewindow"], text=True).strip()
            return subprocess.check_output(["xdotool", "getwindowname", wid], text=True).strip()
    except Exception:
        pass
    return ""

_APP_KEYWORDS = {
    "zoom":     ["zoom"],
    "browser":  ["chrome", "chromium", "brave", "firefox", "edge", "mozilla", "opera"],
    "editor":   ["visual studio code", "vscode", "code", "notepad", "sublime", "atom", "vim"],
    "youtube":  ["youtube"],
    "spotify":  ["spotify"],
    "terminal": ["terminal", "cmd", "powershell", "bash", "zsh", "command prompt",
                 "windows terminal", "wt.exe", "alacritty", "kitty", "konsole"],
}

class ContextEngine:
    def __init__(self):
        self.current_context = "default"

    def detect_context(self) -> str:
        title = get_active_window_title().lower()
        for ctx, keywords in _APP_KEYWORDS.items():
            if any(kw in title for kw in keywords):
                self.current_context = ctx
                return ctx
        self.current_context = "default"
        return "default"
