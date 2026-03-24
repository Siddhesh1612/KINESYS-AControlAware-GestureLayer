"""QWERTY keyboard overlay rendered directly on the camera frame.

Pinch to select keys. No collision with gesture control — keyboard only
activates when STATE_KEYBOARD is active.
"""
from __future__ import annotations

import cv2
import pyautogui

_ROWS = [
    ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
    ["A", "S", "D", "F", "G", "H", "J", "K", "L"],
    ["Z", "X", "C", "V", "B", "N", "M"],
    ["BKSP", "SPACE", "ENTER"],
]

# Colors (BGR)
C_BG      = (20,  20,  20)
C_KEY     = (60,  60,  60)
C_SPECIAL = (50,  50, 110)
C_HOVER   = (180,  80,   0)
C_PRESS   = ( 30, 160,  30)
C_BORDER  = (140, 140, 140)
C_WHITE   = (255, 255, 255)
C_YELLOW  = (  0, 255, 255)
C_ORANGE  = (  0, 165, 255)
C_RED     = (  0,   0, 255)

PINCH_THRESH = 35
SELECT_COOL  = 22   # frames cooldown between presses
KB_HEIGHT_RATIO = 0.46


class KeyboardOverlay:
    """Renders a QWERTY keyboard on the bottom of the frame.

    Call draw(frame, fingertip_px, is_pinching) each frame.
    Returns the key label pressed this frame, or "".
    """

    def __init__(self) -> None:
        self._key_rects: list = []
        self._last_w = 0
        self._last_h = 0
        self._select_cd = 0
        self._hovered = ""
        self._last_key = ""
        self._flash_cd = 0
        self.typed_word = ""

    def draw(self, frame, fingertip: tuple[int, int], is_pinching: bool) -> str:
        h, w = frame.shape[:2]
        if w != self._last_w or h != self._last_h:
            self._build_rects(w, h)
            self._last_w, self._last_h = w, h

        if self._select_cd > 0:
            self._select_cd -= 1

        self._hovered = ""
        fx, fy = int(fingertip[0]), int(fingertip[1])

        # Dark panel behind keyboard
        if self._key_rects:
            top_y = min(r[2] for r in self._key_rects) - 8
            panel = frame.copy()
            cv2.rectangle(panel, (0, top_y), (w, h), C_BG, -1)
            cv2.addWeighted(panel, 0.82, frame, 0.18, 0, frame)
            cv2.line(frame, (0, top_y), (w, top_y), (100, 100, 100), 1)

        for (label, x1, y1, x2, y2) in self._key_rects:
            hover = x1 <= fx <= x2 and y1 <= fy <= y2
            if hover:
                self._hovered = label
            if hover and is_pinching and self._select_cd == 0:
                bg = C_PRESS
            elif hover:
                bg = C_HOVER
            elif label in ("BKSP", "SPACE", "ENTER"):
                bg = C_SPECIAL
            else:
                bg = C_KEY
            cv2.rectangle(frame, (x1, y1), (x2, y2), bg, -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), C_BORDER, 1, cv2.LINE_AA)
            kh = y2 - y1
            fs = max(0.28, min(0.52, (kh - 6) / 30.0))
            sz = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)[0]
            tx = x1 + (x2 - x1 - sz[0]) // 2
            ty = y1 + (y2 - y1 + sz[1]) // 2
            col = C_YELLOW if label in ("BKSP", "SPACE", "ENTER") else C_WHITE
            cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, col, 1, cv2.LINE_AA)

        # Fingertip cursor
        cv2.circle(frame, (fx, fy), 10, C_ORANGE, 2, cv2.LINE_AA)
        cv2.circle(frame, (fx, fy),  3, C_ORANGE, -1, cv2.LINE_AA)
        if is_pinching:
            cv2.circle(frame, (fx, fy), 16, C_RED, 2, cv2.LINE_AA)

        # Flash last key
        if self._flash_cd > 0:
            self._flash_cd -= 1
            fsz = cv2.getTextSize(self._last_key, cv2.FONT_HERSHEY_DUPLEX, 1.4, 2)[0]
            bx = w - fsz[0] - 20
            by = (min(r[2] for r in self._key_rects) if self._key_rects else h // 2) - 50
            cv2.rectangle(frame, (bx - 6, by - fsz[1] - 6), (bx + fsz[0] + 6, by + 6), (40, 40, 40), -1)
            cv2.putText(frame, self._last_key, (bx, by), cv2.FONT_HERSHEY_DUPLEX, 1.4, C_ORANGE, 2, cv2.LINE_AA)

        # Typed word strip
        if self._key_rects:
            strip_y = min(r[2] for r in self._key_rects) - 10
            disp = self.typed_word[-38:]
            cv2.putText(frame, disp + "_", (10, strip_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, C_WHITE, 1, cv2.LINE_AA)

        # Key selection
        if is_pinching and self._hovered and self._select_cd == 0:
            key = self._hovered
            self._select_cd = SELECT_COOL
            self._last_key = key
            self._flash_cd = 18
            self._handle_key(key)
            return key
        return ""

    def _build_rects(self, fw: int, fh: int) -> None:
        self._key_rects.clear()
        n_rows = len(_ROWS)
        pad = max(3, fw // 160)
        kb_h = int(fh * KB_HEIGHT_RATIO)
        key_h = max((kb_h - pad * (n_rows + 1)) // n_rows, 22)
        usable_w = fw - 2 * pad
        total_kb = n_rows * (key_h + pad) + pad
        start_y = fh - total_kb - 2

        for row_i, row in enumerate(_ROWS):
            y1 = start_y + row_i * (key_h + pad)
            y2 = y1 + key_h
            if row == ["BKSP", "SPACE", "ENTER"]:
                bksp_w = int(usable_w * 0.17)
                enter_w = int(usable_w * 0.17)
                space_w = usable_w - bksp_w - enter_w - 2 * pad
                x = pad
                self._key_rects.append(("BKSP",  x, y1, x + bksp_w, y2)); x += bksp_w + pad
                self._key_rects.append(("SPACE", x, y1, x + space_w, y2)); x += space_w + pad
                self._key_rects.append(("ENTER", x, y1, x + enter_w, y2))
                continue
            n_keys = len(row)
            key_w = (usable_w - pad * (n_keys - 1)) // n_keys
            x_off = pad + (usable_w - (key_w * n_keys + pad * (n_keys - 1))) // 2
            for col_i, label in enumerate(row):
                x1k = x_off + col_i * (key_w + pad)
                self._key_rects.append((label, x1k, y1, x1k + key_w, y2))

    def _handle_key(self, key: str) -> None:
        if key == "BKSP":
            pyautogui.press("backspace")
            if self.typed_word:
                self.typed_word = self.typed_word[:-1]
        elif key == "SPACE":
            pyautogui.press("space")
            self.typed_word = ""
        elif key == "ENTER":
            pyautogui.press("enter")
            self.typed_word = ""
        else:
            pyautogui.write(key)
            self.typed_word += key
