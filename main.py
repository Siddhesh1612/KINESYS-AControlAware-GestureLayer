"""KINESYS v4 — Always-on gesture control for accessibility.

Boots straight into CURSOR mode — the user never has to press anything.
All control is gesture-only so a disabled person can use the PC hands-free.

Gestures (from kinesisv3):
  pointing   → move cursor
  pinch      → left click / select keyboard key
  open_hand  → Windows Start menu
  peace      → context shortcut (new tab, etc.)
  thumbs_up  → context shortcut (save, etc.) / confirm launcher
  rock_on    → open on-screen keyboard (hold)
  fist       → scroll down in SCROLL / cancel in LAUNCHER / back to CURSOR
  tracking   → neutral / no action

State transitions (hold gesture for HOLD_FRAMES):
  CURSOR  → SCROLL    : peace (hold) — scroll mode
  CURSOR  → KEYBOARD  : rock_on (hold) — on-screen keyboard
  CURSOR  → LAUNCHER  : thumbs_up (hold) — app launcher
  SCROLL  → CURSOR    : fist (instant)
  KEYBOARD→ CURSOR    : fist (instant)
  LAUNCHER→ CURSOR    : fist (instant)

Exit: both fists held ~1s → TERMINATED
"""
from __future__ import annotations

import logging
import sys
import time

import cv2
import pyautogui

from config import (
    APP_NAME, CAMERA_ID, EXIT_KEY, FRAME_FLIP_CODE, FRAME_WAIT_KEY_MS,
    HUD_WARNING_COLOR, INDEX_FINGER_TIP_ID, LOG_FORMAT, LOG_LEVEL,
    MAIN_WINDOW_NAME, MODIFIER_NONE, PINCH_THRESHOLD,
    SCROLL_COOLDOWN_SECONDS, SCROLL_DIRECT_STEP,
    STATE_CURSOR, STATE_IDLE, STATE_KEYBOARD, STATE_LAUNCHER,
    STATE_SCROLL, STATE_TERMINATED,
    WEBCAM_HEIGHT, WEBCAM_WIDTH,
)
from context_engine import ContextEngine
from cursor_controller import CursorController
from fatigue_detector import FatigueDetector
from gesture_actions import dispatch_action, launch_app, match_app
from hand_tracker import FrameAnalysis, HandTracker
from keyboard_overlay import KeyboardOverlay
from voice_feedback import speak

logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
LOGGER = logging.getLogger(__name__)

# ── Colors (BGR) ──────────────────────────────────────────────────────────────
C_YELLOW = (0, 255, 255)
C_GREEN  = (0, 255, 0)
C_WHITE  = (255, 255, 255)
C_BLACK  = (0, 0, 0)
C_CYAN   = (255, 255, 0)
C_ORANGE = (0, 165, 255)
C_RED    = (0, 0, 255)
C_PURPLE = (200, 50, 200)

STATE_COLORS = {
    STATE_CURSOR:     (0, 200, 80),
    STATE_SCROLL:     (255, 140, 0),
    STATE_KEYBOARD:   (200, 50, 200),
    STATE_LAUNCHER:   (50, 200, 200),
    STATE_TERMINATED: (0, 0, 200),
    STATE_IDLE:       (100, 100, 100),
}

# Frames a gesture must be held to trigger a state switch
HOLD_FRAMES = 28
# Frames both fists must be held to exit
DUAL_FIST_FRAMES = 30
# Frames thumbs_up must be held to shut down (~2s at 30fps)
SHUTDOWN_HOLD_FRAMES = 55

# Gesture debounce — frames a gesture must be stable before firing an action
GESTURE_DEBOUNCE = 12
GESTURE_COOLDOWN = 25  # frames before same action fires again

# Gesture name constants (kinesisv3 style — plain strings)
G_POINTING  = "pointing"
G_PINCH     = "pinch"
G_PEACE     = "peace"
G_OPEN_HAND = "open_hand"
G_FIST      = "fist"
G_ROCK_ON   = "rock_on"
G_THUMBS_UP = "thumbs_up"
G_TRACKING  = "tracking"


# ── HUD helpers ───────────────────────────────────────────────────────────────

def _draw_hud(frame, state: str, context: str, gesture: str,
              last_action: str, fatigue_level: float) -> None:
    h, w = frame.shape[:2]
    panel = frame.copy()
    cv2.rectangle(panel, (0, 0), (340, 140), (20, 20, 20), -1)
    cv2.addWeighted(panel, 0.65, frame, 0.35, 0, frame)

    color = STATE_COLORS.get(state, C_WHITE)
    cv2.putText(frame, f"State: {state}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    cv2.putText(frame, f"Context: {context.upper()}", (10, 54),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, C_YELLOW, 1)
    cv2.putText(frame, f"Gesture: {gesture}", (10, 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, C_GREEN, 1)
    if last_action:
        cv2.putText(frame, f"> {last_action}", (10, 102),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, C_CYAN, 1)

    # State pill top-right
    label = f" {state} "
    sz = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0]
    px = w - sz[0] - 18
    cv2.rectangle(frame, (px - 4, 8), (w - 8, 34), color, -1, cv2.LINE_AA)
    cv2.putText(frame, label, (px, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_BLACK, 2)

    # Fatigue bar
    if fatigue_level > 0.1:
        bar_w = int(w * 0.22)
        bx, by = w - bar_w - 10, 44
        cv2.rectangle(frame, (bx, by), (bx + bar_w, by + 10), (60, 60, 60), -1)
        cv2.rectangle(frame, (bx, by), (bx + int(bar_w * fatigue_level), by + 10),
                      HUD_WARNING_COLOR, -1)
        cv2.putText(frame, "fatigue", (bx, by - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, HUD_WARNING_COLOR, 1)


def _draw_hold_bar(frame, progress: float, label: str, color) -> None:
    h, w = frame.shape[:2]
    bar_w = int(w * 0.35)
    bx, by = w // 2 - bar_w // 2, h - 60
    cv2.rectangle(frame, (bx, by), (bx + bar_w, by + 16), (60, 60, 60), -1)
    fill = int(bar_w * min(progress, 1.0))
    if fill > 0:
        cv2.rectangle(frame, (bx, by), (bx + fill, by + 16), color, -1)
    cv2.putText(frame, label, (bx, by - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


def _draw_help(frame, state: str) -> None:
    h, w = frame.shape[:2]
    hints = {
        STATE_CURSOR:   "point=move | pinch=click | open_hand=Start | peace(hold)=scroll | rock_on(hold)=keyboard | thumbs_up(hold 2s)=SHUTDOWN",
        STATE_SCROLL:   "peace=scroll up | fist=scroll down | open_hand=back to cursor",
        STATE_KEYBOARD: "point to key + pinch = type | fist = close keyboard",
        STATE_LAUNCHER: "ISL letters to spell app | thumbs_up=launch | fist=cancel",
    }
    hint = hints.get(state, "")
    if not hint:
        return
    ov = frame.copy()
    cv2.rectangle(ov, (0, h - 22), (w, h), C_BLACK, -1)
    cv2.addWeighted(ov, 0.7, frame, 0.3, 0, frame)
    cv2.putText(frame, hint, (8, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.34, C_WHITE, 1)


# ── Main app ──────────────────────────────────────────────────────────────────

class KinesysApp:
    def __init__(self) -> None:
        self._tracker  = HandTracker()
        self._cursor   = CursorController()
        self._ctx_eng  = ContextEngine()
        self._fatigue  = FatigueDetector()
        self._keyboard = KeyboardOverlay()

        # Auto-start in CURSOR — no gesture needed to begin
        self._state      = STATE_CURSOR
        self._last_action = ""
        self._context    = "default"

        # Hold-to-switch tracking
        self._hold_gesture = ""
        self._hold_count   = 0

        # Dual-fist exit
        self._shutdown_hold = 0

        # Gesture debounce
        self._prev_gesture  = G_TRACKING
        self._deb_count     = 0
        self._cool_count    = 0

        # Scroll
        self._last_scroll_t = 0.0

        # Launcher
        self._launcher_word    = ""
        self._launcher_matched = ""
        self._launcher_cool    = 0

    def run(self) -> None:
        cap = cv2.VideoCapture(CAMERA_ID)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WEBCAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_HEIGHT)

        if not cap.isOpened():
            LOGGER.error("Cannot open camera %d", CAMERA_ID)
            sys.exit(1)

        speak(f"{APP_NAME} ready")
        LOGGER.info("%s started — auto-running in CURSOR mode", APP_NAME)

        frame_n = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    continue
                frame = cv2.flip(frame, FRAME_FLIP_CODE)
                h, w  = frame.shape[:2]
                frame_n += 1

                # Context detection every 30 frames
                if frame_n % 30 == 0:
                    snap = self._ctx_eng.get_context()
                    self._context = snap.profile_name

                analysis: FrameAnalysis = self._tracker.process(frame)
                self._tracker.draw_annotations(frame)

                # Fatigue
                lm_norm = analysis.action_hand.landmarks_norm if analysis.action_hand else None
                fat = self._fatigue.update(lm_norm)
                if fat.should_alert:
                    speak("Take a break, hand fatigue detected")

                gesture = analysis.action_gesture

                # ── Shutdown: thumbs_up held ~2s from any state ───────────────
                if gesture == G_THUMBS_UP and self._state != STATE_KEYBOARD:
                    self._shutdown_hold += 1
                    if self._shutdown_hold >= SHUTDOWN_HOLD_FRAMES:
                        _draw_hold_bar(frame, 1.0, "SHUTTING DOWN...", C_RED)
                        cv2.imshow(MAIN_WINDOW_NAME, frame)
                        cv2.waitKey(800)
                        self._state = STATE_TERMINATED
                    else:
                        _draw_hold_bar(frame, self._shutdown_hold / SHUTDOWN_HOLD_FRAMES,
                                       "Hold thumbs_up to SHUT DOWN", C_RED)
                else:
                    self._shutdown_hold = 0

                if self._state != STATE_TERMINATED:
                    self._tick(frame, analysis, gesture, fat.smoothing_alpha, w, h)

                _draw_hud(frame, self._state, self._context, gesture,
                          self._last_action, fat.fatigue_level)
                _draw_help(frame, self._state)

                cv2.imshow(MAIN_WINDOW_NAME, frame)
                key = cv2.waitKey(FRAME_WAIT_KEY_MS) & 0xFF
                if key == EXIT_KEY or self._state == STATE_TERMINATED:
                    speak("Goodbye")
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self._tracker.close()

    # ── Per-frame tick ────────────────────────────────────────────────────────

    def _tick(self, frame, analysis: FrameAnalysis, gesture: str,
              smoothing_alpha: float, w: int, h: int) -> None:

        # ── Fist = instant back to CURSOR from non-cursor states ──────────────
        if gesture == G_FIST and self._state in (STATE_KEYBOARD, STATE_LAUNCHER):
            self._switch(STATE_CURSOR)
            return

        # ── Hold-to-switch gestures ───────────────────────────────────────────
        # peace(hold) → SCROLL | rock_on(hold) → KEYBOARD | thumbs_up(hold) → LAUNCHER
        HOLD_MAP = {
            G_PEACE:   STATE_SCROLL,
            G_ROCK_ON: STATE_KEYBOARD,
        }
        target = HOLD_MAP.get(gesture)
        if target and target != self._state and self._state == STATE_CURSOR:
            if gesture == self._hold_gesture:
                self._hold_count += 1
            else:
                self._hold_gesture = gesture
                self._hold_count   = 1
            _draw_hold_bar(frame, self._hold_count / HOLD_FRAMES,
                           f"→ {target}", STATE_COLORS.get(target, C_WHITE))
            if self._hold_count >= HOLD_FRAMES:
                self._switch(target)
            return
        else:
            if gesture != self._hold_gesture:
                self._hold_gesture = ""
                self._hold_count   = 0

        # ── Route to active state ─────────────────────────────────────────────
        if self._state == STATE_CURSOR:
            self._cursor_tick(analysis, gesture, smoothing_alpha, w, h)
        elif self._state == STATE_SCROLL:
            self._scroll_tick(gesture)
        elif self._state == STATE_KEYBOARD:
            self._keyboard_tick(frame, analysis)
        elif self._state == STATE_LAUNCHER:
            self._launcher_tick(frame, analysis, gesture)

    # ── CURSOR ────────────────────────────────────────────────────────────────

    def _cursor_tick(self, analysis: FrameAnalysis, gesture: str,
                     smoothing_alpha: float, w: int, h: int) -> None:
        if not analysis.action_hand:
            return

        tip = analysis.action_hand.landmarks_px[INDEX_FINGER_TIP_ID]
        self._cursor.move_cursor(tip, (w, h), smoothing_alpha)

        # Pinch = left click (debounced inside cursor_controller)
        if gesture == G_PINCH:
            if self._cursor.click():
                self._last_action = "left click"
            return

        # Open hand = Windows Start menu (debounced via cooldown)
        if gesture == G_OPEN_HAND:
            if self._cool_count == 0:
                pyautogui.hotkey("win")
                self._last_action = "Start menu"
                self._cool_count  = GESTURE_COOLDOWN
            return

        # Pointing = just move, no action needed
        if gesture == G_POINTING:
            return

        # peace → context action (debounced); thumbs_up is reserved for shutdown only
        if gesture == G_PEACE:
            self._dispatch_debounced(gesture)

        if self._cool_count > 0:
            self._cool_count -= 1

    def _dispatch_debounced(self, gesture: str) -> None:
        if gesture == self._prev_gesture:
            self._deb_count += 1
        else:
            self._deb_count = 0
        self._prev_gesture = gesture

        if self._deb_count >= GESTURE_DEBOUNCE and self._cool_count == 0:
            action = dispatch_action(gesture, self._context, self._cursor)
            if action:
                self._last_action = action
                self._cool_count  = GESTURE_COOLDOWN
                self._deb_count   = 0

    # ── SCROLL ────────────────────────────────────────────────────────────────

    def _scroll_tick(self, gesture: str) -> None:
        now = time.time()
        if now - self._last_scroll_t < SCROLL_COOLDOWN_SECONDS:
            return
        self._last_scroll_t = now

        if gesture == G_PEACE:
            pyautogui.scroll(SCROLL_DIRECT_STEP)
            self._last_action = "scroll up"
        elif gesture == G_FIST:
            pyautogui.scroll(-SCROLL_DIRECT_STEP)
            self._last_action = "scroll down"
        elif gesture == G_OPEN_HAND:
            self._switch(STATE_CURSOR)

    # ── KEYBOARD ──────────────────────────────────────────────────────────────

    def _keyboard_tick(self, frame, analysis: FrameAnalysis) -> None:
        """On-screen keyboard. Pinch selects a key and types into the focused field."""
        if not analysis.action_hand:
            h, w = frame.shape[:2]
            cv2.putText(frame, "Show hand to type",
                        (w // 2 - 110, frame.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, C_ORANGE, 2)
            return

        tip = analysis.action_hand.landmarks_px[INDEX_FINGER_TIP_ID]
        is_pinching = analysis.action_hand.pinch_distance < PINCH_THRESHOLD

        key = self._keyboard.draw(frame, tip, is_pinching)
        if key:
            self._last_action = f"typed: {key}"
            # pyautogui already called inside keyboard_overlay._handle_key
            speak(key)

    # ── LAUNCHER ──────────────────────────────────────────────────────────────

    def _launcher_tick(self, frame, analysis: FrameAnalysis, gesture: str) -> None:
        h, w = frame.shape[:2]

        # Cooldown between ISL letters
        if self._launcher_cool > 0:
            self._launcher_cool -= 1

        # Recognise ISL letter from hand landmarks
        if analysis.action_hand and self._launcher_cool == 0:
            letter = _isl_letter(analysis.action_hand.landmarks_px,
                                 analysis.action_hand.handedness)
            if letter:
                self._launcher_word    += letter
                self._launcher_cool     = 30
                app_key, _             = match_app(self._launcher_word)
                self._launcher_matched  = app_key or ""

        # thumbs_up = launch
        if gesture == G_THUMBS_UP and self._launcher_word:
            app_key, cmd = match_app(self._launcher_word)
            if cmd:
                launch_app(cmd)
                self._last_action = f"launched {app_key}"
                speak(f"Launching {app_key}")
            else:
                self._last_action = f"no match: {self._launcher_word}"
            self._launcher_word = ""
            self._switch(STATE_CURSOR)
            return

        # peace = backspace
        if gesture == G_PEACE and self._launcher_cool == 0:
            self._launcher_word    = self._launcher_word[:-1]
            app_key, _             = match_app(self._launcher_word)
            self._launcher_matched = app_key or ""
            self._launcher_cool    = 20

        # HUD
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (w, 120), (20, 10, 30), -1)
        cv2.addWeighted(ov, 0.75, frame, 0.25, 0, frame)
        cv2.putText(frame, "LAUNCHER — spell app name",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_PURPLE, 2)
        cv2.putText(frame, (self._launcher_word or "") + "_",
                    (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.6, C_WHITE, 3)
        if self._launcher_matched:
            cv2.putText(frame, f"-> {self._launcher_matched}",
                        (10, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_GREEN, 2)

    # ── State switch ──────────────────────────────────────────────────────────

    def _switch(self, new_state: str) -> None:
        if new_state == self._state:
            return
        LOGGER.info("State: %s → %s", self._state, new_state)
        speak(new_state.lower())
        self._state        = new_state
        self._hold_count   = 0
        self._hold_gesture = ""
        self._deb_count    = 0
        self._cool_count   = 0
        if new_state == STATE_KEYBOARD:
            self._keyboard.typed_word = ""
        if new_state == STATE_LAUNCHER:
            self._launcher_word    = ""
            self._launcher_matched = ""
            self._launcher_cool    = 0


# ── ISL letter recogniser (from kinesisv3) ────────────────────────────────────

def _dist(a, b) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _isl_letter(lm: list, hand_label: str) -> str | None:
    """Recognise ISL static hand-sign letters from pixel landmarks (kinesisv3 logic)."""
    if hand_label == "Right":
        th_ext = lm[4][0] < lm[2][0]
    else:
        th_ext = lm[4][0] > lm[2][0]

    ix_ext = lm[8][1]  < lm[6][1]
    mi_ext = lm[12][1] < lm[10][1]
    ri_ext = lm[16][1] < lm[14][1]
    pi_ext = lm[20][1] < lm[18][1]

    hand_size = _dist(lm[0], lm[9]) + 1e-6
    ti_close  = _dist(lm[4], lm[8])  / hand_size < 0.25
    tm_close  = _dist(lm[4], lm[12]) / hand_size < 0.25
    im_close  = _dist(lm[8], lm[12]) / hand_size < 0.15

    if not ix_ext and not mi_ext and not ri_ext and not pi_ext and th_ext:
        return "A"
    if ix_ext and mi_ext and ri_ext and pi_ext and not th_ext:
        return "B"
    if not ix_ext and not mi_ext and not ri_ext and not pi_ext and not th_ext:
        avg_tip_y = (lm[8][1] + lm[12][1] + lm[16][1]) / 3
        return "M" if avg_tip_y > lm[0][1] * 0.85 else "E"
    if ti_close and mi_ext and ri_ext and pi_ext:
        return "F"
    if ti_close and tm_close and not ix_ext and not mi_ext:
        return "O"
    if ix_ext and mi_ext and not ri_ext and not pi_ext and im_close:
        return "U"
    spread = _dist(lm[8], lm[12]) / hand_size > 0.18
    if ix_ext and mi_ext and spread and not ri_ext and not pi_ext:
        return "V"
    if ix_ext and mi_ext and ri_ext and not pi_ext:
        return "W"
    if ix_ext and th_ext and not mi_ext and not ri_ext and not pi_ext:
        return "L"
    if th_ext and pi_ext and not ix_ext and not mi_ext and not ri_ext:
        return "Y"
    return None


def main() -> None:
    app = KinesysApp()
    app.run()


if __name__ == "__main__":
    main()
