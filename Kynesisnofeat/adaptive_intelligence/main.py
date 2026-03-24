import cv2
import numpy as np
import sys
import os
import time
import pyautogui

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from foundation.cursor_controller import CursorController
from foundation.voice_feedback import speak
from context_awareness.context_engine import ContextEngine
from gesture_engine.hand_tracker import HandTracker
from gesture_engine.gesture_recognizer import classify_gesture, dispatch_action, match_app, launch_app
from adaptive_intelligence.fatigue_detector import FatigueDetector
from adaptive_intelligence.gesture_trainer import load_or_train_model, predict_character
import adaptive_intelligence.config as config

# ── Modes ─────────────────────────────────────────────────────────────────────
MODE_CONTROL  = "CONTROL"
MODE_CURSOR   = "CURSOR"
MODE_WRITING  = "WRITING"
MODE_LAUNCHER = "LAUNCHER"

# ── Colours (BGR) ─────────────────────────────────────────────────────────────
C_YELLOW = (0, 255, 255)
C_GREEN  = (0, 255, 0)
C_WHITE  = (255, 255, 255)
C_BLACK  = (0, 0, 0)
C_CYAN   = (255, 255, 0)
C_PURPLE = (200, 50, 200)
C_ORANGE = (0, 165, 255)
C_RED    = (0, 0, 255)

MODE_COLORS = {
    MODE_CONTROL:  C_GREEN,
    MODE_CURSOR:   (255, 100, 50),
    MODE_WRITING:  C_ORANGE,
    MODE_LAUNCHER: C_PURPLE,
}

# ── Tuning ────────────────────────────────────────────────────────────────────
GESTURE_DEBOUNCE  = 20   # frames gesture must be held before firing
GESTURE_COOLDOWN  = 30   # frames before same gesture can fire again
MODE_HOLD_FRAMES  = 28   # frames to hold trigger gesture to switch mode
CURSOR_ALPHA      = 0.35 # smoothing factor for cursor movement
CURSOR_MARGIN     = 0.15 # dead-zone margin on camera edges


# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def draw_mode_pill(frame, mode: str):
    h, w = frame.shape[:2]
    color = MODE_COLORS.get(mode, C_WHITE)
    label = f" {mode} "
    sz = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0]
    px = w - sz[0] - 18
    cv2.rectangle(frame, (px - 4, 8), (w - 8, 34), color, -1, cv2.LINE_AA)
    cv2.putText(frame, label, (px, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_BLACK, 2)


def draw_trigger_bar(frame, progress: float, label: str, color):
    h, w = frame.shape[:2]
    bar_w = int(w * 0.35)
    bx, by = w // 2 - bar_w // 2, h - 60
    cv2.rectangle(frame, (bx, by), (bx + bar_w, by + 16), (60, 60, 60), -1)
    fill = int(bar_w * min(progress, 1.0))
    if fill > 0:
        cv2.rectangle(frame, (bx, by), (bx + fill, by + 16), color, -1)
    cv2.putText(frame, label, (bx, by - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


def draw_hud(frame, mode, context, gesture, last_action):
    cv2.putText(frame, f"Context: {context.upper()}", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, C_YELLOW, 2)
    cv2.putText(frame, f"Gesture: {gesture}", (10, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, C_GREEN, 2)
    if last_action:
        cv2.putText(frame, f"Action: {last_action}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, C_CYAN, 2)
    draw_mode_pill(frame, mode)


def draw_bottom_strip(frame, lines: list):
    h, w = frame.shape[:2]
    strip_h = 22 * len(lines) + 10
    ov = frame.copy()
    cv2.rectangle(ov, (0, h - strip_h), (w, h), C_BLACK, -1)
    cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (8, h - strip_h + 18 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_WHITE, 1)


def draw_launcher_hud(frame, word_buffer: str, matched_app: str):
    h, w = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, 130), (20, 10, 30), -1)
    cv2.addWeighted(ov, 0.75, frame, 0.25, 0, frame)
    draw_mode_pill(frame, MODE_LAUNCHER)
    cv2.putText(frame, "LAUNCHER — spell app name with gestures",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_PURPLE, 2)
    display = (word_buffer + "_") if word_buffer is not None else "_"
    cv2.putText(frame, display, (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6, C_WHITE, 3)
    if matched_app:
        cv2.putText(frame, f"-> {matched_app}", (10, 118),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, C_GREEN, 2)
    draw_bottom_strip(frame, [
        "THUMBS_UP=confirm+launch  |  PEACE=backspace  |  OPEN_HAND=cancel"
    ])


def flash_overlay(frame, text: str, color, alpha=0.3):
    h, w = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, h), color, -1)
    cv2.addWeighted(ov, alpha, frame, 1 - alpha, 0, frame)
    sz = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
    cv2.putText(frame, text, ((w - sz[0]) // 2, (h + sz[1]) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, C_WHITE, 3)



# ═════════════════════════════════════════════════════════════════════════════
# ISL LETTER RECOGNIZER  (static hand signs A-Z for launcher/typing)
# ═════════════════════════════════════════════════════════════════════════════

def _dist(a, b) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

def recognize_isl_letter(landmarks, hand_label: str):
    """
    Recognise ISL static hand-sign letters from pixel landmarks.
    Returns uppercase letter or None.
    """
    lm = landmarks

    if hand_label == "Right":
        th_ext = lm[4][0] < lm[2][0]
    else:
        th_ext = lm[4][0] > lm[2][0]

    ix_ext = lm[8][1]  < lm[6][1]
    mi_ext = lm[12][1] < lm[10][1]
    ri_ext = lm[16][1] < lm[14][1]
    pi_ext = lm[20][1] < lm[18][1]

    ix_curl = not ix_ext
    mi_curl = not mi_ext
    ri_curl = not ri_ext
    pi_curl = not pi_ext
    th_curl = not th_ext

    hand_size = _dist(lm[0], lm[9]) + 1e-6
    ti_close  = _dist(lm[4], lm[8])  / hand_size < 0.25
    tm_close  = _dist(lm[4], lm[12]) / hand_size < 0.25
    im_close  = _dist(lm[8], lm[12]) / hand_size < 0.15

    if ix_curl and mi_curl and ri_curl and pi_curl and th_ext:
        return "A"
    if ix_ext and mi_ext and ri_ext and pi_ext and th_curl:
        return "B"
    if ix_curl and mi_curl and ri_curl and pi_curl and th_curl:
        avg_tip_y = (lm[8][1] + lm[12][1] + lm[16][1]) / 3
        return "M" if avg_tip_y > lm[0][1] * 0.85 else "E"
    if ix_curl and mi_curl and ri_curl and pi_curl and not th_curl:
        return "N"
    if ix_curl and mi_curl and ri_curl and pi_curl and ti_close:
        return "S"
    if ti_close and mi_ext and ri_ext and pi_ext:
        return "F"
    if ti_close and tm_close and not ix_ext and not mi_ext:
        return "O"
    if ix_ext and mi_ext and im_close and ri_curl and pi_curl:
        return "U"
    spread = _dist(lm[8], lm[12]) / hand_size > 0.18
    if ix_ext and mi_ext and spread and ri_curl and pi_curl:
        return "V"
    if ix_ext and mi_ext and ri_ext and pi_curl:
        return "W"
    if ix_ext and th_ext and mi_curl and ri_curl and pi_curl:
        return "L"
    if th_ext and pi_ext and ix_curl and mi_curl and ri_curl:
        return "Y"
    if ix_ext and not mi_ext and not ri_ext and pi_ext:
        return "R"  # rock_on shape — reused as R in ISL mode
    return None



# ═════════════════════════════════════════════════════════════════════════════
# CURSOR ENGINE
# ═════════════════════════════════════════════════════════════════════════════

class CursorEngine:
    def __init__(self, cam_w: int, cam_h: int):
        scr_w, scr_h  = pyautogui.size()
        self.scr_w    = scr_w
        self.scr_h    = scr_h
        self.sx       = float(scr_w // 2)
        self.sy       = float(scr_h // 2)
        self.click_cd = 0

    def update(self, norm_x: float, norm_y: float):
        nx = (norm_x - CURSOR_MARGIN) / (1 - 2 * CURSOR_MARGIN)
        ny = (norm_y - CURSOR_MARGIN) / (1 - 2 * CURSOR_MARGIN)
        nx = max(0.0, min(1.0, nx))
        ny = max(0.0, min(1.0, ny))
        self.sx = CURSOR_ALPHA * (nx * self.scr_w) + (1 - CURSOR_ALPHA) * self.sx
        self.sy = CURSOR_ALPHA * (ny * self.scr_h) + (1 - CURSOR_ALPHA) * self.sy
        try:
            pyautogui.moveTo(int(self.sx), int(self.sy))
        except Exception:
            pass
        if self.click_cd > 0:
            self.click_cd -= 1

    def click(self):
        if self.click_cd == 0:
            try:
                pyautogui.click()
            except Exception:
                pass
            self.click_cd = 20

    def scroll(self, direction: int):
        try:
            pyautogui.scroll(direction)
        except Exception:
            pass



# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("Initializing Kinesis V3...")
    model       = load_or_train_model()
    tracker     = HandTracker(max_hands=2)
    context_eng = ContextEngine()
    fatigue     = FatigueDetector()

    cap = cv2.VideoCapture(config.CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        sys.exit(1)

    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cursor_engine = CursorEngine(cam_w, cam_h)

    # ── Air-writing state ──
    canvas         = np.zeros((config.CANVAS_SIZE, config.CANVAS_SIZE), dtype=np.uint8)
    drawing_mode   = False
    prev_draw_pos  = None
    prev_scroll_y  = 0
    typed_word     = ""   # accumulates letters; spoken on space/enter
    backspace_cool = 0    # frames cooldown so one pinch = one backspace

    # ── Mode & gesture state ──
    mode         = MODE_CONTROL
    context      = "default"
    gesture      = "tracking"
    last_action  = ""
    deb_count    = 0
    cool_count   = 0
    prev_gesture = "tracking"

    # Mode-switch trigger counters
    trig_cursor   = 0
    trig_launcher = 0
    trig_writing  = 0
    trig_control  = 0
    trig_quit     = 0
    QUIT_FRAMES   = 35   # ~1.2s both fists held to quit

    # Launcher state
    launcher_buffer  = ""
    launcher_matched = ""
    launcher_cool    = 0
    LAUNCHER_LETTER_COOL = 30   # frames between ISL letters

    # Flash state
    flash_text   = ""
    flash_color  = C_GREEN
    flash_frames = 0

    speak("Kinesis Ready")
    print("\n[MODES]  CONTROL | CURSOR | WRITING | LAUNCHER")
    print("[SWITCH] In CONTROL: rock_on=Launcher | peace(hold)=Cursor | thumbs_up(hold)=Writing")
    print("[EXIT]   open_hand (hold) from any mode → back to CONTROL")
    print("[QUIT]   both fists simultaneously (hold ~1.2s) → close app\n")

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]
        frame_count += 1

        # Detect context every 20 frames
        if frame_count % 20 == 0:
            context = context_eng.detect_context()

        hands_info, result = tracker.process(frame)
        tracker.draw_hands(frame, result)

        if fatigue.check_fatigue():
            cv2.putText(frame, "FATIGUE WARNING: Take a break!",
                        (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, C_RED, 2)

        # Identify hands (frame is flipped, so MP Left = user Right)
        left_hand  = None
        right_hand = None
        for hnd in hands_info:
            if hnd["label"] == "Left":
                right_hand = hnd
            else:
                left_hand = hnd

        # Primary hand for gesture control
        primary = right_hand or left_hand
        gesture = "tracking"
        if primary:
            lm_label = "Right" if primary == right_hand else "Left"
            gesture  = classify_gesture(primary["landmarks"], lm_label)

        # ── QUIT: both fists held simultaneously ──────────────────────
        both_fists = False
        if left_hand and right_hand:
            g_left  = classify_gesture(left_hand["landmarks"],  "Left")
            g_right = classify_gesture(right_hand["landmarks"], "Right")
            both_fists = (g_left == "fist" and g_right == "fist")

        if both_fists:
            trig_quit += 1
            progress = trig_quit / QUIT_FRAMES
            draw_trigger_bar(frame, progress, "QUIT — hold both fists", C_RED)
            # Show countdown on frame
            cv2.putText(frame, f"Quitting in {int((1 - progress) * 1.2 + 0.5)}s...",
                        (w // 2 - 100, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, C_RED, 3)
            if trig_quit >= QUIT_FRAMES:
                speak("Goodbye")
                break
        else:
            trig_quit = 0

        # Flash overlay
        if flash_frames > 0:
            flash_overlay(frame, flash_text, flash_color)
            flash_frames -= 1

        # ══════════════════════════════════════════════════════════════
        #  MODE: CONTROL
        # ══════════════════════════════════════════════════════════════
        if mode == MODE_CONTROL:

            # ── Mode-switch triggers ──
            if gesture == "rock_on":
                trig_launcher += 1
                draw_trigger_bar(frame, trig_launcher / MODE_HOLD_FRAMES,
                                 "-> LAUNCHER", C_PURPLE)
                if trig_launcher >= MODE_HOLD_FRAMES:
                    mode = MODE_LAUNCHER
                    launcher_buffer  = ""
                    launcher_matched = ""
                    trig_launcher    = 0
                    speak("Launcher mode")
            else:
                trig_launcher = 0

            if gesture == "peace":
                trig_cursor += 1
                draw_trigger_bar(frame, trig_cursor / MODE_HOLD_FRAMES,
                                 "-> CURSOR", MODE_COLORS[MODE_CURSOR])
                if trig_cursor >= MODE_HOLD_FRAMES:
                    mode = MODE_CURSOR
                    trig_cursor = 0
                    speak("Cursor mode")
            else:
                trig_cursor = 0

            if gesture == "thumbs_up":
                trig_writing += 1
                draw_trigger_bar(frame, trig_writing / MODE_HOLD_FRAMES,
                                 "-> WRITING", C_ORANGE)
                if trig_writing >= MODE_HOLD_FRAMES:
                    mode = MODE_WRITING
                    canvas.fill(0)
                    prev_draw_pos = None
                    drawing_mode  = False
                    trig_writing  = 0
                    speak("Writing mode")
            else:
                trig_writing = 0

            # ── Gesture debounce + cooldown → dispatch ──
            if cool_count > 0:
                cool_count -= 1

            if gesture == prev_gesture and gesture not in ("tracking", "rock_on", "peace", "thumbs_up"):
                deb_count += 1
            else:
                deb_count = 0

            if deb_count >= GESTURE_DEBOUNCE and cool_count == 0:
                action = dispatch_action(gesture, context)
                if action:
                    last_action = action
                    cool_count  = GESTURE_COOLDOWN
                    deb_count   = 0

            prev_gesture = gesture

            draw_hud(frame, mode, context, gesture, last_action)
            draw_bottom_strip(frame, [
                "rock_on(hold)=Launcher | peace(hold)=Cursor | thumbs_up(hold)=Writing",
                "pointing=click | peace=new tab | fist=undo | open_hand=start menu",
            ])

        # ══════════════════════════════════════════════════════════════
        #  MODE: CURSOR
        # ══════════════════════════════════════════════════════════════
        elif mode == MODE_CURSOR:

            if primary:
                lm = primary["landmarks"]
                # Normalise index tip to 0-1
                norm_x = lm[8][0] / w
                norm_y = lm[8][1] / h
                cursor_engine.update(norm_x, norm_y)

                # Draw cursor dot on frame
                cv2.circle(frame, lm[8], 12, C_GREEN, 2, cv2.LINE_AA)
                cv2.circle(frame, lm[8], 4,  C_GREEN, -1, cv2.LINE_AA)

                # Pinch = click
                import math
                pinch_dist = math.hypot(lm[4][0] - lm[8][0], lm[4][1] - lm[8][1])
                if pinch_dist < 30:
                    cursor_engine.click()
                    cv2.circle(frame, lm[8], 18, C_RED, 2, cv2.LINE_AA)

                # Scroll: peace = up, fist = down
                if gesture == "peace":
                    cursor_engine.scroll(3)
                elif gesture == "fist":
                    cursor_engine.scroll(-3)

            # Exit cursor mode
            if gesture == "open_hand":
                trig_control += 1
                draw_trigger_bar(frame, trig_control / MODE_HOLD_FRAMES,
                                 "-> CONTROL", C_GREEN)
                if trig_control >= MODE_HOLD_FRAMES:
                    mode = MODE_CONTROL
                    trig_control = 0
                    speak("Control mode")
            else:
                trig_control = 0

            draw_mode_pill(frame, mode)
            cv2.putText(frame, "CURSOR — point to move  |  pinch to click",
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, C_YELLOW, 2)
            draw_bottom_strip(frame, [
                "peace=scroll up | fist=scroll down | open_hand(hold)=back to CONTROL"
            ])

        # ══════════════════════════════════════════════════════════════
        #  MODE: WRITING  (two-hand air writing → CNN → type)
        # ══════════════════════════════════════════════════════════════
        elif mode == MODE_WRITING:

            if backspace_cool > 0:
                backspace_cool -= 1

            if left_hand and right_hand:
                left_pinching  = tracker.is_pinching(left_hand,  threshold=40)
                right_pinching = tracker.is_pinching(right_hand, threshold=40)

                # ── Right pinch = backspace (when not drawing) ──
                if right_pinching and not left_pinching and not drawing_mode:
                    if backspace_cool == 0:
                        pyautogui.press("backspace")
                        if typed_word:
                            typed_word = typed_word[:-1]
                        speak("backspace")
                        flash_text    = "⌫"
                        flash_color   = C_RED
                        flash_frames  = 10
                        backspace_cool = 25

                # ── Left pinch + right index = draw ──
                elif left_pinching:
                    if not drawing_mode:
                        drawing_mode  = True
                        canvas.fill(0)
                        prev_draw_pos = None

                    idx_pos = tracker.get_draw_canvas_position(right_hand)
                    bx = w // 2 - 200; by = h // 2 - 200; bw = 400; bh = 400
                    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), C_ORANGE, 2)

                    cx = int((idx_pos[0] - bx) * (config.CANVAS_SIZE / bw))
                    cy = int((idx_pos[1] - by) * (config.CANVAS_SIZE / bh))

                    if 0 <= cx < config.CANVAS_SIZE and 0 <= cy < config.CANVAS_SIZE:
                        if prev_draw_pos is not None:
                            cv2.line(canvas, prev_draw_pos, (cx, cy), 255, config.DRAW_THICKNESS)
                        prev_draw_pos = (cx, cy)

                else:
                    # Left pinch released → predict & type
                    if drawing_mode:
                        drawing_mode = False
                        if np.max(canvas) > 0:
                            resized = cv2.resize(canvas, (28, 28), interpolation=cv2.INTER_AREA)
                            char = predict_character(model, resized)
                            print(f"[WRITING] Predicted: {char}")
                            pyautogui.write(char)
                            speak(char)           # say the letter
                            typed_word   += char
                            flash_text    = char
                            flash_color   = C_ORANGE
                            flash_frames  = 15
                        canvas.fill(0)

            # ── Right hand only: scroll or space/enter ──
            elif right_hand and not drawing_mode:
                if tracker.is_two_fingers_up(right_hand):
                    avg_y = (right_hand["landmarks"][8][1] + right_hand["landmarks"][12][1]) / 2
                    if prev_scroll_y != 0:
                        dy = avg_y - prev_scroll_y
                        if abs(dy) > 3:
                            pyautogui.scroll(int(-dy * config.SCROLL_SPEED))
                    prev_scroll_y = avg_y
                else:
                    prev_scroll_y = 0

                # Thumbs up (right only) = space → speak word
                g_right = classify_gesture(right_hand["landmarks"], "Right")
                if g_right == "thumbs_up" and backspace_cool == 0:
                    pyautogui.press("space")
                    if typed_word:
                        speak(typed_word)
                        typed_word = ""
                    backspace_cool = 30

                # Peace (right only) = enter → speak word
                elif g_right == "peace" and backspace_cool == 0:
                    pyautogui.press("enter")
                    if typed_word:
                        speak(typed_word)
                        typed_word = ""
                    backspace_cool = 30
            else:
                prev_scroll_y = 0

            # Exit writing mode
            if gesture == "open_hand" and not drawing_mode:
                trig_control += 1
                draw_trigger_bar(frame, trig_control / MODE_HOLD_FRAMES,
                                 "-> CONTROL", C_GREEN)
                if trig_control >= MODE_HOLD_FRAMES:
                    mode = MODE_CONTROL
                    trig_control = 0
                    canvas.fill(0)
                    typed_word = ""
                    speak("Control mode")
            else:
                trig_control = 0

            # Mini canvas preview (top-right)
            canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
            frame[10:10 + config.CANVAS_SIZE, w - config.CANVAS_SIZE - 10:w - 10] = canvas_bgr

            draw_mode_pill(frame, mode)
            cv2.putText(frame, f"Context: {context.upper()}", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, C_YELLOW, 2)
            cv2.putText(frame, f"State: {'DRAWING' if drawing_mode else 'READY'}",
                        (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        C_ORANGE if drawing_mode else C_WHITE, 2)
            # Show accumulated word
            display_word = typed_word[-24:] if len(typed_word) > 24 else typed_word
            cv2.putText(frame, f"Word: {display_word}_", (10, 102),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, C_CYAN, 2)
            draw_bottom_strip(frame, [
                "L-pinch+R-index=draw | R-pinch=backspace | R-thumbsup=space | R-peace=enter",
                "Two fingers (right)=scroll | open_hand(hold)=back to CONTROL",
            ])

        # ══════════════════════════════════════════════════════════════
        #  MODE: LAUNCHER  (ISL letters → spell app name → launch)
        # ══════════════════════════════════════════════════════════════
        elif mode == MODE_LAUNCHER:

            if launcher_cool > 0:
                launcher_cool -= 1

            if primary and launcher_cool == 0:
                lm_label = "Right" if primary == right_hand else "Left"
                letter = recognize_isl_letter(primary["landmarks"], lm_label)

                if letter and letter not in ("tracking",):
                    launcher_buffer  += letter
                    launcher_cool     = LAUNCHER_LETTER_COOL
                    app_key, _ = match_app(launcher_buffer)
                    launcher_matched  = app_key or ""
                    flash_text   = letter
                    flash_color  = C_PURPLE
                    flash_frames = 10

            # THUMBS_UP = confirm + launch
            if gesture == "thumbs_up":
                app_key, cmd = match_app(launcher_buffer)
                if cmd:
                    launch_app(cmd)
                    flash_text   = f"Launching {app_key}!"
                    flash_color  = C_GREEN
                    flash_frames = 30
                    speak(f"Launching {app_key}")
                else:
                    flash_text   = "No match"
                    flash_color  = C_RED
                    flash_frames = 20
                launcher_buffer  = ""
                launcher_matched = ""
                mode = MODE_CONTROL

            # PEACE = backspace
            elif gesture == "peace" and launcher_cool == 0:
                if launcher_buffer:
                    launcher_buffer = launcher_buffer[:-1]
                    app_key, _ = match_app(launcher_buffer)
                    launcher_matched = app_key or ""
                launcher_cool = LAUNCHER_LETTER_COOL

            # OPEN_HAND = cancel
            elif gesture == "open_hand":
                trig_control += 1
                draw_trigger_bar(frame, trig_control / MODE_HOLD_FRAMES,
                                 "-> CONTROL", C_GREEN)
                if trig_control >= MODE_HOLD_FRAMES:
                    mode = MODE_CONTROL
                    trig_control     = 0
                    launcher_buffer  = ""
                    launcher_matched = ""
                    speak("Control mode")
            else:
                trig_control = 0

            draw_launcher_hud(frame, launcher_buffer, launcher_matched)

        # ── Window ──
        cv2.imshow("Kinesis V3", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
