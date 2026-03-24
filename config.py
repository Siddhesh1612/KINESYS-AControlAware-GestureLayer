"""Unified configuration for KINESYS v4."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ── App ───────────────────────────────────────────────────────────────────────
APP_NAME = "KINESYS v4"
MAIN_WINDOW_NAME = "KINESYS v4"
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

# ── Camera ────────────────────────────────────────────────────────────────────
CAMERA_ID = 0
WEBCAM_WIDTH = 1280
WEBCAM_HEIGHT = 720
FRAME_FLIP_CODE = 1          # mirror horizontally
FRAME_WAIT_KEY_MS = 1

# ── MediaPipe ─────────────────────────────────────────────────────────────────
MAX_HANDS = 2
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.7
GESTURE_HISTORY_FRAMES = 12  # frames kept for motion analysis

# ── Cursor ────────────────────────────────────────────────────────────────────
SMOOTHING_ALPHA = 0.55
CURSOR_ALPHA_MIN = 0.28
CURSOR_ALPHA_MAX = 0.88
CURSOR_HISTORY_LENGTH = 4
CURSOR_DEADZONE_PIXELS = 5.0
CURSOR_FAST_MOVEMENT_PIXELS = 100.0
CURSOR_SLOW_MOVEMENT_PIXELS = 12.0
CURSOR_MAX_STEP_PIXELS = 260.0
CURSOR_PREDICTION_FACTOR = 0.18
CLICK_DEBOUNCE_MS = 200         # fast enough to feel responsive
PYAUTOGUI_FAILSAFE = False
PYAUTOGUI_PAUSE = 0.0
PYAUTOGUI_MINIMUM_DURATION = 0.0
PYAUTOGUI_MINIMUM_SLEEP = 0.0
PYAUTOGUI_MOVE_DURATION = 0.0

# ── Scroll ────────────────────────────────────────────────────────────────────
SCROLL_SPEED = 5
SCROLL_TRIGGER_THRESHOLD = 8.0
SCROLL_SMOOTHING_ALPHA = 0.5
SCROLL_MAX_STEP = 15
SCROLL_COOLDOWN_SECONDS = 0.05
SCROLL_DIRECT_STEP = 5

# ── Gesture thresholds ────────────────────────────────────────────────────────
PINCH_THRESHOLD = 50          # px — thumb-index distance (wider = easier to pinch)
ZOOM_THRESHOLD = 55
ZOOM_SPEED = 5
SWIPE_HORIZONTAL_THRESHOLD = 35.0
SWIPE_VERTICAL_THRESHOLD = 35.0
FOUR_FINGER_SWIPE_THRESHOLD = 30.0
PEACE_FINGER_SPREAD_THRESHOLD = 30.0
PEACE_SWIPE_MIN_HISTORY = 6

# ── Gesture confidence ────────────────────────────────────────────────────────
ACTION_CONFIDENCE_THRESHOLD = 0.75
PINCH_GESTURE_CONFIDENCE = 0.92
PEACE_SIGN_GESTURE_CONFIDENCE = 0.90
INDEX_GESTURE_CONFIDENCE = 0.88
THUMBS_UP_GESTURE_CONFIDENCE = 0.90
ROCK_ON_GESTURE_CONFIDENCE = 0.90
OPEN_PALM_GESTURE_CONFIDENCE = 0.88
CLOSED_FIST_GESTURE_CONFIDENCE = 0.88
RIGHT_CLICK_GESTURE_CONFIDENCE = 0.85
TWO_FINGER_SWIPE_GESTURE_CONFIDENCE = 0.85
THREE_FINGER_GESTURE_CONFIDENCE = 0.85
FOUR_FINGER_SWIPE_GESTURE_CONFIDENCE = 0.85
CIRCLE_GESTURE_CONFIDENCE = 0.80
PINCH_ZOOM_GESTURE_CONFIDENCE = 0.85
LEFT_MODIFIER_GESTURE_CONFIDENCE = 0.85
TERMINATION_GESTURE_CONFIDENCE = 0.95
UNKNOWN_GESTURE_CONFIDENCE = 0.0

# ── Circle detection ──────────────────────────────────────────────────────────
CIRCLE_MIN_PATH_LENGTH = 120.0
CIRCLE_MIN_WIDTH = 30.0
CIRCLE_MIN_HEIGHT = 30.0
CIRCLE_MAX_START_END_DISTANCE = 60.0

# ── Termination gesture ───────────────────────────────────────────────────────
TERMINATION_INDEX_DISTANCE_THRESHOLD = 80
TERMINATION_POINTING_MARGIN = 10
TERMINATION_VERTICAL_ALIGNMENT_THRESHOLD = 40
TERMINATION_HOLD_FRAMES = 6
TERMINATE_GESTURE_HOLD_FRAMES = 6

# ── Gesture names ─────────────────────────────────────────────────────────────
GESTURE_INDEX_POINT = "index_point"
GESTURE_PINCH = "pinch"
GESTURE_PEACE_SIGN = "peace"
GESTURE_TWO_FINGER_SWIPE = "two_finger_swipe"
GESTURE_TWO_FINGER_SWIPE_UP = "two_finger_swipe_up"
GESTURE_TWO_FINGER_SWIPE_DOWN = "two_finger_swipe_down"
GESTURE_THREE_FINGER_LEFT = "three_finger_left"
GESTURE_THREE_FINGER_RIGHT = "three_finger_right"
GESTURE_THREE_FINGER_SCROLL_UP = "three_finger_scroll_up"
GESTURE_THREE_FINGER_SCROLL_DOWN = "three_finger_scroll_down"
GESTURE_FOUR_FINGER_SWIPE = "four_finger_swipe"
GESTURE_FOUR_FINGER_SWIPE_UP = "four_finger_swipe_up"
GESTURE_FOUR_FINGER_SWIPE_DOWN = "four_finger_swipe_down"
GESTURE_PINCH_ZOOM_IN = "pinch_zoom_in"
GESTURE_PINCH_ZOOM_OUT = "pinch_zoom_out"
GESTURE_CIRCLE = "circle"
GESTURE_CLOSED_FIST = "fist"
GESTURE_OPEN_PALM = "open_palm"
GESTURE_THUMBS_UP = "thumbs_up"
GESTURE_ROCK_ON = "rock_on"
GESTURE_RIGHT_CLICK = "right_click"
GESTURE_TWO_HANDS_X = "two_hands_x"
GESTURE_UNKNOWN = "unknown"
# Modifier (left hand)
GESTURE_INDEX_LEFT = "index_left"
GESTURE_PEACE_LEFT = "peace_left"
GESTURE_THREE_FINGERS_LEFT = "three_fingers_left"
GESTURE_OPEN_PALM_LEFT = "open_palm_left"

# ── Modifier names ────────────────────────────────────────────────────────────
MODIFIER_NONE = "none"
MODIFIER_CTRL = "ctrl"
MODIFIER_SHIFT = "shift"
MODIFIER_ALT = "alt"

# ── Landmark IDs ──────────────────────────────────────────────────────────────
WRIST_ID = 0
THUMB_TIP_ID = 4
THUMB_IP_ID = 3
THUMB_MCP_ID = 2
INDEX_FINGER_TIP_ID = 8
INDEX_FINGER_PIP_ID = 6
MIDDLE_FINGER_TIP_ID = 12
MIDDLE_FINGER_PIP_ID = 10
RING_FINGER_TIP_ID = 16
RING_FINGER_PIP_ID = 14
PINKY_TIP_ID = 20
PINKY_PIP_ID = 18
HAND_LANDMARK_COUNT = 21
HAND_LEFT = "Left"
HAND_RIGHT = "Right"
HAND_UNKNOWN = "Unknown"

# ── HUD ───────────────────────────────────────────────────────────────────────
HUD_FONT_SCALE = 0.55
HUD_FONT_THICKNESS = 1
HUD_TEXT_COLOR = (255, 255, 255)
HUD_WARNING_COLOR = (0, 100, 255)
CURSOR_INDICATOR_COLOR = (0, 255, 255)
CURSOR_INDICATOR_RADIUS = 8
CURSOR_INDICATOR_THICKNESS = 2

# ── States ────────────────────────────────────────────────────────────────────
STATE_IDLE = "IDLE"
STATE_CURSOR = "CURSOR"
STATE_SCROLL = "SCROLL"
STATE_WRITE = "WRITE"
STATE_KEYBOARD = "KEYBOARD"
STATE_LAUNCHER = "LAUNCHER"
STATE_TERMINATED = "TERMINATED"

# ── State hold frames ─────────────────────────────────────────────────────────
GESTURE_HOLD_FRAMES = 3       # frames gesture must be stable before firing
LAUNCHER_HOLD_FRAMES = 25
MODE_SWITCH_HOLD_FRAMES = 28

# ── Keyboard ──────────────────────────────────────────────────────────────────
KEYBOARD_HOVER_SECONDS = 1.0
KB_HEIGHT_RATIO = 0.46

# ── Fatigue ───────────────────────────────────────────────────────────────────
FATIGUE_ALPHA = 0.7
FATIGUE_THRESHOLD = 0.15
FATIGUE_WINDOW_FRAMES = 30
FATIGUE_DURATION_SECONDS = 5
FATIGUE_ALERT_COOLDOWN_SECONDS = 60.0
FATIGUE_SCORE_SCALE = 10.0

# ── Air writer / EMNIST ───────────────────────────────────────────────────────
CANVAS_SIZE = 280
CHAR_SIZE = 28
WRITE_CONFIDENCE_THRESHOLD = 0.70
AIR_WRITER_STROKE_THICKNESS = 16
AIR_WRITER_STATIONARY_DISTANCE = 5
AIR_WRITER_MIN_DRAW_DISTANCE = 2
AIR_WRITER_MIN_PIXELS = 48
AIR_WRITER_PADDING = 16
PAUSE_THRESHOLD = 0.8

# ── KNN trainer ───────────────────────────────────────────────────────────────
KNN_K = 3
KNN_SAMPLES_REQUIRED = 5
TRAINER_MIN_SAMPLES_TO_PREDICT = 3
LANDMARK_VECTOR_LENGTH = 63
MODELS_DIR = str(BASE_DIR / "models")
PERSONAL_MODEL_FILE = str(BASE_DIR / "models" / "personal_gestures.pkl")
EMNIST_MODEL_FILE = str(BASE_DIR / "models" / "emnist_cnn.h5")
TRAINER_SUPPORTED_GESTURES = [
    GESTURE_INDEX_POINT, GESTURE_PINCH, GESTURE_PEACE_SIGN,
    GESTURE_THUMBS_UP, GESTURE_ROCK_ON, GESTURE_OPEN_PALM,
    GESTURE_CLOSED_FIST, GESTURE_RIGHT_CLICK,
]

# ── Profiles ──────────────────────────────────────────────────────────────────
PROFILES_DIR = str(BASE_DIR / "profiles")

# ── Exit ──────────────────────────────────────────────────────────────────────
EXIT_KEY = ord("q")

# ── Additional gesture constants ──────────────────────────────────────────────
ZOOM_DISTANCE_DELTA_THRESHOLD = 20.0
