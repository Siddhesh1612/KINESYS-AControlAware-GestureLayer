"""Advanced MediaPipe hand tracker with motion history and role-based gesture analysis.

Combines the best of kinesisv3 (clean modular design) and KINESYS AIR_TOUCH_system
(motion features, confidence scoring, modifier hand support).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import cv2
import mediapipe as mp

from config import (
    ACTION_CONFIDENCE_THRESHOLD,
    CIRCLE_GESTURE_CONFIDENCE, CIRCLE_MAX_START_END_DISTANCE,
    CIRCLE_MIN_HEIGHT, CIRCLE_MIN_PATH_LENGTH, CIRCLE_MIN_WIDTH,
    CLOSED_FIST_GESTURE_CONFIDENCE, CURSOR_INDICATOR_COLOR,
    CURSOR_INDICATOR_RADIUS, CURSOR_INDICATOR_THICKNESS,
    FOUR_FINGER_SWIPE_GESTURE_CONFIDENCE, FOUR_FINGER_SWIPE_THRESHOLD,
    FRAME_WAIT_KEY_MS,
    GESTURE_CIRCLE, GESTURE_CLOSED_FIST, GESTURE_FOUR_FINGER_SWIPE,
    GESTURE_FOUR_FINGER_SWIPE_DOWN, GESTURE_FOUR_FINGER_SWIPE_UP,
    GESTURE_INDEX_LEFT, GESTURE_INDEX_POINT, GESTURE_OPEN_PALM,
    GESTURE_OPEN_PALM_LEFT, GESTURE_PEACE_LEFT, GESTURE_PEACE_SIGN,
    GESTURE_PINCH, GESTURE_PINCH_ZOOM_IN, GESTURE_PINCH_ZOOM_OUT,
    GESTURE_RIGHT_CLICK, GESTURE_ROCK_ON, GESTURE_THREE_FINGER_LEFT,
    GESTURE_THREE_FINGER_RIGHT, GESTURE_THREE_FINGER_SCROLL_DOWN,
    GESTURE_THREE_FINGER_SCROLL_UP, GESTURE_THREE_FINGERS_LEFT,
    GESTURE_THUMBS_UP, GESTURE_TWO_FINGER_SWIPE, GESTURE_TWO_FINGER_SWIPE_DOWN,
    GESTURE_TWO_FINGER_SWIPE_UP, GESTURE_TWO_HANDS_X, GESTURE_UNKNOWN,
    GESTURE_HISTORY_FRAMES, HAND_LANDMARK_COUNT, HAND_LEFT, HAND_RIGHT,
    HAND_UNKNOWN, HUD_FONT_SCALE, HUD_FONT_THICKNESS, HUD_TEXT_COLOR,
    INDEX_FINGER_PIP_ID, INDEX_FINGER_TIP_ID, INDEX_GESTURE_CONFIDENCE,
    LEFT_MODIFIER_GESTURE_CONFIDENCE, MAX_HANDS, MIDDLE_FINGER_PIP_ID,
    MIDDLE_FINGER_TIP_ID, MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE,
    MODIFIER_ALT, MODIFIER_CTRL, MODIFIER_NONE, MODIFIER_SHIFT,
    OPEN_PALM_GESTURE_CONFIDENCE, PEACE_FINGER_SPREAD_THRESHOLD,
    PEACE_SIGN_GESTURE_CONFIDENCE, PEACE_SWIPE_MIN_HISTORY,
    PINCH_GESTURE_CONFIDENCE, PINCH_THRESHOLD, PINCH_ZOOM_GESTURE_CONFIDENCE,
    PINKY_PIP_ID, PINKY_TIP_ID, RING_FINGER_PIP_ID, RING_FINGER_TIP_ID,
    RIGHT_CLICK_GESTURE_CONFIDENCE, ROCK_ON_GESTURE_CONFIDENCE,
    SWIPE_HORIZONTAL_THRESHOLD, SWIPE_VERTICAL_THRESHOLD,
    TERMINATION_GESTURE_CONFIDENCE, TERMINATION_INDEX_DISTANCE_THRESHOLD,
    TERMINATION_POINTING_MARGIN, TERMINATION_VERTICAL_ALIGNMENT_THRESHOLD,
    THREE_FINGER_GESTURE_CONFIDENCE, THUMB_IP_ID, THUMB_MCP_ID, THUMB_TIP_ID,
    THUMBS_UP_GESTURE_CONFIDENCE, TWO_FINGER_SWIPE_GESTURE_CONFIDENCE,
    UNKNOWN_GESTURE_CONFIDENCE, WRIST_ID, ZOOM_DISTANCE_DELTA_THRESHOLD,
    ZOOM_THRESHOLD,
)

FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX


@dataclass(slots=True)
class FingerState:
    thumb: bool
    index: bool
    middle: bool
    ring: bool
    pinky: bool
    extended_count: int


@dataclass(slots=True)
class MotionFeatures:
    palm_dx: float
    palm_dy: float
    index_path_length: float
    pinch_delta: float
    index_bbox_width: float
    index_bbox_height: float
    start_end_distance: float
    history_length: int


@dataclass(slots=True)
class HandHistorySample:
    palm_center_px: tuple[int, int]
    index_tip_px: tuple[int, int]
    pinch_distance: float


@dataclass(slots=True)
class HandObservation:
    handedness: str
    landmarks_px: list[tuple[int, int]]
    landmarks_norm: list[tuple[float, float, float]]
    finger_state: FingerState
    palm_center_px: tuple[int, int]
    pinch_distance: float
    motion_features: MotionFeatures
    gesture: str
    confidence: float


@dataclass(slots=True)
class FrameAnalysis:
    hands: list[HandObservation]
    action_hand: HandObservation | None
    modifier_hand: HandObservation | None
    action_gesture: str
    action_confidence: float
    modifier_gesture: str | None
    modifier_confidence: float
    modifier_active: str | None
    termination_detected: bool
    termination_confidence: float


class HandTracker:
    """Track hands, classify gestures, and expose role-aware frame analysis."""

    def __init__(self) -> None:
        self._mp_hands = mp.solutions.hands
        self._mp_drawing = mp.solutions.drawing_utils
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=MAX_HANDS,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        )
        self._history: dict[str, deque[HandHistorySample]] = {
            HAND_RIGHT: deque(maxlen=GESTURE_HISTORY_FRAMES),
            HAND_LEFT: deque(maxlen=GESTURE_HISTORY_FRAMES),
            HAND_UNKNOWN: deque(maxlen=GESTURE_HISTORY_FRAMES),
        }
        self._last_results = None
        self._last_analysis = FrameAnalysis(
            hands=[], action_hand=None, modifier_hand=None,
            action_gesture=GESTURE_UNKNOWN, action_confidence=UNKNOWN_GESTURE_CONFIDENCE,
            modifier_gesture=None, modifier_confidence=UNKNOWN_GESTURE_CONFIDENCE,
            modifier_active=MODIFIER_NONE, termination_detected=False,
            termination_confidence=UNKNOWN_GESTURE_CONFIDENCE,
        )

    def process(self, frame: cv2.typing.MatLike) -> FrameAnalysis:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._hands.process(rgb)
        rgb.flags.writeable = True

        h, w = frame.shape[:2]
        hands: list[HandObservation] = []

        if results.multi_hand_landmarks and results.multi_handedness:
            for lms, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness.classification[0].label
                hands.append(self._build_observation(lms, label, w, h))

        action_hand, modifier_hand = self._resolve_roles(hands)
        self._apply_gestures(action_hand, modifier_hand)

        mod_gesture = modifier_hand.gesture if modifier_hand else None
        mod_conf = modifier_hand.confidence if modifier_hand else UNKNOWN_GESTURE_CONFIDENCE
        term_detected, term_conf = self._detect_two_hands_x(action_hand, modifier_hand)

        self._last_results = results
        self._last_analysis = FrameAnalysis(
            hands=hands,
            action_hand=action_hand,
            modifier_hand=modifier_hand,
            action_gesture=action_hand.gesture if action_hand else GESTURE_UNKNOWN,
            action_confidence=action_hand.confidence if action_hand else UNKNOWN_GESTURE_CONFIDENCE,
            modifier_gesture=mod_gesture,
            modifier_confidence=mod_conf,
            modifier_active=self._map_modifier(mod_gesture),
            termination_detected=term_detected,
            termination_confidence=term_conf,
        )
        return self._last_analysis

    def draw_annotations(self, frame: cv2.typing.MatLike) -> cv2.typing.MatLike:
        if not self._last_results or not self._last_results.multi_hand_landmarks:
            return frame
        for i, lms in enumerate(self._last_results.multi_hand_landmarks):
            self._mp_drawing.draw_landmarks(frame, lms, self._mp_hands.HAND_CONNECTIONS)
            if i >= len(self._last_analysis.hands):
                continue
            obs = self._last_analysis.hands[i]
            wx, wy = obs.landmarks_px[WRIST_ID]
            cv2.putText(frame, f"{obs.handedness}: {obs.gesture} ({obs.confidence:.2f})",
                        (wx + 10, wy - 10), FONT_FACE, HUD_FONT_SCALE, HUD_TEXT_COLOR,
                        HUD_FONT_THICKNESS, cv2.LINE_AA)
            if obs.gesture in {GESTURE_INDEX_POINT, GESTURE_PINCH, GESTURE_RIGHT_CLICK,
                               GESTURE_PEACE_SIGN, GESTURE_THUMBS_UP, GESTURE_ROCK_ON,
                               GESTURE_TWO_FINGER_SWIPE_UP, GESTURE_TWO_FINGER_SWIPE_DOWN}:
                cx, cy = obs.landmarks_px[INDEX_FINGER_TIP_ID]
                cv2.circle(frame, (cx, cy), CURSOR_INDICATOR_RADIUS,
                           CURSOR_INDICATOR_COLOR, CURSOR_INDICATOR_THICKNESS)
        return frame

    def close(self) -> None:
        self._hands.close()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _build_observation(self, hand_lms, label: str, fw: int, fh: int) -> HandObservation:
        norm_label = label if label in {HAND_LEFT, HAND_RIGHT} else HAND_UNKNOWN
        lm_norm = [(lm.x, lm.y, lm.z) for lm in hand_lms.landmark[:HAND_LANDMARK_COUNT]]
        lm_px = [(min(max(int(lm.x * fw), 0), fw - 1),
                  min(max(int(lm.y * fh), 0), fh - 1))
                 for lm in hand_lms.landmark[:HAND_LANDMARK_COUNT]]
        finger_state = self._compute_finger_state(lm_px, norm_label)
        palm_center = self._compute_palm_center(lm_px)
        pinch_dist = self._dist(lm_px[THUMB_TIP_ID], lm_px[INDEX_FINGER_TIP_ID])
        motion = self._compute_motion(norm_label, palm_center, lm_px[INDEX_FINGER_TIP_ID], pinch_dist)
        return HandObservation(
            handedness=norm_label, landmarks_px=lm_px, landmarks_norm=lm_norm,
            finger_state=finger_state, palm_center_px=palm_center,
            pinch_distance=pinch_dist, motion_features=motion,
            gesture=GESTURE_UNKNOWN, confidence=UNKNOWN_GESTURE_CONFIDENCE,
        )

    def _resolve_roles(self, hands):
        if not hands:
            return None, None
        if len(hands) == 1:
            return hands[0], None
        right = next((h for h in hands if h.handedness == HAND_RIGHT), None)
        left = next((h for h in hands if h.handedness == HAND_LEFT), None)
        if right and left:
            return right, left
        return hands[0], hands[1] if len(hands) > 1 else None

    def _apply_gestures(self, action_hand, modifier_hand):
        if action_hand:
            g, c = self._classify_action(action_hand)
            action_hand.gesture = g if c >= ACTION_CONFIDENCE_THRESHOLD else GESTURE_UNKNOWN
            action_hand.confidence = c if c >= ACTION_CONFIDENCE_THRESHOLD else UNKNOWN_GESTURE_CONFIDENCE
        if modifier_hand:
            g, c = self._classify_modifier(modifier_hand)
            modifier_hand.gesture = g if c >= ACTION_CONFIDENCE_THRESHOLD else GESTURE_UNKNOWN
            modifier_hand.confidence = c if c >= ACTION_CONFIDENCE_THRESHOLD else UNKNOWN_GESTURE_CONFIDENCE

    def _classify_action(self, obs: HandObservation) -> tuple[str, float]:
        """Classify gesture using kinesisv3 landmark logic with pinch checked first."""
        lm = obs.landmarks_px
        label = obs.handedness

        # ── Pinch FIRST — thumb tip close to index tip ────────────────────────
        # Must check before finger-up logic because during a pinch the index
        # tip is still above PIP (finger looks "up") and would wrongly fire "pointing".
        pinch_dist = obs.pinch_distance
        if pinch_dist <= PINCH_THRESHOLD:
            return "pinch", PINCH_GESTURE_CONFIDENCE

        # ── Finger extension (y-axis: tip above PIP = extended) ───────────────
        if label == HAND_RIGHT:
            thumb_up = lm[THUMB_TIP_ID][0] < lm[THUMB_MCP_ID][0]
        else:
            thumb_up = lm[THUMB_TIP_ID][0] > lm[THUMB_MCP_ID][0]

        index_up  = lm[INDEX_FINGER_TIP_ID][1]  < lm[INDEX_FINGER_PIP_ID][1]
        middle_up = lm[MIDDLE_FINGER_TIP_ID][1] < lm[MIDDLE_FINGER_PIP_ID][1]
        ring_up   = lm[RING_FINGER_TIP_ID][1]   < lm[RING_FINGER_PIP_ID][1]
        pinky_up  = lm[PINKY_TIP_ID][1]         < lm[PINKY_PIP_ID][1]

        # No fingers up → thumbs_up or fist
        if not any([index_up, middle_up, ring_up, pinky_up]):
            return ("thumbs_up", THUMBS_UP_GESTURE_CONFIDENCE) if thumb_up \
                else ("fist", CLOSED_FIST_GESTURE_CONFIDENCE)

        # Index only → pointing
        if index_up and not middle_up and not ring_up and not pinky_up:
            return "pointing", INDEX_GESTURE_CONFIDENCE

        # Index + middle → peace
        if index_up and middle_up and not ring_up and not pinky_up:
            return "peace", PEACE_SIGN_GESTURE_CONFIDENCE

        # Index + pinky → rock_on
        if index_up and not middle_up and not ring_up and pinky_up:
            return "rock_on", ROCK_ON_GESTURE_CONFIDENCE

        # All four fingers → open_hand
        if index_up and middle_up and ring_up and pinky_up:
            return "open_hand", OPEN_PALM_GESTURE_CONFIDENCE

        return "tracking", UNKNOWN_GESTURE_CONFIDENCE

    def _classify_modifier(self, obs: HandObservation) -> tuple[str, float]:
        fs = obs.finger_state
        if fs.index and fs.middle and fs.ring and not fs.pinky:
            return GESTURE_THREE_FINGERS_LEFT, LEFT_MODIFIER_GESTURE_CONFIDENCE
        if fs.index and fs.middle and not fs.ring and not fs.pinky:
            return GESTURE_PEACE_LEFT, LEFT_MODIFIER_GESTURE_CONFIDENCE
        if fs.index and not fs.middle and not fs.ring and not fs.pinky:
            return GESTURE_INDEX_LEFT, LEFT_MODIFIER_GESTURE_CONFIDENCE
        if fs.index and fs.middle and fs.ring and fs.pinky:
            return GESTURE_OPEN_PALM_LEFT, LEFT_MODIFIER_GESTURE_CONFIDENCE
        return GESTURE_UNKNOWN, UNKNOWN_GESTURE_CONFIDENCE

    def _compute_finger_state(self, lm_px, label) -> FingerState:
        thumb = self._is_thumb_extended(lm_px, label)
        index = lm_px[INDEX_FINGER_TIP_ID][1] < lm_px[INDEX_FINGER_PIP_ID][1]
        middle = lm_px[MIDDLE_FINGER_TIP_ID][1] < lm_px[MIDDLE_FINGER_PIP_ID][1]
        ring = lm_px[RING_FINGER_TIP_ID][1] < lm_px[RING_FINGER_PIP_ID][1]
        pinky = lm_px[PINKY_TIP_ID][1] < lm_px[PINKY_PIP_ID][1]
        return FingerState(thumb=thumb, index=index, middle=middle, ring=ring, pinky=pinky,
                           extended_count=sum([thumb, index, middle, ring, pinky]))

    def _compute_palm_center(self, lm_px) -> tuple[int, int]:
        pts = [lm_px[WRIST_ID], lm_px[INDEX_FINGER_PIP_ID], lm_px[MIDDLE_FINGER_PIP_ID],
               lm_px[RING_FINGER_PIP_ID], lm_px[PINKY_PIP_ID]]
        return int(sum(p[0] for p in pts) / len(pts)), int(sum(p[1] for p in pts) / len(pts))

    def _compute_motion(self, label, palm_center, index_tip, pinch_dist) -> MotionFeatures:
        history = self._history[label]
        history.append(HandHistorySample(palm_center, index_tip, pinch_dist))
        if not history:
            return MotionFeatures(0, 0, 0, 0, 0, 0, 0, 0)
        first, last = history[0], history[-1]
        palm_dx = float(last.palm_center_px[0] - first.palm_center_px[0])
        palm_dy = float(last.palm_center_px[1] - first.palm_center_px[1])
        pinch_delta = last.pinch_distance - first.pinch_distance
        pts = [s.index_tip_px for s in history]
        path_len = sum(self._dist(pts[i - 1], pts[i]) for i in range(1, len(pts)))
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        return MotionFeatures(
            palm_dx=palm_dx, palm_dy=palm_dy, index_path_length=path_len,
            pinch_delta=pinch_delta,
            index_bbox_width=float(max(xs) - min(xs)),
            index_bbox_height=float(max(ys) - min(ys)),
            start_end_distance=self._dist(pts[0], pts[-1]),
            history_length=len(history),
        )

    def _detect_two_hands_x(self, action, modifier) -> tuple[bool, float]:
        if not action or not modifier:
            return False, UNKNOWN_GESTURE_CONFIDENCE
        rw_x = action.landmarks_px[WRIST_ID][0]
        lw_x = modifier.landmarks_px[WRIST_ID][0]
        ri_tip = action.landmarks_px[INDEX_FINGER_TIP_ID]
        ri_pip = action.landmarks_px[INDEX_FINGER_PIP_ID]
        li_tip = modifier.landmarks_px[INDEX_FINGER_TIP_ID]
        li_pip = modifier.landmarks_px[INDEX_FINGER_PIP_ID]
        if (rw_x < lw_x
                and ri_tip[0] < ri_pip[0] - TERMINATION_POINTING_MARGIN
                and li_tip[0] > li_pip[0] + TERMINATION_POINTING_MARGIN
                and self._dist(ri_tip, li_tip) <= TERMINATION_INDEX_DISTANCE_THRESHOLD
                and abs(ri_tip[1] - li_tip[1]) <= TERMINATION_VERTICAL_ALIGNMENT_THRESHOLD
                and action.finger_state.index and modifier.finger_state.index):
            return True, TERMINATION_GESTURE_CONFIDENCE
        return False, UNKNOWN_GESTURE_CONFIDENCE

    def _is_circle(self, m: MotionFeatures) -> bool:
        return (m.index_path_length >= CIRCLE_MIN_PATH_LENGTH
                and m.index_bbox_width >= CIRCLE_MIN_WIDTH
                and m.index_bbox_height >= CIRCLE_MIN_HEIGHT
                and m.start_end_distance <= CIRCLE_MAX_START_END_DISTANCE)

    @staticmethod
    def _map_modifier(gesture: str | None) -> str:
        if gesture == GESTURE_INDEX_LEFT:
            return MODIFIER_CTRL
        if gesture == GESTURE_PEACE_LEFT:
            return MODIFIER_SHIFT
        if gesture == GESTURE_THREE_FINGERS_LEFT:
            return MODIFIER_ALT
        return MODIFIER_NONE

    @staticmethod
    def _is_thumb_extended(lm_px, label) -> bool:
        tip_x = lm_px[THUMB_TIP_ID][0]
        mcp_x = lm_px[THUMB_MCP_ID][0]
        return tip_x < mcp_x if label == HAND_RIGHT else tip_x > mcp_x

    @staticmethod
    def _dist(a: tuple[int, int], b: tuple[int, int]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])
