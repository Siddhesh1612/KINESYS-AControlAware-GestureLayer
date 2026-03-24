"""MediaPipe hand tracking and gesture analysis for the KINESYS gesture engine."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import cv2
import mediapipe as mp

from config import (
    ACTION_CONFIDENCE_THRESHOLD,
    CIRCLE_GESTURE_CONFIDENCE,
    CIRCLE_MAX_START_END_DISTANCE,
    CIRCLE_MIN_HEIGHT,
    CIRCLE_MIN_PATH_LENGTH,
    CIRCLE_MIN_WIDTH,
    CLOSED_FIST_GESTURE_CONFIDENCE,
    CURSOR_INDICATOR_COLOR,
    CURSOR_INDICATOR_RADIUS,
    CURSOR_INDICATOR_THICKNESS,
    FOUR_FINGER_SWIPE_GESTURE_CONFIDENCE,
    FOUR_FINGER_SWIPE_THRESHOLD,
    FRAME_WAIT_KEY_MS,
    GESTURE_CIRCLE,
    GESTURE_CLOSED_FIST,
    GESTURE_FOUR_FINGER_SWIPE,
    GESTURE_FOUR_FINGER_SWIPE_DOWN,
    GESTURE_FOUR_FINGER_SWIPE_UP,
    GESTURE_INDEX_LEFT,
    GESTURE_INDEX_POINT,
    GESTURE_OPEN_PALM,
    GESTURE_OPEN_PALM_LEFT,
    GESTURE_PEACE_LEFT,
    GESTURE_PEACE_SIGN,
    GESTURE_PINCH,
    GESTURE_PINCH_ZOOM_IN,
    GESTURE_PINCH_ZOOM_OUT,
    GESTURE_RIGHT_CLICK,
    GESTURE_ROCK_ON,
    GESTURE_THREE_FINGER_LEFT,
    GESTURE_THREE_FINGER_RIGHT,
    GESTURE_THREE_FINGER_SCROLL_UP,
    GESTURE_THREE_FINGER_SCROLL_DOWN,
    GESTURE_THREE_FINGERS_LEFT,
    GESTURE_THUMBS_UP,
    GESTURE_TWO_FINGER_SWIPE,
    GESTURE_TWO_FINGER_SWIPE_DOWN,
    GESTURE_TWO_FINGER_SWIPE_UP,
    GESTURE_UNKNOWN,
    GESTURE_TWO_HANDS_X,
    GESTURE_HISTORY_FRAMES,
    HAND_LANDMARK_COUNT,
    HAND_LEFT,
    HAND_RIGHT,
    HAND_UNKNOWN,
    HUD_FONT_SCALE,
    HUD_FONT_THICKNESS,
    HUD_TEXT_COLOR,
    INDEX_FINGER_PIP_ID,
    INDEX_FINGER_TIP_ID,
    INDEX_GESTURE_CONFIDENCE,
    LEFT_MODIFIER_GESTURE_CONFIDENCE,
    MAX_HANDS,
    MIDDLE_FINGER_PIP_ID,
    MIDDLE_FINGER_TIP_ID,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    MODIFIER_ALT,
    MODIFIER_CTRL,
    MODIFIER_NONE,
    MODIFIER_SHIFT,
    OPEN_PALM_GESTURE_CONFIDENCE,
    PEACE_FINGER_SPREAD_THRESHOLD,
    PEACE_SIGN_GESTURE_CONFIDENCE,
    PEACE_SWIPE_MIN_HISTORY,
    PINCH_GESTURE_CONFIDENCE,
    PINCH_THRESHOLD,
    PINCH_ZOOM_GESTURE_CONFIDENCE,
    PINKY_PIP_ID,
    PINKY_TIP_ID,
    RING_FINGER_PIP_ID,
    RING_FINGER_TIP_ID,
    RIGHT_CLICK_GESTURE_CONFIDENCE,
    ROCK_ON_GESTURE_CONFIDENCE,
    SWIPE_HORIZONTAL_THRESHOLD,
    SWIPE_VERTICAL_THRESHOLD,
    TERMINATION_GESTURE_CONFIDENCE,
    TERMINATION_INDEX_DISTANCE_THRESHOLD,
    TERMINATION_POINTING_MARGIN,
    TERMINATION_VERTICAL_ALIGNMENT_THRESHOLD,
    THREE_FINGER_GESTURE_CONFIDENCE,
    THUMB_IP_ID,
    THUMB_TIP_ID,
    THUMBS_UP_GESTURE_CONFIDENCE,
    TWO_FINGER_SWIPE_GESTURE_CONFIDENCE,
    UNKNOWN_GESTURE_CONFIDENCE,
    WRIST_ID,
    ZOOM_DISTANCE_DELTA_THRESHOLD,
    ZOOM_THRESHOLD,
)


FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
TEXT_OFFSET_X = 10
TEXT_OFFSET_Y = 10
FIRST_CLASSIFICATION_INDEX = 0
EMPTY_HISTORY_LENGTH = 0
FIRST_POINT_INDEX = 0
SECOND_POINT_INDEX = 1


@dataclass(slots=True)
class FingerState:
    """Boolean representation of which fingers are currently extended."""

    thumb: bool
    index: bool
    middle: bool
    ring: bool
    pinky: bool
    extended_count: int


@dataclass(slots=True)
class MotionFeatures:
    """Aggregated short-term motion features for one tracked hand."""

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
    """Compact per-frame sample stored for motion-aware gesture detection."""

    palm_center_px: tuple[int, int]
    index_tip_px: tuple[int, int]
    pinch_distance: float


@dataclass(slots=True)
class HandObservation:
    """Represents one tracked hand and its geometry-derived gesture state."""

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
    """Summarizes the gestures and roles detected in the current frame."""

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
        """Initialize MediaPipe Hands and short-term motion history buffers."""

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
            hands=[],
            action_hand=None,
            modifier_hand=None,
            action_gesture=GESTURE_UNKNOWN,
            action_confidence=UNKNOWN_GESTURE_CONFIDENCE,
            modifier_gesture=None,
            modifier_confidence=UNKNOWN_GESTURE_CONFIDENCE,
            modifier_active=MODIFIER_NONE,
            termination_detected=False,
            termination_confidence=UNKNOWN_GESTURE_CONFIDENCE,
        )

    def process(self, frame: cv2.typing.MatLike) -> FrameAnalysis:
        """Process a BGR frame and return role-aware gesture analysis."""

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self._hands.process(rgb_frame)
        rgb_frame.flags.writeable = True

        frame_height, frame_width = frame.shape[:2]
        hands: list[HandObservation] = []

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks,
                results.multi_handedness,
            ):
                handedness_label = handedness.classification[FIRST_CLASSIFICATION_INDEX].label
                hands.append(
                    self._build_observation(
                        hand_landmarks=hand_landmarks,
                        handedness_label=handedness_label,
                        frame_width=frame_width,
                        frame_height=frame_height,
                    )
                )

        action_hand, modifier_hand = self._resolve_frame_roles(hands)
        self._apply_role_gestures(action_hand=action_hand, modifier_hand=modifier_hand)

        modifier_gesture = modifier_hand.gesture if modifier_hand is not None else None
        modifier_confidence = (
            modifier_hand.confidence
            if modifier_hand is not None
            else UNKNOWN_GESTURE_CONFIDENCE
        )
        modifier_active = self._map_modifier(modifier_gesture)
        termination_detected, termination_confidence = self._detect_two_hands_x(
            action_hand=action_hand,
            modifier_hand=modifier_hand,
        )

        self._last_results = results
        self._last_analysis = FrameAnalysis(
            hands=hands,
            action_hand=action_hand,
            modifier_hand=modifier_hand,
            action_gesture=action_hand.gesture if action_hand is not None else GESTURE_UNKNOWN,
            action_confidence=(
                action_hand.confidence
                if action_hand is not None
                else UNKNOWN_GESTURE_CONFIDENCE
            ),
            modifier_gesture=modifier_gesture,
            modifier_confidence=modifier_confidence,
            modifier_active=modifier_active,
            termination_detected=termination_detected,
            termination_confidence=termination_confidence,
        )
        return self._last_analysis

    def draw_annotations(self, frame: cv2.typing.MatLike) -> cv2.typing.MatLike:
        """Draw landmarks, gesture labels, and cursor indicator overlays."""

        if not self._last_results or not self._last_results.multi_hand_landmarks:
            return frame

        for index, hand_landmarks in enumerate(self._last_results.multi_hand_landmarks):
            self._mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                self._mp_hands.HAND_CONNECTIONS,
            )
            if index >= len(self._last_analysis.hands):
                continue

            observation = self._last_analysis.hands[index]
            wrist_x, wrist_y = observation.landmarks_px[WRIST_ID]
            label_x = wrist_x + TEXT_OFFSET_X
            label_y = wrist_y - TEXT_OFFSET_Y
            label = (
                f"{observation.handedness}: {observation.gesture} "
                f"({observation.confidence:.2f})"
            )
            cv2.putText(
                frame,
                label,
                (label_x, label_y),
                FONT_FACE,
                HUD_FONT_SCALE,
                HUD_TEXT_COLOR,
                HUD_FONT_THICKNESS,
                cv2.LINE_AA,
            )

            if observation.gesture in {
                GESTURE_INDEX_POINT,
                GESTURE_PINCH,
                GESTURE_RIGHT_CLICK,
                GESTURE_PINCH_ZOOM_IN,
                GESTURE_PINCH_ZOOM_OUT,
                GESTURE_PEACE_SIGN,
                GESTURE_TWO_FINGER_SWIPE,
                GESTURE_TWO_FINGER_SWIPE_UP,
                GESTURE_TWO_FINGER_SWIPE_DOWN,
                GESTURE_THUMBS_UP,
                GESTURE_ROCK_ON,
            }:
                cursor_x, cursor_y = observation.landmarks_px[INDEX_FINGER_TIP_ID]
                cv2.circle(
                    frame,
                    (cursor_x, cursor_y),
                    CURSOR_INDICATOR_RADIUS,
                    CURSOR_INDICATOR_COLOR,
                    CURSOR_INDICATOR_THICKNESS,
                )

        return frame

    def close(self) -> None:
        """Release MediaPipe resources owned by the tracker."""

        self._hands.close()

    def _build_observation(
        self,
        hand_landmarks: object,
        handedness_label: str,
        frame_width: int,
        frame_height: int,
    ) -> HandObservation:
        """Create a geometry-rich observation from raw MediaPipe output."""

        normalized_handedness = (
            handedness_label if handedness_label in {HAND_LEFT, HAND_RIGHT} else HAND_UNKNOWN
        )
        landmarks_norm = [
            (landmark.x, landmark.y, landmark.z)
            for landmark in hand_landmarks.landmark[:HAND_LANDMARK_COUNT]
        ]
        landmarks_px = [
            (
                min(max(int(landmark.x * frame_width), WRIST_ID), frame_width - FRAME_WAIT_KEY_MS),
                min(max(int(landmark.y * frame_height), WRIST_ID), frame_height - FRAME_WAIT_KEY_MS),
            )
            for landmark in hand_landmarks.landmark[:HAND_LANDMARK_COUNT]
        ]
        finger_state = self._compute_finger_state(
            landmarks_px=landmarks_px,
            handedness_label=normalized_handedness,
        )
        palm_center_px = self._compute_palm_center(landmarks_px)
        pinch_distance = self._distance(
            landmarks_px[THUMB_TIP_ID],
            landmarks_px[INDEX_FINGER_TIP_ID],
        )
        motion_features = self._compute_motion_features(
            handedness_label=normalized_handedness,
            palm_center_px=palm_center_px,
            index_tip_px=landmarks_px[INDEX_FINGER_TIP_ID],
            pinch_distance=pinch_distance,
        )

        return HandObservation(
            handedness=normalized_handedness,
            landmarks_px=landmarks_px,
            landmarks_norm=landmarks_norm,
            finger_state=finger_state,
            palm_center_px=palm_center_px,
            pinch_distance=pinch_distance,
            motion_features=motion_features,
            gesture=GESTURE_UNKNOWN,
            confidence=UNKNOWN_GESTURE_CONFIDENCE,
        )

    def _resolve_frame_roles(
        self,
        hands: list[HandObservation],
    ) -> tuple[HandObservation | None, HandObservation | None]:
        """Assign the current frame's action and modifier hands."""

        if not hands:
            return None, None

        if len(hands) == FIRST_CLASSIFICATION_INDEX + FRAME_WAIT_KEY_MS:
            return hands[FIRST_CLASSIFICATION_INDEX], None

        right_hand = self._find_hand_by_label(hands, HAND_RIGHT)
        left_hand = self._find_hand_by_label(hands, HAND_LEFT)

        if right_hand is not None and left_hand is not None:
            return right_hand, left_hand

        if right_hand is not None:
            remaining_hand = self._find_first_other_hand(hands, right_hand)
            return right_hand, remaining_hand

        if left_hand is not None:
            remaining_hand = self._find_first_other_hand(hands, left_hand)
            return remaining_hand or left_hand, left_hand if remaining_hand is not None else None

        return hands[FIRST_CLASSIFICATION_INDEX], hands[SECOND_POINT_INDEX]

    def _apply_role_gestures(
        self,
        action_hand: HandObservation | None,
        modifier_hand: HandObservation | None,
    ) -> None:
        """Classify gestures according to each hand's current role."""

        if action_hand is not None:
            action_hand.gesture, action_hand.confidence = self._classify_action_hand(action_hand)
            if action_hand.confidence < ACTION_CONFIDENCE_THRESHOLD:
                action_hand.gesture = GESTURE_UNKNOWN
                action_hand.confidence = UNKNOWN_GESTURE_CONFIDENCE

        if modifier_hand is not None:
            modifier_hand.gesture, modifier_hand.confidence = self._classify_modifier_hand(
                modifier_hand
            )
            if modifier_hand.confidence < ACTION_CONFIDENCE_THRESHOLD:
                modifier_hand.gesture = GESTURE_UNKNOWN
                modifier_hand.confidence = UNKNOWN_GESTURE_CONFIDENCE

    def _classify_action_hand(self, observation: HandObservation) -> tuple[str, float]:
        """Classify the right-hand action gesture from geometry and motion features."""

        finger_state = observation.finger_state
        motion = observation.motion_features
        pinch_active = observation.pinch_distance <= PINCH_THRESHOLD
        finger_spread = self._distance(
            observation.landmarks_px[INDEX_FINGER_TIP_ID],
            observation.landmarks_px[MIDDLE_FINGER_TIP_ID],
        )
        four_fingers_extended = (
            finger_state.index
            and finger_state.middle
            and finger_state.ring
            and finger_state.pinky
        )
        three_fingers_extended = (
            finger_state.index
            and finger_state.middle
            and finger_state.ring
            and not finger_state.pinky
        )
        peace_fingers = (
            finger_state.index
            and finger_state.middle
            and not finger_state.ring
            and not finger_state.pinky
        )
        index_only = (
            finger_state.index
            and not finger_state.middle
            and not finger_state.ring
            and not finger_state.pinky
        )
        # Thumbs up: thumb tip clearly above thumb IP (pointing up), all fingers curled
        thumb_tip_y = observation.landmarks_px[THUMB_TIP_ID][1]
        thumb_ip_y = observation.landmarks_px[THUMB_IP_ID][1]
        thumbs_up = (
            (thumb_tip_y < thumb_ip_y - 20)
            and not finger_state.index
            and not finger_state.middle
            and not finger_state.ring
            and not finger_state.pinky
        )
        # Rock on (horns): index + pinky extended, middle + ring curled
        rock_on = (
            finger_state.index
            and not finger_state.middle
            and not finger_state.ring
            and finger_state.pinky
        )

        # ── Circle ───────────────────────────────────────────────────────────
        if index_only and self._is_circle_motion(motion):
            return GESTURE_CIRCLE, CIRCLE_GESTURE_CONFIDENCE

        # ── 4-finger swipe ───────────────────────────────────────────────────
        if four_fingers_extended and abs(motion.palm_dy) >= FOUR_FINGER_SWIPE_THRESHOLD:
            if motion.palm_dy < 0:
                return GESTURE_FOUR_FINGER_SWIPE_UP, FOUR_FINGER_SWIPE_GESTURE_CONFIDENCE
            return GESTURE_FOUR_FINGER_SWIPE_DOWN, FOUR_FINGER_SWIPE_GESTURE_CONFIDENCE

        if four_fingers_extended and abs(motion.palm_dx) >= FOUR_FINGER_SWIPE_THRESHOLD:
            return GESTURE_FOUR_FINGER_SWIPE, FOUR_FINGER_SWIPE_GESTURE_CONFIDENCE

        # ── 3-finger swipe ───────────────────────────────────────────────────
        # Vertical motion → scroll (takes priority over horizontal back/forward)
        if three_fingers_extended and motion.palm_dy <= -SWIPE_VERTICAL_THRESHOLD:
            return GESTURE_THREE_FINGER_SCROLL_UP, THREE_FINGER_GESTURE_CONFIDENCE

        if three_fingers_extended and motion.palm_dy >= SWIPE_VERTICAL_THRESHOLD:
            return GESTURE_THREE_FINGER_SCROLL_DOWN, THREE_FINGER_GESTURE_CONFIDENCE

        # Horizontal motion → back / forward
        if three_fingers_extended and motion.palm_dx <= -SWIPE_HORIZONTAL_THRESHOLD:
            return GESTURE_THREE_FINGER_LEFT, THREE_FINGER_GESTURE_CONFIDENCE

        if three_fingers_extended and motion.palm_dx >= SWIPE_HORIZONTAL_THRESHOLD:
            return GESTURE_THREE_FINGER_RIGHT, THREE_FINGER_GESTURE_CONFIDENCE

        # ── Peace sign — check SPREAD first, then swipe ──────────────────────
        # Must confirm it's a real peace sign (fingers spread apart) before
        # allowing swipe detection — prevents accidental scroll while holding peace.
        if peace_fingers:
            is_peace = finger_spread >= PEACE_FINGER_SPREAD_THRESHOLD
            has_enough_history = motion.history_length >= PEACE_SWIPE_MIN_HISTORY

            # Only allow swipe if we have enough history AND fingers are NOT spread
            # (spread = static peace sign, not a swipe)
            if has_enough_history and not is_peace:
                if motion.palm_dy <= -SWIPE_VERTICAL_THRESHOLD:
                    return GESTURE_TWO_FINGER_SWIPE_UP, TWO_FINGER_SWIPE_GESTURE_CONFIDENCE
                if motion.palm_dy >= SWIPE_VERTICAL_THRESHOLD:
                    return GESTURE_TWO_FINGER_SWIPE_DOWN, TWO_FINGER_SWIPE_GESTURE_CONFIDENCE
                if abs(motion.palm_dy) >= SWIPE_VERTICAL_THRESHOLD:
                    return GESTURE_TWO_FINGER_SWIPE, TWO_FINGER_SWIPE_GESTURE_CONFIDENCE

            if is_peace:
                return GESTURE_PEACE_SIGN, PEACE_SIGN_GESTURE_CONFIDENCE

        # ── Pinch — clean (other fingers curled) ─────────────────────────────
        # Pinch = thumb tip close to index tip, other fingers curled.
        # This is ALWAYS left-click — no exceptions, no confusion.
        pinch_clean = (
            pinch_active
            and not finger_state.middle
            and not finger_state.ring
            and not finger_state.pinky
        )
        if pinch_clean:
            return GESTURE_PINCH, PINCH_GESTURE_CONFIDENCE

        # ── Right-click — "gun" shape ─────────────────────────────────────────
        # Thumb extended sideways + index pointing up, NOT pinching.
        # Thumb tip must be far from index tip so it never overlaps with pinch.
        right_click_shape = (
            index_only
            and finger_state.thumb
            and not pinch_active
        )
        if right_click_shape:
            return GESTURE_RIGHT_CLICK, RIGHT_CLICK_GESTURE_CONFIDENCE

        # ── Remaining gestures ───────────────────────────────────────────────
        if thumbs_up:
            return GESTURE_THUMBS_UP, THUMBS_UP_GESTURE_CONFIDENCE

        if rock_on:
            return GESTURE_ROCK_ON, ROCK_ON_GESTURE_CONFIDENCE

        if index_only:
            return GESTURE_INDEX_POINT, INDEX_GESTURE_CONFIDENCE

        if four_fingers_extended:
            return GESTURE_OPEN_PALM, OPEN_PALM_GESTURE_CONFIDENCE

        if finger_state.extended_count == WRIST_ID:
            return GESTURE_CLOSED_FIST, CLOSED_FIST_GESTURE_CONFIDENCE

        return GESTURE_UNKNOWN, UNKNOWN_GESTURE_CONFIDENCE

    def _classify_modifier_hand(self, observation: HandObservation) -> tuple[str, float]:
        """Classify the left-hand modifier gesture."""

        finger_state = observation.finger_state

        if (
            finger_state.index
            and finger_state.middle
            and finger_state.ring
            and not finger_state.pinky
        ):
            return GESTURE_THREE_FINGERS_LEFT, LEFT_MODIFIER_GESTURE_CONFIDENCE

        if (
            finger_state.index
            and finger_state.middle
            and not finger_state.ring
            and not finger_state.pinky
        ):
            return GESTURE_PEACE_LEFT, LEFT_MODIFIER_GESTURE_CONFIDENCE

        if (
            finger_state.index
            and not finger_state.middle
            and not finger_state.ring
            and not finger_state.pinky
        ):
            return GESTURE_INDEX_LEFT, LEFT_MODIFIER_GESTURE_CONFIDENCE

        if (
            finger_state.index
            and finger_state.middle
            and finger_state.ring
            and finger_state.pinky
        ):
            return GESTURE_OPEN_PALM_LEFT, LEFT_MODIFIER_GESTURE_CONFIDENCE

        return GESTURE_UNKNOWN, UNKNOWN_GESTURE_CONFIDENCE

    def _compute_finger_state(
        self,
        landmarks_px: list[tuple[int, int]],
        handedness_label: str,
    ) -> FingerState:
        """Return the extension state for all five fingers."""

        thumb_extended = self._is_thumb_extended(landmarks_px, handedness_label)
        index_extended = self._is_finger_extended(landmarks_px, INDEX_FINGER_TIP_ID, INDEX_FINGER_PIP_ID)
        middle_extended = self._is_finger_extended(
            landmarks_px,
            MIDDLE_FINGER_TIP_ID,
            MIDDLE_FINGER_PIP_ID,
        )
        ring_extended = self._is_finger_extended(landmarks_px, RING_FINGER_TIP_ID, RING_FINGER_PIP_ID)
        pinky_extended = self._is_finger_extended(landmarks_px, PINKY_TIP_ID, PINKY_PIP_ID)
        extended_count = sum(
            [
                thumb_extended,
                index_extended,
                middle_extended,
                ring_extended,
                pinky_extended,
            ]
        )

        return FingerState(
            thumb=thumb_extended,
            index=index_extended,
            middle=middle_extended,
            ring=ring_extended,
            pinky=pinky_extended,
            extended_count=extended_count,
        )

    def _compute_palm_center(
        self,
        landmarks_px: list[tuple[int, int]],
    ) -> tuple[int, int]:
        """Compute a stable palm-center anchor from wrist and finger bases."""

        points = [
            landmarks_px[WRIST_ID],
            landmarks_px[INDEX_FINGER_PIP_ID],
            landmarks_px[MIDDLE_FINGER_PIP_ID],
            landmarks_px[RING_FINGER_PIP_ID],
            landmarks_px[PINKY_PIP_ID],
        ]
        average_x = int(sum(point[FIRST_CLASSIFICATION_INDEX] for point in points) / len(points))
        average_y = int(sum(point[SECOND_POINT_INDEX] for point in points) / len(points))
        return average_x, average_y

    def _compute_motion_features(
        self,
        handedness_label: str,
        palm_center_px: tuple[int, int],
        index_tip_px: tuple[int, int],
        pinch_distance: float,
    ) -> MotionFeatures:
        """Update and summarize the rolling motion history for one hand."""

        history = self._history[handedness_label]
        history.append(
            HandHistorySample(
                palm_center_px=palm_center_px,
                index_tip_px=index_tip_px,
                pinch_distance=pinch_distance,
            )
        )
        if len(history) == EMPTY_HISTORY_LENGTH:
            return MotionFeatures(
                palm_dx=0.0,
                palm_dy=0.0,
                index_path_length=0.0,
                pinch_delta=0.0,
                index_bbox_width=0.0,
                index_bbox_height=0.0,
                start_end_distance=0.0,
                history_length=EMPTY_HISTORY_LENGTH,
            )

        first_sample = history[FIRST_POINT_INDEX]
        last_sample = history[-FRAME_WAIT_KEY_MS]
        palm_dx = last_sample.palm_center_px[FIRST_CLASSIFICATION_INDEX] - first_sample.palm_center_px[FIRST_CLASSIFICATION_INDEX]
        palm_dy = last_sample.palm_center_px[SECOND_POINT_INDEX] - first_sample.palm_center_px[SECOND_POINT_INDEX]
        pinch_delta = last_sample.pinch_distance - first_sample.pinch_distance

        index_points = [sample.index_tip_px for sample in history]
        path_length = 0.0
        for point_index in range(FRAME_WAIT_KEY_MS, len(index_points)):
            previous_point = index_points[point_index - FRAME_WAIT_KEY_MS]
            current_point = index_points[point_index]
            path_length += self._distance(previous_point, current_point)

        x_values = [point[FIRST_CLASSIFICATION_INDEX] for point in index_points]
        y_values = [point[SECOND_POINT_INDEX] for point in index_points]
        bbox_width = float(max(x_values) - min(x_values))
        bbox_height = float(max(y_values) - min(y_values))
        start_end_distance = self._distance(
            index_points[FIRST_POINT_INDEX],
            index_points[-FRAME_WAIT_KEY_MS],
        )

        return MotionFeatures(
            palm_dx=float(palm_dx),
            palm_dy=float(palm_dy),
            index_path_length=path_length,
            pinch_delta=pinch_delta,
            index_bbox_width=bbox_width,
            index_bbox_height=bbox_height,
            start_end_distance=start_end_distance,
            history_length=len(history),
        )

    def _detect_two_hands_x(
        self,
        action_hand: HandObservation | None,
        modifier_hand: HandObservation | None,
    ) -> tuple[bool, float]:
        """Return whether both hands form the termination X gesture."""

        if action_hand is None or modifier_hand is None:
            return False, UNKNOWN_GESTURE_CONFIDENCE

        right_wrist_x, _ = action_hand.landmarks_px[WRIST_ID]
        left_wrist_x, _ = modifier_hand.landmarks_px[WRIST_ID]
        right_index_tip = action_hand.landmarks_px[INDEX_FINGER_TIP_ID]
        right_index_pip = action_hand.landmarks_px[INDEX_FINGER_PIP_ID]
        left_index_tip = modifier_hand.landmarks_px[INDEX_FINGER_TIP_ID]
        left_index_pip = modifier_hand.landmarks_px[INDEX_FINGER_PIP_ID]

        wrists_crossed = right_wrist_x < left_wrist_x
        right_pointing_inward = (
            right_index_tip[FIRST_CLASSIFICATION_INDEX]
            < right_index_pip[FIRST_CLASSIFICATION_INDEX] - TERMINATION_POINTING_MARGIN
        )
        left_pointing_inward = (
            left_index_tip[FIRST_CLASSIFICATION_INDEX]
            > left_index_pip[FIRST_CLASSIFICATION_INDEX] + TERMINATION_POINTING_MARGIN
        )
        tips_close = (
            self._distance(right_index_tip, left_index_tip)
            <= TERMINATION_INDEX_DISTANCE_THRESHOLD
        )
        vertical_alignment = (
            abs(
                right_index_tip[SECOND_POINT_INDEX]
                - left_index_tip[SECOND_POINT_INDEX]
            )
            <= TERMINATION_VERTICAL_ALIGNMENT_THRESHOLD
        )
        index_extended = action_hand.finger_state.index and modifier_hand.finger_state.index

        if wrists_crossed and right_pointing_inward and left_pointing_inward and tips_close and vertical_alignment and index_extended:
            return True, TERMINATION_GESTURE_CONFIDENCE

        return False, UNKNOWN_GESTURE_CONFIDENCE

    @staticmethod
    def _map_modifier(modifier_gesture: str | None) -> str | None:
        """Map a left-hand gesture label to the active keyboard modifier name."""

        if modifier_gesture == GESTURE_INDEX_LEFT:
            return MODIFIER_CTRL
        if modifier_gesture == GESTURE_PEACE_LEFT:
            return MODIFIER_SHIFT
        if modifier_gesture == GESTURE_THREE_FINGERS_LEFT:
            return MODIFIER_ALT
        return MODIFIER_NONE

    @staticmethod
    def _find_hand_by_label(
        hands: list[HandObservation],
        handedness_label: str,
    ) -> HandObservation | None:
        """Return the first hand that matches the requested handedness label."""

        for hand in hands:
            if hand.handedness == handedness_label:
                return hand
        return None

    @staticmethod
    def _find_first_other_hand(
        hands: list[HandObservation],
        selected_hand: HandObservation,
    ) -> HandObservation | None:
        """Return the first hand that is not the provided selected hand."""

        for hand in hands:
            if hand is not selected_hand:
                return hand
        return None

    @staticmethod
    def _is_finger_extended(
        landmarks_px: list[tuple[int, int]],
        tip_id: int,
        pip_id: int,
    ) -> bool:
        """Return whether a non-thumb finger is extended upward (tip above pip)."""
        _, tip_y = landmarks_px[tip_id]
        _, pip_y = landmarks_px[pip_id]
        return tip_y < pip_y

    @staticmethod
    def _is_thumb_extended(
        landmarks_px: list[tuple[int, int]],
        handedness_label: str,
    ) -> bool:
        """Return whether the thumb is extended away from the palm."""
        thumb_tip_x, thumb_tip_y = landmarks_px[THUMB_TIP_ID]
        thumb_ip_x, thumb_ip_y = landmarks_px[THUMB_IP_ID]
        # Horizontal extension (left/right away from palm)
        if handedness_label == HAND_RIGHT:
            return thumb_tip_x < thumb_ip_x
        if handedness_label == HAND_LEFT:
            return thumb_tip_x > thumb_ip_x
        return abs(thumb_tip_x - thumb_ip_x) > PINCH_THRESHOLD // 2

    @staticmethod
    def _is_circle_motion(motion: MotionFeatures) -> bool:
        """Return whether recent index-tip movement resembles a circle."""

        return (
            motion.history_length >= GESTURE_HISTORY_FRAMES
            and motion.index_bbox_width >= CIRCLE_MIN_WIDTH
            and motion.index_bbox_height >= CIRCLE_MIN_HEIGHT
            and motion.start_end_distance <= CIRCLE_MAX_START_END_DISTANCE
            and motion.index_path_length >= CIRCLE_MIN_PATH_LENGTH
        )

    @staticmethod
    def _distance(
        first_point: tuple[int, int],
        second_point: tuple[int, int],
    ) -> float:
        """Return the Euclidean distance between two pixel coordinates."""

        return math.hypot(
            first_point[FIRST_CLASSIFICATION_INDEX] - second_point[FIRST_CLASSIFICATION_INDEX],
            first_point[SECOND_POINT_INDEX] - second_point[SECOND_POINT_INDEX],
        )
