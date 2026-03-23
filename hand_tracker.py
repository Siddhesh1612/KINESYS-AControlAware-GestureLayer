"""Basic MediaPipe hand tracking and gesture classification for KINESYS."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import mediapipe as mp

from config import (
    ACTION_CONFIDENCE_THRESHOLD,
    CLOSED_FIST_GESTURE_CONFIDENCE,
    CURSOR_INDICATOR_COLOR,
    CURSOR_INDICATOR_RADIUS,
    CURSOR_INDICATOR_THICKNESS,
    FRAME_WAIT_KEY_MS,
    GESTURE_CLOSED_FIST,
    GESTURE_INDEX_POINT,
    GESTURE_OPEN_PALM,
    GESTURE_PINCH,
    GESTURE_UNKNOWN,
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
    MAX_HANDS,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    MIDDLE_FINGER_PIP_ID,
    MIDDLE_FINGER_TIP_ID,
    OPEN_PALM_GESTURE_CONFIDENCE,
    PINCH_GESTURE_CONFIDENCE,
    PINCH_THRESHOLD,
    PINKY_PIP_ID,
    PINKY_TIP_ID,
    RING_FINGER_PIP_ID,
    RING_FINGER_TIP_ID,
    THUMB_IP_ID,
    THUMB_TIP_ID,
    UNKNOWN_GESTURE_CONFIDENCE,
    WRIST_ID,
)


LOGGER = logging.getLogger(__name__)

FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
TEXT_OFFSET_X = 10
TEXT_OFFSET_Y = 10
FIRST_CLASSIFICATION_INDEX = 0


@dataclass(slots=True)
class HandObservation:
    """Represents one tracked hand and its current gesture estimate."""

    handedness: str
    landmarks_px: list[tuple[int, int]]
    landmarks_norm: list[tuple[float, float, float]]
    gesture: str
    confidence: float


class HandTracker:
    """Track hands in a frame and classify a small gesture set for cursor control."""

    def __init__(self) -> None:
        """Initialize MediaPipe Hands and internal tracking buffers."""

        self._mp_hands = mp.solutions.hands
        self._mp_drawing = mp.solutions.drawing_utils
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=MAX_HANDS,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        )
        self._last_results = None
        self._last_hands: list[HandObservation] = []

    def process(self, frame: cv2.typing.MatLike) -> list[HandObservation]:
        """Process a BGR frame and return the currently tracked hands."""

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
                observation = self._build_observation(
                    hand_landmarks=hand_landmarks,
                    handedness_label=handedness_label,
                    frame_width=frame_width,
                    frame_height=frame_height,
                )
                hands.append(observation)

        self._last_results = results
        self._last_hands = hands
        return hands

    def draw_annotations(self, frame: cv2.typing.MatLike) -> cv2.typing.MatLike:
        """Draw MediaPipe landmarks and current gesture labels onto a frame."""

        if not self._last_results or not self._last_results.multi_hand_landmarks:
            return frame

        for index, hand_landmarks in enumerate(self._last_results.multi_hand_landmarks):
            self._mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                self._mp_hands.HAND_CONNECTIONS,
            )
            if index < len(self._last_hands):
                observation = self._last_hands[index]
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

                if observation.gesture in {GESTURE_INDEX_POINT, GESTURE_PINCH}:
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
        """Create a normalized hand observation from raw MediaPipe results."""

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

        gesture, confidence = self._classify_hand_gesture(
            handedness_label=handedness_label,
            landmarks_px=landmarks_px,
        )

        if confidence < ACTION_CONFIDENCE_THRESHOLD:
            gesture = GESTURE_UNKNOWN
            confidence = UNKNOWN_GESTURE_CONFIDENCE

        normalized_handedness = (
            handedness_label if handedness_label in {HAND_LEFT, HAND_RIGHT} else HAND_UNKNOWN
        )
        return HandObservation(
            handedness=normalized_handedness,
            landmarks_px=landmarks_px,
            landmarks_norm=landmarks_norm,
            gesture=gesture,
            confidence=confidence,
        )

    def _classify_hand_gesture(
        self,
        handedness_label: str,
        landmarks_px: list[tuple[int, int]],
    ) -> tuple[str, float]:
        """Classify a limited gesture set from pixel landmarks."""

        if self._is_pinch(landmarks_px):
            return GESTURE_PINCH, PINCH_GESTURE_CONFIDENCE

        index_extended = self._is_finger_extended(
            landmarks_px=landmarks_px,
            tip_id=INDEX_FINGER_TIP_ID,
            pip_id=INDEX_FINGER_PIP_ID,
        )
        middle_extended = self._is_finger_extended(
            landmarks_px=landmarks_px,
            tip_id=MIDDLE_FINGER_TIP_ID,
            pip_id=MIDDLE_FINGER_PIP_ID,
        )
        ring_extended = self._is_finger_extended(
            landmarks_px=landmarks_px,
            tip_id=RING_FINGER_TIP_ID,
            pip_id=RING_FINGER_PIP_ID,
        )
        pinky_extended = self._is_finger_extended(
            landmarks_px=landmarks_px,
            tip_id=PINKY_TIP_ID,
            pip_id=PINKY_PIP_ID,
        )
        thumb_extended = self._is_thumb_extended(
            landmarks_px=landmarks_px,
            handedness_label=handedness_label,
        )

        if index_extended and not middle_extended and not ring_extended and not pinky_extended:
            return GESTURE_INDEX_POINT, INDEX_GESTURE_CONFIDENCE

        if index_extended and middle_extended and ring_extended and pinky_extended and thumb_extended:
            return GESTURE_OPEN_PALM, OPEN_PALM_GESTURE_CONFIDENCE

        if not index_extended and not middle_extended and not ring_extended and not pinky_extended:
            return GESTURE_CLOSED_FIST, CLOSED_FIST_GESTURE_CONFIDENCE

        return GESTURE_UNKNOWN, UNKNOWN_GESTURE_CONFIDENCE

    @staticmethod
    def _is_finger_extended(
        landmarks_px: list[tuple[int, int]],
        tip_id: int,
        pip_id: int,
    ) -> bool:
        """Return whether a non-thumb finger is extended upward."""

        _, tip_y = landmarks_px[tip_id]
        _, pip_y = landmarks_px[pip_id]
        return tip_y < pip_y

    @staticmethod
    def _is_thumb_extended(
        landmarks_px: list[tuple[int, int]],
        handedness_label: str,
    ) -> bool:
        """Return whether the thumb is extended away from the hand."""

        thumb_tip_x, _ = landmarks_px[THUMB_TIP_ID]
        thumb_ip_x, _ = landmarks_px[THUMB_IP_ID]
        if handedness_label == HAND_RIGHT:
            return thumb_tip_x < thumb_ip_x
        if handedness_label == HAND_LEFT:
            return thumb_tip_x > thumb_ip_x
        return False

    @staticmethod
    def _is_pinch(landmarks_px: list[tuple[int, int]]) -> bool:
        """Return whether the thumb and index fingertips are close enough to count as a pinch."""

        thumb_x, thumb_y = landmarks_px[THUMB_TIP_ID]
        index_x, index_y = landmarks_px[INDEX_FINGER_TIP_ID]
        delta_x = thumb_x - index_x
        delta_y = thumb_y - index_y
        distance_squared = (delta_x * delta_x) + (delta_y * delta_y)
        return distance_squared <= (PINCH_THRESHOLD * PINCH_THRESHOLD)
