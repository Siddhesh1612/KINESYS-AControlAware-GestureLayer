import cv2
import mediapipe as mp
import math

class HandTracker:
    def __init__(self, max_hands=2):
        import mediapipe.python.solutions.hands as mp_hands
        import mediapipe.python.solutions.drawing_utils as mp_draw
        self.mp_hands = mp_hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp_draw

    def process(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(img_rgb)
        
        hands_info = []
        if result.multi_hand_landmarks and result.multi_handedness:
            for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
                # Note: MediaPipe labels are mirrored for selfie-view by default if not flipped.
                # Assuming standard webcam feed without flip.
                label = handedness.classification[0].label
                h, w, c = frame.shape
                landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]
                
                hands_info.append({
                    "label": label,
                    "landmarks": landmarks,
                    "raw": hand_landmarks
                })
        return hands_info, result

    def get_draw_canvas_position(self, hand_info):
        """Index finger tip is at index 8."""
        return hand_info["landmarks"][8]

    def is_pinching(self, hand_info, threshold=30):
        """Thumb tip (4) and Index tip (8) distance."""
        lm = hand_info["landmarks"]
        dist = math.hypot(lm[4][0] - lm[8][0], lm[4][1] - lm[8][1])
        return dist < threshold

    def is_two_fingers_up(self, hand_info):
        """Index (8) and Middle (12) are up. Ring (16) and Pinky (20) are down."""
        lm = hand_info["landmarks"]
        # Y-coordinate lower value means higher on screen
        index_up = lm[8][1] < lm[6][1]
        middle_up = lm[12][1] < lm[10][1]
        ring_down = lm[16][1] > lm[14][1]
        pinky_down = lm[20][1] > lm[18][1]
        return index_up and middle_up and ring_down and pinky_down

    def draw_hands(self, frame, result):
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
