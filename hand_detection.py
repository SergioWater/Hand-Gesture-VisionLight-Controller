import cv2 as cv
import mediapipe as mp
import math


class HandDetection:
    """Detects hands and provides landmark data + gesture recognition."""

    # MediaPipe hand landmark indices
    WRIST = 0
    THUMB_TIP = 4
    INDEX_TIP = 8
    MIDDLE_TIP = 12
    RING_TIP = 16
    PINKY_TIP = 20
    INDEX_MCP = 5
    MIDDLE_MCP = 9
    RING_MCP = 13
    PINKY_MCP = 17
    INDEX_PIP = 6
    MIDDLE_PIP = 10
    RING_PIP = 14
    PINKY_PIP = 18

    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )
        self.mp_draw = mp.solutions.drawing_utils

    def detect_hands(self, frame):
        """Process frame and return annotated frame + hand data list."""
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        hands_data = []

        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Draw skeleton
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )

                h, w, _ = frame.shape
                landmarks = []
                x_min, y_min = w, h
                x_max, y_max = 0, 0

                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    landmarks.append((x, y))
                    if x < x_min:
                        x_min = x
                    if x > x_max:
                        x_max = x
                    if y < y_min:
                        y_min = y
                    if y > y_max:
                        y_max = y

                # Bounding box
                cv.rectangle(
                    frame,
                    (x_min - 20, y_min - 20),
                    (x_max + 20, y_max + 20),
                    (0, 255, 0),
                    2,
                )

                # Determine handedness
                handedness = "Unknown"
                if results.multi_handedness:
                    handedness = results.multi_handedness[idx].classification[
                        0
                    ].label  # "Left" or "Right"

                hand_info = {
                    "landmarks": landmarks,
                    "bbox": (x_min, y_min, x_max, y_max),
                    "handedness": handedness,
                }
                hands_data.append(hand_info)

        return frame, hands_data

    @staticmethod
    def is_finger_extended(landmarks, tip_idx, pip_idx):
        """Check if a finger is extended (tip is above pip in y)."""
        return landmarks[tip_idx][1] < landmarks[pip_idx][1]

    @staticmethod
    def distance(p1, p2):
        """Euclidean distance between two points."""
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
