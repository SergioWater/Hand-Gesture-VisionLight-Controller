"""
gesture_engine.py
-----------------
Stateless gesture recogniser for the Hand Gesture Light Controller.

Responsibilities (per-frame):
  - Detect COLOR_DIAL  : right hand moving in a circle; maps current angle → hue
  - Detect DOUBLE_CLAP : two claps within a short window → toggle on/off
  - Detect BOTH_FISTS  : both hands raised with fists closed → confirm color

This module does NOT own the IDLE→DIALING→CONFIRMED state machine.
That logic lives in main.py.  Here we only report what we see right now.

Usage:
    engine = GestureEngine(frame_width=640, frame_height=480)
    result = engine.recognize(hands_data)
    # result is a dict – see recognize() docstring for schema

Depends on hand_detection.py (HandDetection helper class).
Frame is expected to already be flipped/mirrored, so MediaPipe's "Left"
label corresponds to the user's physical left hand.
"""

import math
import collections
from hand_detection import HandDetection as HD

# ---------------------------------------------------------------------------
# Tuning constants – adjust without touching algorithm logic
# ---------------------------------------------------------------------------

# Circular-motion tracking
_CIRCLE_WINDOW      = 15    # rolling deque length for right-hand positions
_CIRCLE_MIN_RADIUS  = 20    # minimum mean-radius (px) to consider "circular"
_CIRCLE_MIN_POINTS  = 5    # need at least this many points before analysing

# Clap detection
_CLAP_FAR_DIST      = 200   # wrist distance (px) considered "hands apart"
_CLAP_NEAR_DIST     = 80    # wrist distance (px) considered "clapped"
_CLAP_WINDOW        = 10    # frames within which the transition must happen
_CLAP_COOLDOWN      = 20    # frames to ignore clap after one is fired

# Double-clap detection: two individual claps must occur within this window.
_DOUBLE_CLAP_WINDOW = 20    # ~0.67s at 30 fps

# Fist detection: all four non-thumb fingers must be folded.
# Allow 0 extended fingers for a strict fist.  Using 0 here avoids
# ambiguity with a 1-finger point being mis-classified as a fist.
_FIST_MAX_EXTENDED  = 0     # zero fingers extended = closed fist

# Global gesture cooldown – after any gesture fires (except COLOR_DIAL),
# suppress new gestures for this many frames to prevent rapid re-triggers.
_GESTURE_COOLDOWN   = 15

# BOTH_FISTS cooldown – prevents rapid confirm.
_BOTH_FISTS_COOLDOWN = 3


class GestureEngine:
    """
    Per-frame gesture recogniser.  Instantiate once; call recognize() every frame.
    """

    def __init__(self, frame_width: int = 640, frame_height: int = 480):
        """
        Parameters
        ----------
        frame_width  : pixel width of the video frame
        frame_height : pixel height of the video frame
        """
        self.frame_width  = frame_width
        self.frame_height = frame_height

        # --- Circular motion state ---
        # Rolling window of (x, y) positions for the right hand's index tip
        self._circle_positions: collections.deque = collections.deque(
            maxlen=_CIRCLE_WINDOW
        )
        # Last computed hue so we can return a stable value even mid-rotation
        self._last_hue: float = 0.0

        # --- Clap detection state ---
        # Rolling window of recent inter-wrist distances (one value per frame)
        self._clap_dist_history: collections.deque = collections.deque(
            maxlen=_CLAP_WINDOW
        )
        # Cooldown counter; when > 0 we don't fire another clap
        self._clap_cooldown: int = 0

        # --- Double-clap detection state ---
        # True when we've seen one clap and are waiting for the second
        self._pending_clap: bool = False
        # Frames remaining in the double-clap window
        self._pending_clap_timer: int = 0

        # --- Global gesture cooldown ---
        # When > 0, suppress all gestures except COLOR_DIAL
        self._gesture_cooldown: int = 0

        # --- BOTH_FISTS cooldown ---
        # Separate cooldown to prevent rapid confirm
        self._both_fists_cooldown: int = 0

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def recognize(self, hands_data: list) -> dict:
        """
        Analyse one frame's worth of hand-detection data and return a gesture dict.

        Parameters
        ----------
        hands_data : list of dicts produced by HandDetection.
            Each dict has:
              "landmarks"   – list of 21 (x, y) tuples (pixel coords)
              "bbox"        – (x_min, y_min, x_max, y_max)
              "handedness"  – "Left" or "Right"
                              (already corrected for mirror flip in main.py)

        Returns
        -------
        dict with keys:
          "gesture"         – str  : "NONE" | "COLOR_DIAL" | "DOUBLE_CLAP" | "BOTH_FISTS"
          "dial_hue"        – float or None   : 0.0–1.0 from circular motion
          "dial_active"     – bool            : True when circle detected
        """
        result = {
            "gesture":         "NONE",
            "dial_hue":        None,
            "dial_active":     False,
        }
        if not hands_data:
            # No hands visible – reset transient state to avoid stale data
            self._circle_positions.clear()
            self._clap_dist_history.clear()
            if self._clap_cooldown > 0:
                self._clap_cooldown -= 1
            if self._gesture_cooldown > 0:
                self._gesture_cooldown -= 1
            if self._both_fists_cooldown > 0:
                self._both_fists_cooldown -= 1
            # Tick pending-clap timer even with no hands visible
            if self._pending_clap and self._pending_clap_timer > 0:
                self._pending_clap_timer -= 1
                if self._pending_clap_timer == 0:
                    self._pending_clap = False  # window expired, discard
            return result

        # Separate hands by handedness
        left_hand  = self._get_hand(hands_data, "Left")
        right_hand = self._get_hand(hands_data, "Right")

        # Tick down cooldowns each frame
        if self._clap_cooldown > 0:
            self._clap_cooldown -= 1
        if self._gesture_cooldown > 0:
            self._gesture_cooldown -= 1
        if self._both_fists_cooldown > 0:
            self._both_fists_cooldown -= 1

        # Tick pending-clap timer
        if self._pending_clap and self._pending_clap_timer > 0:
            self._pending_clap_timer -= 1
            if self._pending_clap_timer == 0:
                self._pending_clap = False  # window expired, discard

        # ------------------------------------------------------------------
        # 1. Double-clap detection (highest priority)
        #    Individual claps feed into a two-stage detector.
        #    DOUBLE_CLAP is exempt from global gesture cooldown.
        # ------------------------------------------------------------------
        if left_hand and right_hand:
            clap_fired = self._update_clap(
                left_hand["landmarks"], right_hand["landmarks"]
            )
            if clap_fired:
                if self._pending_clap:
                    # Second clap within window → DOUBLE_CLAP!
                    self._pending_clap = False
                    self._pending_clap_timer = 0
                    result["gesture"] = "DOUBLE_CLAP"
                    self._gesture_cooldown = _GESTURE_COOLDOWN
                    return result
                else:
                    # First clap – start the double-clap window
                    self._pending_clap = True
                    self._pending_clap_timer = _DOUBLE_CLAP_WINDOW
        else:
            # If one hand disappears, reset clap history to avoid false trigger
            self._clap_dist_history.clear()

        # ------------------------------------------------------------------
        # 2. BOTH_FISTS detection (both hands raised with closed fists)
        #    Has its own cooldown; also respects global cooldown.
        # ------------------------------------------------------------------
        if left_hand and right_hand and self._both_fists_cooldown == 0:
            if self._is_both_fists_raised(left_hand["landmarks"],
                                          right_hand["landmarks"]):
                result["gesture"] = "BOTH_FISTS"
                self._both_fists_cooldown = _BOTH_FISTS_COOLDOWN
                self._gesture_cooldown = _GESTURE_COOLDOWN
                return result

        # ------------------------------------------------------------------
        # 3. Right-hand COLOR_DIAL (circular motion + hue)
        #    COLOR_DIAL is exempt from the global gesture cooldown.
        # ------------------------------------------------------------------
        if right_hand:
            dial_hue, dial_active = self._update_circle(right_hand["landmarks"])
            result["dial_hue"]    = dial_hue
            result["dial_active"] = dial_active
            if dial_active and result["gesture"] == "NONE":
                result["gesture"] = "COLOR_DIAL"
        else:
            # Right hand gone – keep deque so brief occlusion doesn't reset hue
            # but do trim it so stale positions don't pollute a fresh appearance
            self._circle_positions.clear()

        # ------------------------------------------------------------------
        # Global cooldown gate: suppress non-exempt gestures
        # ------------------------------------------------------------------
        if self._gesture_cooldown > 0 and result["gesture"] not in ("COLOR_DIAL", "DOUBLE_CLAP", "NONE"):
            result["gesture"] = "NONE"

        return result

    # -----------------------------------------------------------------------
    # Gesture helpers
    # -----------------------------------------------------------------------

    def _count_extended(self, landmarks: list) -> int:
        """
        Returns the number of non-thumb fingers that are extended.
        Order: index, middle, ring, pinky.
        """
        pairs = [
            (HD.INDEX_TIP,  HD.INDEX_PIP),
            (HD.MIDDLE_TIP, HD.MIDDLE_PIP),
            (HD.RING_TIP,   HD.RING_PIP),
            (HD.PINKY_TIP,  HD.PINKY_PIP),
        ]
        return sum(
            1 for tip, pip in pairs
            if HD.is_finger_extended(landmarks, tip, pip)
        )

    def _is_fist(self, landmarks: list) -> bool:
        """
        Returns True when at most _FIST_MAX_EXTENDED fingers are extended
        (a closed fist may still have one finger slightly above threshold).
        """
        return self._count_extended(landmarks) <= _FIST_MAX_EXTENDED

    def _is_both_fists_raised(self, left_lm: list, right_lm: list) -> bool:
        """
        Returns True when both hands are closed fists AND both wrists are
        in the upper half of the frame (y < frame_height / 2).
        """
        if not self._is_fist(left_lm) or not self._is_fist(right_lm):
            return False
        half_h = self.frame_height / 2
        left_wrist_y  = left_lm[HD.WRIST][1]
        right_wrist_y = right_lm[HD.WRIST][1]
        return left_wrist_y < half_h and right_wrist_y < half_h

    # -----------------------------------------------------------------------
    # Circular motion / hue dial
    # -----------------------------------------------------------------------

    def _update_circle(self, landmarks: list) -> tuple:
        """
        Push the right hand's index-tip position into the rolling buffer and
        compute the current circular-motion state.

        Returns
        -------
        (hue, dial_active)
          hue         : float 0.0–1.0 mapped from the current angle
          dial_active : bool, True when the buffer looks like circular motion
        """
        tip = landmarks[HD.INDEX_TIP]
        self._circle_positions.append(tip)

        n = len(self._circle_positions)
        if n < _CIRCLE_MIN_POINTS:
            return self._last_hue, False

        # Centroid of the rolling window
        cx = sum(p[0] for p in self._circle_positions) / n
        cy = sum(p[1] for p in self._circle_positions) / n

        # Mean radius – if the hand isn't really moving in a circle this is tiny
        radii = [
            math.sqrt((p[0] - cx) ** 2 + (p[1] - cy) ** 2)
            for p in self._circle_positions
        ]
        mean_radius = sum(radii) / len(radii)

        dial_active = mean_radius >= _CIRCLE_MIN_RADIUS

        # Current angle of the latest tip position relative to centroid
        latest = self._circle_positions[-1]
        angle  = math.atan2(latest[1] - cy, latest[0] - cx)  # –π … +π

        # Normalise to 0.0–1.0 hue
        hue = (angle + math.pi) / (2 * math.pi)   # 0.0 at –π, 1.0 at +π
        hue = max(0.0, min(1.0, hue))              # clamp for safety

        self._last_hue = hue
        return hue, dial_active

    # -----------------------------------------------------------------------
    # Clap detection
    # -----------------------------------------------------------------------

    def _update_clap(self, left_lm: list, right_lm: list) -> bool:
        """
        Append the current inter-wrist distance to the history buffer and
        check whether a clap transition (far→near within the window) occurred.

        Returns True exactly once per clap event (then starts cooldown).
        """
        if self._clap_cooldown > 0:
            # Still in cooldown – record distance but don't fire
            dist = HD.distance(left_lm[HD.WRIST], right_lm[HD.WRIST])
            self._clap_dist_history.append(dist)
            return False

        dist = HD.distance(left_lm[HD.WRIST], right_lm[HD.WRIST])
        self._clap_dist_history.append(dist)

        history = list(self._clap_dist_history)
        if len(history) < 2:
            return False

        # A clap is: the earliest sample in the window was far, and the
        # latest sample is near.  We require at least one "far" reading
        # somewhere in the first half of the buffer and a "near" reading
        # at the end, so a slow drift doesn't trigger a false clap.
        recent_near = history[-1] < _CLAP_NEAR_DIST
        had_far     = any(d > _CLAP_FAR_DIST for d in history[:-1])

        if recent_near and had_far:
            self._clap_cooldown = _CLAP_COOLDOWN
            self._clap_dist_history.clear()   # reset so no double-fire
            return True

        return False

    # -----------------------------------------------------------------------
    # Internal utility
    # -----------------------------------------------------------------------

    @staticmethod
    def _get_hand(hands_data: list, handedness: str) -> dict | None:
        """
        Return the first entry in hands_data whose 'handedness' matches,
        or None if not present.
        """
        for hand in hands_data:
            if hand.get("handedness") == handedness:
                return hand
        return None
