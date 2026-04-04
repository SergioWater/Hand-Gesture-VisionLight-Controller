"""
main.py — Hand Gesture Light Controller
========================================
State machine that maps hand gestures to smart LED bulb control.

Gesture flow:
  1. Circle right hand                  → DIALING  (real-time color preview)
  2. Close both fists (raised)          → CONFIRMED (color set permanently)

Bulb layout (ceiling, viewed from below):
    WB1 (front-left)  ---- WB2 (front-right)
             |    (user)    |
    WB4 (back-left)   ---- WB5 (back-right)

Special override: double clap toggles all bulbs on/off.
"""

import cv2 as cv
import time
import colorsys

from open_camera import Camera
from hand_detection import HandDetection
from gesture_engine import GestureEngine
from color_picker_ui import ColorPickerUI
from bulb_controller import BulbController


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# State names
STATE_IDLE      = "IDLE"
STATE_DIALING   = "DIALING"
STATE_CONFIRMED = "CONFIRMED"

ALL_BULBS = {"WB1"}

# Timeout thresholds (in frames)
DIALING_TIMEOUT_FRAMES = 30   # frames without circle before DIALING → IDLE

# Minimum seconds between bulb color updates while dialing (avoids flooding)
DIAL_COLOR_COOLDOWN_SEC = 0.3

# Seconds to hold CONFIRMED state before returning to IDLE
CONFIRMED_HOLD_SEC = 1.0

# Display
WINDOW_NAME  = "Hand Gesture Light Controller"


# ---------------------------------------------------------------------------
# Startup banner
# ---------------------------------------------------------------------------

BANNER = r"""
╔══════════════════════════════════════════════════════════╗
║         HAND GESTURE LIGHT CONTROLLER — v2.0            ║
╠══════════════════════════════════════════════════════════╣
║  Gesture Flow                                            ║
║  ─────────────────────────────────────────────────  ║
║  1. RIGHT CIRCLE→  Dial the color (live preview)        ║
║  2. BOTH FISTS  →  Confirm & set color permanently      ║
║                                                          ║
║  DOUBLE CLAP (any time)  →  Toggle bulbs ON / OFF       ║
║  Q key  →  Quit          R key  →  Reset to IDLE        ║
╚══════════════════════════════════════════════════════════╝
"""


# ---------------------------------------------------------------------------
# StateMachine
# ---------------------------------------------------------------------------

class StateMachine:
    """
    Manages the IDLE → AIMING → LOCKED → DIALING → CONFIRMED state graph.

    Each call to update(gesture_data) advances the machine one frame and
    returns the current state string so the UI layer can react.
    """

    def __init__(self, bulb_controller: BulbController):
        self.bc = bulb_controller

        self.state: str = STATE_IDLE

        # The single bulb we are controlling
        self.locked_bulb: str = "WB1"

        # Current hue for the dialing state (0.0–1.0)
        self.current_hue: float = 0.0

        # Frame-level timeout counters
        self._no_gesture_frames: int = 0 # counts frames without gesture in DIALING

        # Time when CONFIRMED state was entered
        self._confirmed_at: float = 0.0

        # Timestamp of the last color update sent to the bulb while dialing
        self._last_dial_update: float = 0.0

        # Track bulb power state for BOTH_FISTS toggle
        self._bulbs_on: bool = True

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(self, gesture_data: dict) -> str:
        """
        Process one frame of gesture data and advance the state machine.

        Args:
            gesture_data: dict from GestureEngine with keys:
                gesture, dial_hue, dial_active

        Returns:
            Current state string after processing.
        """
        gesture       = gesture_data.get("gesture", "NONE")
        dial_hue      = gesture_data.get("dial_hue")
        dial_active   = gesture_data.get("dial_active", False)

        # ------------------------------------------------------------------
        # Global override: DOUBLE_CLAP toggles bulbs on/off
        # ------------------------------------------------------------------
        if gesture == "DOUBLE_CLAP":
            if self._bulbs_on:
                self._turn_off_all()
                self._bulbs_on = False
            else:
                # Turn on by restoring last dialed color (or white)
                r, g, b = self._hue_to_rgb(self.current_hue) if self.current_hue else (255, 255, 255)
                self.bc.set_color({self.locked_bulb}, r, g, b)
                self._bulbs_on = True
            self._reset()
            return self.state

        # ------------------------------------------------------------------
        # State-specific logic
        # ------------------------------------------------------------------
        if self.state == STATE_IDLE:
            self._handle_idle(gesture, dial_active)

        elif self.state == STATE_DIALING:
            self._handle_dialing(gesture, dial_hue, dial_active)

        elif self.state == STATE_CONFIRMED:
            self._handle_confirmed()

        return self.state

    def reset(self):
        """Force reset to IDLE without touching bulbs."""
        self._reset()

    # ------------------------------------------------------------------
    # State handlers
    # ------------------------------------------------------------------

    def _handle_idle(self, gesture: str, dial_active: bool):
        if gesture == "COLOR_DIAL" and dial_active:
            self._no_gesture_frames = 0
            self._transition(STATE_DIALING)

    def _handle_dialing(self, gesture: str, dial_hue: float | None, dial_active: bool):
        if gesture == "BOTH_FISTS":
            # Confirm: permanently set the color and flash the UI
            self._apply_confirmed_color()
            self._confirmed_at = time.monotonic()
            self._transition(STATE_CONFIRMED)
            return

        if not dial_active or gesture == "NONE":
            # Right hand disappeared — increment timeout
            self._no_gesture_frames += 1
            if self._no_gesture_frames > DIALING_TIMEOUT_FRAMES:
                self._transition(STATE_IDLE)
            return

        # Actively dialing: update hue and push to bulb at cooldown rate
        self._no_gesture_frames = 0
        if dial_hue is not None:
            self.current_hue = dial_hue

        now = time.monotonic()
        if now - self._last_dial_update >= DIAL_COLOR_COOLDOWN_SEC:
            self._push_dial_color_to_bulb()
            self._last_dial_update = now

    def _handle_confirmed(self):
        elapsed = time.monotonic() - self._confirmed_at
        if elapsed >= CONFIRMED_HOLD_SEC:
            # Return to idle after the hold period
            self._reset()

    # ------------------------------------------------------------------
    # Bulb helpers
    # ------------------------------------------------------------------

    def _push_dial_color_to_bulb(self):
        """Convert current hue to RGB and update the locked bulb live."""
        if not self.locked_bulb:
            return
        r, g, b = self._hue_to_rgb(self.current_hue)
        self.bc.set_color({self.locked_bulb}, r, g, b)

    def _apply_confirmed_color(self):
        """Permanently write the selected color to the locked bulb."""
        if not self.locked_bulb:
            return
        r, g, b = self._hue_to_rgb(self.current_hue)
        self.bc.set_color({self.locked_bulb}, r, g, b)

    def _turn_off_all(self):
        self.bc.turn_off(ALL_BULBS)

    @staticmethod
    def _hue_to_rgb(hue: float) -> tuple[int, int, int]:
        """Convert a hue (0–1) at full saturation/value to an (R, G, B) tuple (0–255)."""
        r_f, g_f, b_f = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        return int(r_f * 255), int(g_f * 255), int(b_f * 255)

    # ------------------------------------------------------------------
    # Transition helpers
    # ------------------------------------------------------------------

    def _transition(self, new_state: str):
        """Log and execute a state transition."""
        print(f"  [StateMachine] {self.state} → {new_state}")
        self.state = new_state

        # Clear targeting info when returning to idle
        if new_state == STATE_IDLE:
            self.current_hue = 0.0
            self._no_gesture_frames = 0

    def _reset(self):
        self._transition(STATE_IDLE)


# ---------------------------------------------------------------------------
# UI sync helper
# ---------------------------------------------------------------------------

def sync_ui(ui: ColorPickerUI, sm: StateMachine):
    """
    Push state-machine values into the UI object so the overlay reflects
    the current gesture state each frame.
    """
    ui.state          = sm.state
    ui.target_bulb    = None
    ui.locked_bulb    = sm.locked_bulb
    ui.show_color_wheel = (sm.state == STATE_DIALING)

    # Mirror the live hue into the UI's hue value so the color wheel and
    # hue bar stay in sync during DIALING
    if sm.state == STATE_DIALING:
        ui.hue = sm.current_hue


# ---------------------------------------------------------------------------
# Status text helper
# ---------------------------------------------------------------------------

def get_status_text(state: str, locked_bulb: str | None, current_hue: float) -> str:
    """Return a human-readable status line for the current state."""
    if state == STATE_IDLE:
        return "Circle right hand to start dialing"
    elif state == STATE_DIALING:
        hue_deg = int(current_hue * 360)
        return f"Hue {hue_deg}°  —  Close both fists to confirm"
    elif state == STATE_CONFIRMED:
        return "Color confirmed!"
    return ""


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    print(BANNER)

    # --- Hardware / module initialisation ----------------------------------
    cam = Camera()                      # uses open_camera.py (device_index=1, fallback 0)
    hand_detector = HandDetection()     # uses hand_detection.py
    bc = BulbController()               # uses bulb_controller.py + bulbs.json

    # Read one test frame to discover actual resolution
    ret, test_frame = cam.cap.read()
    if ret:
        fh, fw = test_frame.shape[:2]
    else:
        fw, fh = 640, 480

    engine = GestureEngine(fw, fh)      # needs frame dimensions for quadrant mapping
    ui = ColorPickerUI(fw, fh)
    sm = StateMachine(bc)

    print("[main] All systems initialised. Starting capture loop…")
    print("[main] Press Q to quit, R to reset.\n")

    try:
        while True:
            # ---------------------------------------------------------------
            # 1. Grab frame
            # ---------------------------------------------------------------
            ret, frame = cam.cap.read()
            if not ret:
                print("[main] WARNING: empty frame received — skipping")
                continue

            # Mirror for natural feel (left hand appears on left)
            frame = cv.flip(frame, 1)

            # ---------------------------------------------------------------
            # 2. Hand detection + gesture recognition
            # ---------------------------------------------------------------
            frame, hands_data = hand_detector.detect_hands(frame)
            gesture_data = engine.recognize(hands_data)

            # ---------------------------------------------------------------
            # 3. Advance state machine
            # ---------------------------------------------------------------
            state = sm.update(gesture_data)

            # ---------------------------------------------------------------
            # 4. Sync UI with state machine values
            # ---------------------------------------------------------------
            sync_ui(ui, sm)
            ui.status_text = get_status_text(state, sm.locked_bulb, sm.current_hue)

            # ---------------------------------------------------------------
            # 5. Draw UI overlay onto frame
            # ---------------------------------------------------------------
            frame = ui.draw(frame)

            # Flash a green border when CONFIRMED
            if state == STATE_CONFIRMED:
                h, w = frame.shape[:2]
                thickness = 12
                cv.rectangle(
                    frame,
                    (thickness // 2, thickness // 2),
                    (w - thickness // 2, h - thickness // 2),
                    (0, 255, 0),  # BGR green
                    thickness,
                )

            # ---------------------------------------------------------------
            # 6. Display
            # ---------------------------------------------------------------
            cv.imshow(WINDOW_NAME, frame)

            # ---------------------------------------------------------------
            # 7. Key handling
            # ---------------------------------------------------------------
            key = cv.waitKey(1) & 0xFF
            if key == ord("q") or key == ord("Q"):
                print("[main] Q pressed — quitting.")
                break
            elif key == ord("r") or key == ord("R"):
                print("[main] R pressed — resetting to IDLE.")
                sm.reset()

    finally:
        print("[main] Cleaning up…")
        bc.cleanup()
        cam.cap.release()
        cv.destroyAllWindows()
        print("[main] Goodbye.")


if __name__ == "__main__":
    main()
