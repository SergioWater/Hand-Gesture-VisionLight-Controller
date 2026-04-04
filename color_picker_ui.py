"""
color_picker_ui.py — Color Picker UI Overlay
=============================================
Renders all interactive UI elements as OpenCV overlays on a live camera frame.

Elements
--------
  Existing (preserved):
    • Hue bar          — horizontal gradient bar for hue selection
    • Brightness slider— vertical slider (0-255)
    • Saturation slider— vertical slider (0.0-1.0)
    • Preview swatch   — filled rectangle showing the current HSV color
    • Bulb buttons     — four clickable buttons (WB1, WB2, WB4, WB5)
    • Status text      — single-line message at the bottom of the frame

  New:
    • State indicator  — top-center panel showing current FSM state
    • Color wheel      — circular hue ring when in DIALING mode
    • Bulb glow        — highlight effect on the targeted / locked bulb button
"""

import cv2 as cv
import colorsys
import math
import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Colour palette helpers
# ---------------------------------------------------------------------------

def _hsv_to_bgr(h: float, s: float, v: float) -> tuple[int, int, int]:
    """Convert HSV (each 0–1) to an OpenCV BGR int tuple."""
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(b * 255), int(g * 255), int(r * 255))


def _hue_to_bgr(hue: float) -> tuple[int, int, int]:
    """Hue (0–1) at full saturation/value → BGR."""
    return _hsv_to_bgr(hue, 1.0, 1.0)


# ---------------------------------------------------------------------------
# State display configuration
# ---------------------------------------------------------------------------

# Maps FSM state name → (label text, BGR colour)
STATE_CONFIG: dict[str, tuple[str, tuple[int, int, int]]] = {
    "IDLE":      ("IDLE",      (130, 130, 130)),   # gray
    "AIMING":    ("AIMING",    (0,   220, 220)),   # yellow (BGR: low B, high G+R)
    "LOCKED":    ("LOCKED",    (0,   140, 255)),   # orange
    "DIALING":   ("DIALING",   (200, 180, 0)),     # cyan
    "CONFIRMED": ("CONFIRMED", (0,   200, 80)),    # green
}

# Correct BGR values for state colours (OpenCV uses BGR)
_STATE_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    "IDLE":      (130, 130, 130),   # gray
    "AIMING":    (0,   215, 255),   # yellow   (B=0, G=215, R=255)
    "LOCKED":    (0,   140, 255),   # orange   (B=0, G=140, R=255)
    "DIALING":   (255, 220, 0),     # cyan     (B=255, G=220, R=0)
    "CONFIRMED": (50,  205, 50),    # green    (B=50,  G=205, R=50)
}


# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------

# --- Hue bar ---
HUE_BAR_X      = 30
HUE_BAR_Y      = 50
HUE_BAR_W      = 300
HUE_BAR_H      = 24

# --- Brightness slider ---
BRIGHT_X       = 360
BRIGHT_Y_TOP   = 40
BRIGHT_H       = 120
BRIGHT_W       = 24

# --- Saturation slider ---
SAT_X          = 410
SAT_Y_TOP      = 40
SAT_H          = 120
SAT_W          = 24

# --- Preview swatch ---
SWATCH_X       = 460
SWATCH_Y       = 40
SWATCH_W       = 80
SWATCH_H       = 50

# --- Bulb buttons ---
# Laid out to mirror the physical ceiling arrangement:
#   WB1 (front-left)  WB2 (front-right)
#   WB4 (back-left)   WB5 (back-right)
BULB_BTN_W   = 70
BULB_BTN_H   = 30
BULB_BTN_GAP = 10

# Top-left of the 2×2 bulb button grid
BULB_GRID_X  = 30
BULB_GRID_Y  = 110   # below the hue bar

# (name, col, row)  — col/row are 0-indexed
BULB_GRID: list[tuple[str, int, int]] = [
    ("WB1", 0, 0),   # front-left
    ("WB2", 1, 0),   # front-right
    ("WB4", 0, 1),   # back-left
    ("WB5", 1, 1),   # back-right
]

# --- Status text ---
STATUS_Y_OFFSET = 30   # px from bottom of frame

# --- Color wheel (DIALING overlay) ---
WHEEL_CENTER_X_FRAC = 0.75   # fraction of frame width
WHEEL_CENTER_Y_FRAC = 0.55   # fraction of frame height
WHEEL_OUTER_R = 150
WHEEL_INNER_R = 100
WHEEL_SEGMENTS = 360          # one arc per degree for smooth gradient


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ColorPickerUI:
    """
    OpenCV-based UI overlay for the Hand Gesture Light Controller.

    Usage (each frame):
        ui.state      = sm.state        # set by main.py
        ui.target_bulb = sm.target_bulb  # set by main.py
        ui.locked_bulb = sm.locked_bulb  # set by main.py
        ui.show_color_wheel = (sm.state == "DIALING")
        ui.status_text = "Some message"
        frame = ui.draw(frame)
    """

    def __init__(self, frame_width: int, frame_height: int):
        self.frame_width  = frame_width
        self.frame_height = frame_height

        # --- Colour values ---
        self.hue        = 0.0    # 0.0 – 1.0
        self.saturation = 1.0    # 0.0 – 1.0
        self.brightness = 255    # 0   – 255

        # --- State (set by main.py each frame) ---
        self.state:           str           = "IDLE"
        self.target_bulb:     Optional[str] = None
        self.locked_bulb:     Optional[str] = None
        self.show_color_wheel: bool         = False

        # --- Active bulb set (toggled by bulb buttons) ---
        self.active_bulbs: set[str] = {"WB1", "WB2", "WB4", "WB5"}

        # --- Status message ---
        self.status_text: str = "Point at a bulb to begin"

        # Pre-compute bulb button rectangles for hit-testing
        self._bulb_rects: dict[str, tuple[int, int, int, int]] = {}
        self._build_bulb_rects()

        # Pre-render the static hue bar gradient image
        self._hue_bar_img = self._make_hue_bar(HUE_BAR_W, HUE_BAR_H)

    # -----------------------------------------------------------------------
    # Public read accessors
    # -----------------------------------------------------------------------

    def get_rgb(self) -> tuple[int, int, int]:
        """Return the current colour as (R, G, B) each 0–255."""
        r_f, g_f, b_f = colorsys.hsv_to_rgb(
            self.hue, self.saturation, self.brightness / 255.0
        )
        return int(r_f * 255), int(g_f * 255), int(b_f * 255)

    def get_hsv_display(self) -> tuple[int, int, int]:
        """Return HSV in OpenCV display convention: H∈[0,179], S∈[0,255], V∈[0,255]."""
        return (
            int(self.hue * 179),
            int(self.saturation * 255),
            self.brightness,
        )

    # -----------------------------------------------------------------------
    # Hit-testing and interaction
    # -----------------------------------------------------------------------

    def hit_test(self, x: int, y: int) -> Optional[str]:
        """
        Return which UI element the point (x, y) lands on, or None.

        Return values: "hue_bar", "brightness_slider", "saturation_slider",
                       "WB1", "WB2", "WB4", "WB5"
        """
        # Hue bar
        if (HUE_BAR_X <= x <= HUE_BAR_X + HUE_BAR_W and
                HUE_BAR_Y <= y <= HUE_BAR_Y + HUE_BAR_H):
            return "hue_bar"

        # Brightness slider
        if (BRIGHT_X <= x <= BRIGHT_X + BRIGHT_W and
                BRIGHT_Y_TOP <= y <= BRIGHT_Y_TOP + BRIGHT_H):
            return "brightness_slider"

        # Saturation slider
        if (SAT_X <= x <= SAT_X + SAT_W and
                SAT_Y_TOP <= y <= SAT_Y_TOP + SAT_H):
            return "saturation_slider"

        # Bulb buttons
        for name, (bx, by, bw, bh) in self._bulb_rects.items():
            if bx <= x <= bx + bw and by <= y <= by + bh:
                return name

        return None

    def update_from_pointer(self, x: int, y: int):
        """
        Drag-style update: if (x, y) is over a slider or bar, update the
        corresponding value.  Call this every frame with the left-hand pointer
        position.
        """
        element = self.hit_test(x, y)
        if element == "hue_bar":
            t = (x - HUE_BAR_X) / max(HUE_BAR_W, 1)
            self.hue = float(max(0.0, min(1.0, t)))

        elif element == "brightness_slider":
            t = (y - BRIGHT_Y_TOP) / max(BRIGHT_H, 1)
            self.brightness = int((1.0 - t) * 255)
            self.brightness = max(0, min(255, self.brightness))

        elif element == "saturation_slider":
            t = (y - SAT_Y_TOP) / max(SAT_H, 1)
            self.saturation = float(max(0.0, min(1.0, 1.0 - t)))

    def toggle_bulb(self, bulb_name: str):
        """Toggle a bulb on/off in the active set."""
        if bulb_name in self.active_bulbs:
            self.active_bulbs.discard(bulb_name)
        else:
            self.active_bulbs.add(bulb_name)

    # -----------------------------------------------------------------------
    # Main draw method
    # -----------------------------------------------------------------------

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw all UI overlay elements onto *frame* and return the modified frame.
        Draws in-place but also returns for convenience chaining.
        """
        self._draw_hue_bar(frame)
        self._draw_brightness_slider(frame)
        self._draw_saturation_slider(frame)
        self._draw_swatch(frame)
        self._draw_bulb_buttons(frame)
        self._draw_status_text(frame)
        self._draw_state_indicator(frame)

        if self.show_color_wheel:
            self.draw_color_wheel(frame)

        return frame

    # -----------------------------------------------------------------------
    # Color wheel (new)
    # -----------------------------------------------------------------------

    def draw_color_wheel(self, frame: np.ndarray):
        """
        Draw a circular hue ring (donut shape) in the center-right of the frame.

        The ring spans outer radius WHEEL_OUTER_R to inner radius WHEEL_INNER_R.
        Each 1° arc segment is tinted with the corresponding hue.
        A white dot indicator is drawn at the angle matching self.hue.
        """
        cx = int(self.frame_width  * WHEEL_CENTER_X_FRAC)
        cy = int(self.frame_height * WHEEL_CENTER_Y_FRAC)

        # --- Draw each degree as a filled triangle fan segment ---
        # We draw from outer_r to inner_r per segment so it looks like a donut.
        step_deg = 360.0 / WHEEL_SEGMENTS
        mid_r = (WHEEL_OUTER_R + WHEEL_INNER_R) / 2.0

        for i in range(WHEEL_SEGMENTS):
            angle_start = math.radians(i * step_deg - 90)       # offset so 0° is top
            angle_end   = math.radians((i + 1) * step_deg - 90)
            hue_val     = i / WHEEL_SEGMENTS

            bgr = _hue_to_bgr(hue_val)

            # Four corners of the arc trapezoid
            pts = np.array([
                [cx + int(WHEEL_OUTER_R * math.cos(angle_start)),
                 cy + int(WHEEL_OUTER_R * math.sin(angle_start))],
                [cx + int(WHEEL_OUTER_R * math.cos(angle_end)),
                 cy + int(WHEEL_OUTER_R * math.sin(angle_end))],
                [cx + int(WHEEL_INNER_R * math.cos(angle_end)),
                 cy + int(WHEEL_INNER_R * math.sin(angle_end))],
                [cx + int(WHEEL_INNER_R * math.cos(angle_start)),
                 cy + int(WHEEL_INNER_R * math.sin(angle_start))],
            ], dtype=np.int32)

            cv.fillConvexPoly(frame, pts, bgr)

        # --- Smooth black outline rings ---
        cv.circle(frame, (cx, cy), WHEEL_OUTER_R, (0, 0, 0), 2)
        cv.circle(frame, (cx, cy), WHEEL_INNER_R, (0, 0, 0), 2)

        # --- White indicator dot at current hue angle ---
        # Map hue (0→1) to angle (top of ring → full circle)
        indicator_angle = math.radians(self.hue * 360 - 90)
        dot_r = int(mid_r)
        dot_x = cx + int(dot_r * math.cos(indicator_angle))
        dot_y = cy + int(dot_r * math.sin(indicator_angle))

        cv.circle(frame, (dot_x, dot_y), 10, (255, 255, 255), -1)   # filled white
        cv.circle(frame, (dot_x, dot_y), 10, (0,   0,   0),   2)    # black border

        # --- "DIALING" label above the wheel ---
        label_y = cy - WHEEL_OUTER_R - 14
        cv.putText(
            frame, "Color Wheel",
            (cx - 55, label_y),
            cv.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv.LINE_AA,
        )

        # --- Current hue hex swatch inside the hole ---
        swatch_r = int(WHEEL_INNER_R * 0.55)
        bgr_cur = _hue_to_bgr(self.hue)
        cv.circle(frame, (cx, cy), swatch_r, bgr_cur, -1)
        cv.circle(frame, (cx, cy), swatch_r, (200, 200, 200), 1)

        hue_deg_text = f"{int(self.hue * 360)}°"
        (tw, _), _ = cv.getTextSize(hue_deg_text, cv.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv.putText(
            frame, hue_deg_text,
            (cx - tw // 2, cy + 8),
            cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv.LINE_AA,
        )

    # -----------------------------------------------------------------------
    # State indicator (new)
    # -----------------------------------------------------------------------

    def _draw_state_indicator(self, frame: np.ndarray):
        """
        Draw a pill-shaped panel at the top-center of the frame showing the
        current FSM state name, colored by state.
        """
        state_label = self.state if self.state in _STATE_COLORS_BGR else "IDLE"
        color_bgr   = _STATE_COLORS_BGR.get(state_label, (130, 130, 130))

        text     = f"STATE:  {state_label}"
        font     = cv.FONT_HERSHEY_SIMPLEX
        scale    = 0.75
        thick    = 2

        (tw, th), baseline = cv.getTextSize(text, font, scale, thick)

        pad_x = 18
        pad_y = 10
        panel_w = tw + pad_x * 2
        panel_h = th + pad_y * 2 + baseline

        panel_x = (self.frame_width - panel_w) // 2
        panel_y = 8

        # Background pill (filled rectangle with rounded corners via ellipse mask)
        cv.rectangle(
            frame,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + panel_h),
            (30, 30, 30),
            -1,
        )
        # Colored accent border
        cv.rectangle(
            frame,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + panel_h),
            color_bgr,
            2,
        )
        # Text
        text_x = panel_x + pad_x
        text_y = panel_y + pad_y + th
        cv.putText(frame, text, (text_x, text_y), font, scale, color_bgr, thick, cv.LINE_AA)

    # -----------------------------------------------------------------------
    # Hue bar
    # -----------------------------------------------------------------------

    def _draw_hue_bar(self, frame: np.ndarray):
        """Draw the horizontal hue gradient bar with a triangular cursor."""
        x0, y0 = HUE_BAR_X, HUE_BAR_Y
        x1, y1 = x0 + HUE_BAR_W, y0 + HUE_BAR_H

        # Paste the pre-rendered gradient
        frame[y0:y1, x0:x1] = self._hue_bar_img

        # Cursor line
        cur_x = x0 + int(self.hue * HUE_BAR_W)
        cv.line(frame, (cur_x, y0 - 4), (cur_x, y1 + 4), (255, 255, 255), 2)

        # Label
        cv.putText(
            frame, "Hue",
            (x0, y0 - 8),
            cv.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv.LINE_AA,
        )

    # -----------------------------------------------------------------------
    # Brightness slider
    # -----------------------------------------------------------------------

    def _draw_brightness_slider(self, frame: np.ndarray):
        x0, y0 = BRIGHT_X, BRIGHT_Y_TOP
        x1, y1 = x0 + BRIGHT_W, y0 + BRIGHT_H

        # Vertical gradient: white at top, black at bottom
        for row in range(BRIGHT_H):
            v = int(255 * (1.0 - row / BRIGHT_H))
            frame[y0 + row, x0:x1] = (v, v, v)

        cv.rectangle(frame, (x0, y0), (x1, y1), (180, 180, 180), 1)

        # Cursor
        cur_y = y0 + int((1.0 - self.brightness / 255.0) * BRIGHT_H)
        cv.line(frame, (x0 - 3, cur_y), (x1 + 3, cur_y), (255, 255, 0), 2)

        cv.putText(
            frame, "Bri",
            (x0, y0 - 6),
            cv.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1, cv.LINE_AA,
        )

    # -----------------------------------------------------------------------
    # Saturation slider
    # -----------------------------------------------------------------------

    def _draw_saturation_slider(self, frame: np.ndarray):
        x0, y0 = SAT_X, SAT_Y_TOP
        x1, y1 = x0 + SAT_W, y0 + SAT_H

        # Vertical gradient: full saturation at top, gray at bottom
        hue_bgr = _hue_to_bgr(self.hue)
        for row in range(SAT_H):
            s_val = 1.0 - row / SAT_H
            # Blend between hue color and gray based on saturation
            seg_bgr = _hsv_to_bgr(self.hue, s_val, 0.85)
            frame[y0 + row, x0:x1] = seg_bgr

        cv.rectangle(frame, (x0, y0), (x1, y1), (180, 180, 180), 1)

        # Cursor
        cur_y = y0 + int((1.0 - self.saturation) * SAT_H)
        cv.line(frame, (x0 - 3, cur_y), (x1 + 3, cur_y), (255, 255, 0), 2)

        cv.putText(
            frame, "Sat",
            (x0, y0 - 6),
            cv.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1, cv.LINE_AA,
        )

    # -----------------------------------------------------------------------
    # Preview swatch
    # -----------------------------------------------------------------------

    def _draw_swatch(self, frame: np.ndarray):
        x0, y0 = SWATCH_X, SWATCH_Y
        x1, y1 = x0 + SWATCH_W, y0 + SWATCH_H

        r, g, b = self.get_rgb()
        bgr = (b, g, r)
        cv.rectangle(frame, (x0, y0), (x1, y1), bgr, -1)
        cv.rectangle(frame, (x0, y0), (x1, y1), (200, 200, 200), 1)

        # Show R,G,B values below
        label = f"R{r} G{g} B{b}"
        cv.putText(
            frame, label,
            (x0, y1 + 14),
            cv.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv.LINE_AA,
        )

    # -----------------------------------------------------------------------
    # Bulb buttons
    # -----------------------------------------------------------------------

    def _draw_bulb_buttons(self, frame: np.ndarray):
        """Draw a 2×2 grid of bulb toggle buttons with targeting / locked glows."""
        for name, (bx, by, bw, bh) in self._bulb_rects.items():
            is_active = name in self.active_bulbs
            is_target = (name == self.target_bulb)
            is_locked = (name == self.locked_bulb)

            # --- Glow / highlight effect ---
            if is_locked:
                # Bright orange glow (locked)
                glow_size = 6
                cv.rectangle(
                    frame,
                    (bx - glow_size, by - glow_size),
                    (bx + bw + glow_size, by + bh + glow_size),
                    (0, 140, 255),   # orange BGR
                    glow_size,
                )
            elif is_target:
                # Yellow glow (targeting)
                glow_size = 4
                cv.rectangle(
                    frame,
                    (bx - glow_size, by - glow_size),
                    (bx + bw + glow_size, by + bh + glow_size),
                    (0, 215, 255),   # yellow BGR
                    glow_size,
                )

            # --- Button fill ---
            if is_active:
                btn_color = (60, 60, 200)    # blue-ish = on
            else:
                btn_color = (50, 50, 50)     # dark = off

            cv.rectangle(frame, (bx, by), (bx + bw, by + bh), btn_color, -1)
            cv.rectangle(frame, (bx, by), (bx + bw, by + bh), (180, 180, 180), 1)

            # --- Label ---
            (tw, th), _ = cv.getTextSize(name, cv.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            tx = bx + (bw - tw) // 2
            ty = by + (bh + th) // 2
            cv.putText(
                frame, name,
                (tx, ty),
                cv.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1, cv.LINE_AA,
            )

        # Section label
        cv.putText(
            frame, "Bulbs",
            (BULB_GRID_X, BULB_GRID_Y - 6),
            cv.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv.LINE_AA,
        )

    # -----------------------------------------------------------------------
    # Status text
    # -----------------------------------------------------------------------

    def _draw_status_text(self, frame: np.ndarray):
        """Draw the status message at the bottom of the frame."""
        text    = self.status_text
        font    = cv.FONT_HERSHEY_SIMPLEX
        scale   = 0.65
        thick   = 2
        color   = _STATE_COLORS_BGR.get(self.state, (200, 200, 200))

        (tw, th), _ = cv.getTextSize(text, font, scale, thick)
        tx = (self.frame_width - tw) // 2
        ty = self.frame_height - STATUS_Y_OFFSET

        # Semi-transparent dark background strip
        bg_pad  = 8
        overlay = frame.copy()
        cv.rectangle(
            overlay,
            (tx - bg_pad, ty - th - bg_pad),
            (tx + tw + bg_pad, ty + bg_pad),
            (20, 20, 20),
            -1,
        )
        cv.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        cv.putText(frame, text, (tx, ty), font, scale, color, thick, cv.LINE_AA)

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _build_bulb_rects(self):
        """Pre-compute pixel rectangles for each bulb button."""
        for name, col, row in BULB_GRID:
            bx = BULB_GRID_X + col * (BULB_BTN_W + BULB_BTN_GAP)
            by = BULB_GRID_Y + row * (BULB_BTN_H + BULB_BTN_GAP)
            self._bulb_rects[name] = (bx, by, BULB_BTN_W, BULB_BTN_H)

    @staticmethod
    def _make_hue_bar(width: int, height: int) -> np.ndarray:
        """Create a (height × width × 3) BGR image of the full hue gradient."""
        bar = np.zeros((height, width, 3), dtype=np.uint8)
        for col in range(width):
            hue = col / width
            bgr = _hue_to_bgr(hue)
            bar[:, col] = bgr
        return bar
