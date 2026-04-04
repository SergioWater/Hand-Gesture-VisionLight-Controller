"""
bulb_controller.py — Tuya Cloud API bulb controller for Hand Gesture Light Controller.

Loads device configuration from bulbs.json (same directory).
Falls back to mock mode when credentials are placeholders.
"""

import os
import json
import time
import colorsys

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bulbs.json")

_PLACEHOLDER_VALUES = {
    "YOUR_TUYA_ACCESS_ID",
    "YOUR_TUYA_ACCESS_SECRET",
}

def _load_config():
    with open(_CONFIG_PATH, "r") as f:
        return json.load(f)

def _is_placeholder(*values):
    return any(v in _PLACEHOLDER_VALUES for v in values)


# ---------------------------------------------------------------------------
# BulbController
# ---------------------------------------------------------------------------

class BulbController:
    """Controls Tuya smart LED bulbs via the Tuya Cloud API (tinytuya)."""

    # Minimum seconds between consecutive commands to the same device
    _RATE_LIMIT_SECONDS = 0.2

    def __init__(self):
        self.mock_mode = False
        self.cloud = None
        self.devices = {}           # name -> device_id
        self._last_command_time = {}  # device_id -> float (epoch)

        try:
            config = _load_config()
        except Exception as e:
            print(f"[BulbController] Failed to load bulbs.json: {e}. Running in MOCK MODE.")
            self.mock_mode = True
            return

        tuya_cfg = config.get("tuya", {})
        access_id     = tuya_cfg.get("access_id", "")
        access_secret = tuya_cfg.get("access_secret", "")
        region        = tuya_cfg.get("region", "us")

        # Populate device map
        for name, info in config.get("devices", {}).items():
            self.devices[name] = info.get("device_id", "")

        if _is_placeholder(access_id, access_secret):
            print("[BulbController] Placeholder credentials detected. Running in MOCK MODE.")
            self.mock_mode = True
            return

        try:
            import tinytuya
            self.cloud = tinytuya.Cloud(
                apiRegion=region,
                apiKey=access_id,
                apiSecret=access_secret,
            )
            print("[BulbController] Tuya Cloud API loaded successfully.")
        except ImportError:
            print("[BulbController] tinytuya not installed. Running in MOCK MODE.")
            self.mock_mode = True
        except Exception as e:
            print(f"[BulbController] Failed to initialise Tuya Cloud: {e}. Running in MOCK MODE.")
            self.mock_mode = True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rate_limit(self, device_id: str):
        """Block until the per-device rate limit has elapsed."""
        now = time.time()
        last = self._last_command_time.get(device_id, 0.0)
        wait = self._RATE_LIMIT_SECONDS - (now - last)
        if wait > 0:
            time.sleep(wait)
        self._last_command_time[device_id] = time.time()

    def _send(self, device_id: str, commands: dict):
        """Send a command dict to a device, with rate limiting and error handling."""
        self._rate_limit(device_id)
        try:
            result = self.cloud.sendcommand(device_id, commands)
            return result
        except Exception as e:
            print(f"[BulbController] Error sending command to {device_id}: {e}")
            return None

    def _resolve_devices(self, bulb_names: set) -> list:
        """Return list of (name, device_id) pairs for the requested bulb names."""
        resolved = []
        for name in bulb_names:
            device_id = self.devices.get(name)
            if not device_id:
                print(f"[BulbController] Unknown bulb name: {name!r} — skipping.")
                continue
            if _is_placeholder(device_id):
                print(f"[BulbController] Placeholder device_id for {name!r} — skipping.")
                continue
            resolved.append((name, device_id))
        return resolved

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_color(self, bulb_names: set, r: int, g: int, b: int):
        """
        Set bulb color from RGB (each 0-255).

        Converts to Tuya HSV: H 0-360, S 0-1000, V 0-1000 and sends via
        the colour_data_v2 / work_mode codes.
        """
        # colorsys works in 0.0-1.0 floats
        rf, gf, bf = r / 255.0, g / 255.0, b / 255.0
        h_frac, s_frac, v_frac = colorsys.rgb_to_hsv(rf, gf, bf)

        h = round(h_frac * 360)
        s = round(s_frac * 1000)
        v = round(v_frac * 1000)

        if self.mock_mode:
            print(f"[BulbController][MOCK] set_color({bulb_names}, r={r}, g={g}, b={b}) "
                  f"-> H={h} S={s} V={v}")
            return

        commands = {
            "commands": [
                {"code": "work_mode",      "value": "colour"},
                {"code": "colour_data_v2", "value": {"h": h, "s": s, "v": v}},
            ]
        }

        for name, device_id in self._resolve_devices(bulb_names):
            print(f"[BulbController] set_color {name} -> H={h} S={s} V={v}")
            self._send(device_id, commands)

    def set_brightness(self, bulb_names: set, brightness: int):
        """
        Set bulb brightness.

        brightness: 0-255 (caller scale) mapped to Tuya's 10-1000 range.
        """
        # Map 0-255 -> 10-1000 (Tuya minimum is 10, not 0)
        bright_val = max(10, round((brightness / 255.0) * 1000))

        if self.mock_mode:
            print(f"[BulbController][MOCK] set_brightness({bulb_names}, {brightness}) "
                  f"-> tuya={bright_val}")
            return

        commands = {
            "commands": [
                {"code": "bright_value_v2", "value": bright_val},
            ]
        }

        for name, device_id in self._resolve_devices(bulb_names):
            print(f"[BulbController] set_brightness {name} -> {bright_val}")
            self._send(device_id, commands)

    def turn_on(self, bulb_names: set):
        """Turn bulbs on (switch_led = True)."""
        if self.mock_mode:
            print(f"[BulbController][MOCK] turn_on({bulb_names})")
            return

        commands = {"commands": [{"code": "switch_led", "value": True}]}

        for name, device_id in self._resolve_devices(bulb_names):
            print(f"[BulbController] turn_on {name}")
            self._send(device_id, commands)

    def turn_off(self, bulb_names: set):
        """Turn bulbs off (switch_led = False)."""
        if self.mock_mode:
            print(f"[BulbController][MOCK] turn_off({bulb_names})")
            return

        commands = {"commands": [{"code": "switch_led", "value": False}]}

        for name, device_id in self._resolve_devices(bulb_names):
            print(f"[BulbController] turn_off {name}")
            self._send(device_id, commands)

    def cleanup(self):
        """No-op cleanup hook (retained for API compatibility with main.py)."""
        pass
