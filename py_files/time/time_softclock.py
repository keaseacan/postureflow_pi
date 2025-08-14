# fn_time.py
import os
import json
import time
import subprocess
from datetime import datetime, timezone
from typing import Optional, List
import smbus2 as sm

from py_files.fn_cfg import RTC_ADDR, I2C_PORT, ANCHOR_PATH, MIN_VALID_UNIX_MS
from py_files.time.time_helpers import bcd_to_int, int_to_bcd

def _monotonic_ms() -> int:
    return int(time.monotonic() * 1000)

def _plausible(ms: int) -> bool:
    return MIN_VALID_UNIX_MS <= ms <= 4102444800000  # <= 2100-01-01 UTC

class SoftClock:
    """
    Monotonic-anchored UTC clock:
      now_ms() = monotonic_ms() + offset
    Offset is learned from phone time or reconstructed from persisted anchor.
    """
    def __init__(self, anchor_path: str = ANCHOR_PATH):
        self.anchor_path = anchor_path
        self._offset_ms: Optional[int] = None
        self._load_anchor_into_offset_if_available()

    def now_ms(self) -> int:
        if self._offset_ms is not None:
            return _monotonic_ms() + self._offset_ms
        # fallback if never synced (may be wrong if system clock is wrong)
        return int(time.time() * 1000)

    def sync(self, epoch_ms: int) -> None:
        """Set from trusted absolute UTC (e.g., phone)."""
        self._offset_ms = int(epoch_ms) - _monotonic_ms()
        self._save_anchor(epoch_ms)

    def bootstrap_from_anchor(self) -> bool:
        """Rebuild offset from saved anchor (last good UTC + monotonic delta)."""
        a = self._load_anchor()
        if not a:
            return False
        approx = int(a["utc_ms"]) + (_monotonic_ms() - int(a["mono_ms"]))
        if _plausible(approx):
            self._offset_ms = approx - _monotonic_ms()
            return True
        return False

    # ---- anchor I/O ----
    def _save_anchor(self, epoch_ms: int) -> None:
        try:
            os.makedirs(os.path.dirname(self.anchor_path), exist_ok=True)
            with open(self.anchor_path, "w") as f:
                json.dump({"utc_ms": int(epoch_ms), "mono_ms": _monotonic_ms()}, f)
        except Exception as e:
            print(f"[TIME] anchor save failed: {e}")

    def _load_anchor(self) -> Optional[dict]:
        try:
            with open(self.anchor_path, "r") as f:
                return json.load(f)
        except Exception:
            return None

    def _load_anchor_into_offset_if_available(self) -> None:
        a = self._load_anchor()
        if a and _plausible(a.get("utc_ms", 0)):
            self._offset_ms = int(a["utc_ms"]) - _monotonic_ms()

# Global soft clock
clock = SoftClock()

# ---------- I2C / RTC ----------
def setup_i2c() -> bool:
    """Open/close I2C to prove the bus is okay."""
    try:
        with sm.SMBus(I2C_PORT):
            print("[I2C] init OK")
            return True
    except Exception as e:
        print(f"[I2C] init FAILED: {e}")
        return False

def read_rtc() -> Optional[List[int]]:
    """
    Raw DS3231 registers 0x00..0x06 (sec..year). Returns None on failure.
    """
    try:
        with sm.SMBus(I2C_PORT) as bus:
            td = bus.read_i2c_block_data(RTC_ADDR, 0x00, 7)
            print("[RTC] read OK")
            return td
    except Exception as e:
        print(f"[RTC] read FAILED: {e}")
        return None

def rtc_read_epoch_ms() -> Optional[int]:
    """Read DS3231 and convert to UTC epoch ms. Returns None if unreadable or implausible."""
    td = read_rtc()
    if not td:
        return None
    try:
        s, m, h, _dow, d, mo, y = td
        year  = 2000 + bcd_to_int(y)
        month = bcd_to_int(mo & 0x1F)
        day   = bcd_to_int(d)
        hour  = bcd_to_int(h & 0x3F)  # 24h
        minute= bcd_to_int(m)
        sec   = bcd_to_int(s & 0x7F)
        dt = datetime(year, month, day, hour, minute, sec, tzinfo=timezone.utc)
        ms = int(dt.timestamp() * 1000)
        if not _plausible(ms):
            print(f"[RTC] implausible time {dt.isoformat()}")
            return None
        return ms
    except Exception as e:
        print(f"[RTC] parse FAILED: {e}")
        return None

def rtc_write_epoch_ms(epoch_ms: int) -> bool:
    """Write UTC epoch ms to DS3231."""
    try:
        dt = datetime.fromtimestamp(epoch_ms / 1000, tz=timezone.utc)
        data = [
            int_to_bcd(dt.second),
            int_to_bcd(dt.minute),
            int_to_bcd(dt.hour),                         # 24h
            int_to_bcd((dt.isoweekday() % 7) or 7),      # 1..7
            int_to_bcd(dt.day),
            int_to_bcd(dt.month),                        # ignore century bit
            int_to_bcd(dt.year - 2000),
        ]
        with sm.SMBus(I2C_PORT) as bus:
            bus.write_i2c_block_data(RTC_ADDR, 0x00, data)
        print(f"[RTC] write OK -> {dt.isoformat()}")
        return True
    except Exception as e:
        print(f"[RTC] write FAILED: {e}")
        return False

# ---------- System clock ----------
def _system_set_time_epoch_ms(epoch_ms: int) -> bool:
    """
    Set Linux system time (UTC). Needs root; uses sudo -n.
    """
    try:
        dt = datetime.fromtimestamp(epoch_ms / 1000, tz=timezone.utc)
        # Use -u to avoid local TZ influence.
        proc = subprocess.run(
            ["sudo", "-n", "date", "-u", "-s", dt.strftime("%Y-%m-%d %H:%M:%S")],
            check=True, capture_output=True, text=True
        )
        print("[SYS] clock set:", proc.stdout.strip() or dt.isoformat())
        return True
    except subprocess.CalledProcessError as e:
        print(f"[SYS] set FAILED (rc={e.returncode}): {e.stderr.strip()}")
        return False
    except Exception as e:
        print(f"[SYS] set FAILED: {e}")
        return False