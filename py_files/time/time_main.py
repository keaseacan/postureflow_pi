# wrapper call on softclock file
import py_files.time.time_softclock as ob_rtc

# dependencies
from typing import Tuple

# ---------- Public helpers matching your original names ----------
def write_to_pi() -> bool:
    """
    (Kept name) Set system clock from RTC, if plausible.
    Also seeds the soft clock so _now_ms() is consistent afterward.
    """
    ms = ob_rtc.rtc_read_epoch_ms()
    if ms is None:
        print("[RTC] system NOT set (no/invalid RTC)")
        return False
    ok = ob_rtc._system_set_time_epoch_ms(ms)
    if ok:
        ob_rtc.clock.sync(ms)
        print("[RTC] pi clock successfully overwritten (and soft clock synced)")
    return ok

def _now_ms() -> int:
    """
    Use soft clock for all event timestamps.
    Falls back to time.time() if never synced.
    """
    return ob_rtc.clock.now_ms()

# ---------- Startup / BLE integration ----------
def init_clock() -> str:
    """
    Call ONCE at startup:
      1) Try RTC -> if plausible: set system clock + soft clock
      2) Else, try anchor -> set soft clock (no system change)
    Returns "rtc", "anchor", or "none".
    """
    ms = ob_rtc.rtc_read_epoch_ms()
    if ms is not None:
        ob_rtc._system_set_time_epoch_ms(ms)  # best effort
        ob_rtc.clock.sync(ms)
        return "rtc"

    if ob_rtc.clock.bootstrap_from_anchor():
        print("[TIME] bootstrapped from anchor")
        return "anchor"

    print("[TIME] no RTC and no anchor; running unsynced until phone sync")
    return "none"

def apply_phone_time_sync(epoch_ms: int) -> Tuple[bool, bool]:
    """
    Call on BLE connect with phone's UTC epoch_ms.
    Sets soft clock, writes RTC, and sets system time.
    Returns (sys_ok, rtc_ok).
    """
    ob_rtc.clock.sync(epoch_ms)
    sys_ok = ob_rtc._system_set_time_epoch_ms(epoch_ms)
    rtc_ok = ob_rtc.rtc_write_epoch_ms(epoch_ms)
    return sys_ok, rtc_ok
