# py_files/bt/bt_protocol.py
from __future__ import annotations
from typing import Any, Callable, Optional, Tuple, List, Dict
import json

from py_files.bt.bt_transport import init_ble, ble_send  # transport only
from py_files.time.time_main import _now_ms, apply_phone_time_sync

# ---------------- State (callbacks provided by core) ----------------
_diag: bool = False
_on_start: Optional[Callable[[], None]] = None
_on_stop: Optional[Callable[[], None]] = None
_on_status: Optional[Callable[[], Dict[str, Any]]] = None
_on_ack: Optional[Callable[[List[int]], None]] = None


# ---------------- TX helpers (all your custom outbound) -------------
def send(obj: Any) -> None:
    """Generic JSON line over BLE."""
    # ble_send already json.dumps on dict/list; it wraps text as needed in bt_transport
    ble_send(obj)

def send_label(label: str, **extra) -> None:
    """Common {ts_ms,label,...} envelope."""
    payload = {"ts_ms": _now_ms(), "label": str(label)}
    if extra:
        payload.update(extra)
    ble_send(payload)

def send_ok(**kw) -> None:
    ble_send({"ok": True, **kw})

def send_err(err: str, **kw) -> None:
    ble_send({"ok": False, "err": str(err), **kw})

def send_time_sync_req() -> None:
    ble_send({"cmd": "time_sync_req"})

def send_hello() -> None:
    send_label("hello")

def send_status(payload: Dict[str, Any]) -> None:
    ble_send({"ok": True, "status": payload})


# ---------------- RX handling (single source of truth) --------------
def _normalize_to_obj(msg: Any) -> Optional[dict]:
    if isinstance(msg, (bytes, bytearray)):
        s = msg.decode("utf-8", "ignore").strip()
    elif isinstance(msg, str):
        s = msg.strip()
    elif isinstance(msg, dict):
        return msg.copy()
    else:
        return None
    # treat non-JSON lines as {"cmd": "<line>"}
    try:
        return json.loads(s)
    except Exception:
        return {"cmd": s}

def _handle_time_sync(obj: dict) -> None:
    epoch_ms = obj.get("epoch_ms")
    if epoch_ms is None:
        send_err("missing_epoch_ms"); return
    try:
        epoch_ms = int(epoch_ms)
    except Exception:
        send_err("bad_epoch_ms"); return
    drift = abs(_now_ms() - epoch_ms)
    if drift > 1000:
        sys_ok, rtc_ok = apply_phone_time_sync(epoch_ms)
    else:
        sys_ok = rtc_ok = True
    send_ok(cmd="time_sync", drift_ms=drift, sys_ok=sys_ok, rtc_ok=rtc_ok)

def _handle_ack(obj: dict) -> None:
    if _on_ack is None:
        send_err("ack_handler_unavailable"); return
    ids = obj.get("ids", [])
    if not isinstance(ids, list):
        send_err("ack.ids_not_list"); return
    try:
        _on_ack([int(i) for i in ids])
        send_ok(cmd="ack", n=len(ids))
    except Exception as e:
        send_err(f"bad_ack:{e}")

def _handle_status() -> None:
    if _on_status is None:
        send_status({})
    else:
        try:
            send_status(_on_status() or {})
        except Exception as e:
            send_err(f"bad_status:{e}")

def _handle_rx(msg: Any) -> None:
    """Registered as bt_transport on_command callback."""
    obj = _normalize_to_obj(msg)
    if obj is None:
        send_err("bad_type"); return
    cmd = str(obj.get("cmd", "")).strip().lower()
    if not cmd:
        send_err("bad_command"); return

    # Lightweight utilities
    if cmd == "ping":
        send_label("pong"); return
    if cmd == "echo":
        send_label("echo", msg=obj.get("msg", "")); return

    # Core controls
    if cmd == "start":
        if _on_start: _on_start()
        send_ok(state="running"); return
    if cmd == "stop":
        if _on_stop: _on_stop()
        send_ok(state="stopped"); return
    if cmd == "status":
        _handle_status(); return
    if cmd == "ack":
        _handle_ack(obj); return
    if cmd == "time_sync":
        _handle_time_sync(obj); return

    send_err(f"unknown_cmd:{cmd}")

# ---------------- Public API ---------------------------------------
def init_protocol(
    *,
    on_start: Callable[[], None],
    on_stop: Callable[[], None],
    on_status: Optional[Callable[[], Dict[str, Any]]] = None,
    on_ack: Optional[Callable[[List[int]], None]] = None,
    device_name: str = "PosturePi",
    demo_heartbeat: bool = False,
    diag: bool = False,
) -> None:
    """Initialize BLE (transport) and register this file's RX handler."""
    global _on_start, _on_stop, _on_status, _on_ack, _diag
    _on_start, _on_stop = on_start, on_stop
    _on_status, _on_ack = on_status, on_ack
    _diag = diag

    # Wire our RX handler into the transport; let transport manage pairing/loops.
    init_ble(
        on_command=_handle_rx,
        service_hooks=(on_stop, on_start),  # pause services while pairing if transport uses it
        device_name=device_name,
        demo_heartbeat=demo_heartbeat,
        diag=diag,
    )
