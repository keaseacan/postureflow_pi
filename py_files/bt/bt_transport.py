# py_files/bt/bt_transport.py
from __future__ import annotations
import threading
import json
from typing import Any, Optional, Callable, Tuple

# Your existing BLE server (GLib/DBus, NUS, etc.)
import py_files.bt.bt_server as ble


class BleTransport:
    def __init__(
        self,
        *,
        device_name: str = "PosturePi",
        demo_heartbeat: bool = False,
        on_command: Optional[Callable[[Any], None]] = None,
        service_hooks: Optional[Tuple[Callable[[], None], Callable[[], None]]] = None,
        diag: bool = False,
    ):
        self._thread: Optional[threading.Thread] = None
        self._on_command = on_command
        self._service_hooks = service_hooks
        self._diag = diag
        self._started = False

        # Configure BLE defaults
        ble.DEVICE_NAME = device_name
        ble.DEMO_HEARTBEAT = demo_heartbeat

    def start(self) -> None:
        if self._started:
            if self._diag: print("[BLE] transport already started")
            return

        # Optional pause/resume hooks (pairing/connecting)
        if self._service_hooks and hasattr(ble, "set_service_hooks"):
            try:
                on_pause, on_resume = self._service_hooks
                ble.set_service_hooks(on_pause, on_resume)
            except Exception as e:
                if self._diag: print("[BLE] set_service_hooks failed:", repr(e))

        # Optional RX handler for inbound commands
        if self._on_command and hasattr(ble, "set_rx_handler"):
            try:
                ble.set_rx_handler(self._on_command)
                if self._diag: print("[BLE] RX handler attached")
            except Exception as e:
                if self._diag: print("[BLE] set_rx_handler failed:", repr(e))

        self._thread = threading.Thread(target=ble.main, name="ble-thread", daemon=True)
        self._thread.start()
        self._started = True
        if self._diag: print("[BLE] transport started")

    def stop(self) -> None:
        try:
            main_loop = getattr(ble, "MAIN_LOOP", None)
            if main_loop is not None:
                main_loop.quit()
        except Exception as e:
            if self._diag: print("[BLE] MAIN_LOOP.quit() failed:", repr(e))

    def join(self, timeout: float = 1.5) -> None:
        t = self._thread
        if t and t.is_alive():
            t.join(timeout=timeout)

    def send(self, payload: Any) -> None:
        try:
            if isinstance(payload, (dict, list)) and hasattr(ble, "send_json"):
                ble.send_json(payload)
            elif hasattr(ble, "send_text"):
                text = payload if isinstance(payload, str) else json.dumps(payload)
                ble.send_text(text)
            elif hasattr(ble, "send"):
                ble.send(payload)
            else:
                if self._diag: print("[BLE] No send_* API available; cannot reply")
        except Exception as e:
            if self._diag: print("[BLE] send failed:", repr(e))


# ---------- Simple wrappers (singleton-style), like init_outbox/start_audio_pipeline ----------
_BT: Optional[BleTransport] = None
_BT_LOCK = threading.Lock()

def init_ble(*,
             on_command: Optional[Callable[[Any], None]] = None,
             service_hooks: Optional[Tuple[Callable[[], None], Callable[[], None]]] = None,
             device_name: str = "PosturePi",
             demo_heartbeat: bool = False,
             diag: bool = False) -> BleTransport:
    """Initialize BLE transport (no thread started), like init_outbox()."""
    global _BT
    with _BT_LOCK:
        if _BT is None:
            _BT = BleTransport(
                device_name=device_name,
                demo_heartbeat=demo_heartbeat,
                on_command=on_command,
                service_hooks=service_hooks,
                diag=diag,
            )
        else:
            # Allow late wiring of handlers if re-initialized
            if on_command is not None: _BT._on_command = on_command
            if service_hooks is not None: _BT._service_hooks = service_hooks
        return _BT

def start_ble() -> BleTransport:
    """Start BLE thread, like start_audio_pipeline() (returns the transport)."""
    global _BT
    with _BT_LOCK:
        if _BT is None:
            _BT = BleTransport()
        _BT.start()
        return _BT

def stop_ble() -> None:
    global _BT
    with _BT_LOCK:
        if _BT is not None:
            _BT.stop()

def join_ble(timeout: float = 1.5) -> None:
    global _BT
    with _BT_LOCK:
        if _BT is not None:
            _BT.join(timeout=timeout)

def ble_send(payload: Any) -> None:
    """Convenience send (no-op if BLE not started)."""
    global _BT
    with _BT_LOCK:
        if _BT is not None:
            _BT.send(payload)

def is_ble_running() -> bool:
    global _BT
    t = getattr(_BT, "_thread", None)
    return bool(_BT and getattr(_BT, "_started", False) and t and t.is_alive())


# Back-compat alias to match your current import style
bleTransport = BleTransport
