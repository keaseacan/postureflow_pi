#!/usr/bin/env python3
"""
BLE Peripheral (NUS) with "press to pair, hold to disconnect" on GPIO17.
"""

# dependencies
import json
import queue
import threading
import time
import signal
import sys
import dbus
import dbus.exceptions
import dbus.mainloop.glib
import dbus.service
from typing import Callable, Optional, Any, Union
from gi.repository import GLib

# config constants
from py_files.fn_cfg import DEVICE_NAME, GPIO_BUTTON_PIN, BUTTON_HOLD_SEC, DEMO_HEARTBEAT, DEMO_HEARTBEAT_SEC
from py_files.fn_cfg import NUS_SERVICE_UUID, NUS_RX_UUID, NUS_TX_UUID, ADVERTISE_SERVICE_UUIDS
from py_files.fn_cfg import RUN_LIVE_BLE_DIAGNOSTICS

# ---------- BlueZ constants ----------
BLUEZ_SERVICE_NAME = 'org.bluez'
GATT_MANAGER_IFACE = 'org.bluez.GattManager1'
LE_ADVERTISING_MANAGER_IFACE = 'org.bluez.LEAdvertisingManager1'
DBUS_OM_IFACE = 'org.freedesktop.DBus.ObjectManager'
DBUS_PROP_IFACE = 'org.freedesktop.DBus.Properties'
DEVICE_IFACE = 'org.bluez.Device1'

MAIN_LOOP = None
_tx_queue = queue.Queue()

# service pause/resume hooks
_pause_cb: Optional[Callable[[], None]] = None
_resume_cb: Optional[Callable[[], None]] = None
_services_paused = False

def set_service_hooks(pause_cb: Optional[Callable[[], None]], resume_cb: Optional[Callable[[], None]]):
    global _pause_cb, _resume_cb
    _pause_cb, _resume_cb = pause_cb, resume_cb

def _safe_call(tag: str, fn: Callable[[], None]):
    try: fn()
    except Exception as e: print(f"[BLE] {tag}_cb error:", e)

def _pause_services_async():
    global _services_paused
    if _services_paused: return
    _services_paused = True
    if _pause_cb:
        threading.Thread(target=_safe_call, args=("pause", _pause_cb), daemon=True).start()

def _resume_services_async():
    global _services_paused
    if not _services_paused: return
    _services_paused = False
    if _resume_cb:
        threading.Thread(target=_safe_call, args=("resume", _resume_cb), daemon=True).start()

# --------- Convenience sends ----------
def _now_ms() -> int:
    return int(time.time() * 1000)

def ble_send(obj: dict):
    """Queue a JSON line for TX notifications."""
    _tx_queue.put((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))

def ble_send_label(label: str, ts_ms: Optional[int]=None, **extra):
    """Shorthand to send a {ts_ms, label, ...} JSON line."""
    payload = {"ts_ms": ts_ms if ts_ms is not None else _now_ms(), "label": label}
    if extra:
        payload.update(extra)
    ble_send(payload)

# --- public hooks/shims for transport integration (2-way API) ---
_rx_handler = None  # fn(dict|list|str|bytes) -> None

def set_rx_handler(fn: Callable[[Any], None]):
    """Transport/core-provided RX handler. Receives parsed JSON (any type) or raw text."""
    global _rx_handler
    _rx_handler = fn

def send_json(obj: Any):
    """Outbound JSON helper (newline-delimited)."""
    ble_send(obj if isinstance(obj, dict) else {"data": obj})

def send_text(s: str):
    """Outbound text helper (wrapped in a small JSON)."""
    ble_send({"text": str(s)})

def send(payload: Any):
    """Generic outbound helper."""
    if isinstance(payload, (dict, list)):
        send_json(payload)
    else:
        send_text(str(payload))

# ---------- Rolling status line ----------
_spinner_frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

def _fmt_dur(sec: float) -> str:
    if not sec or sec < 0: return "--:--"
    sec = int(sec)
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

# ---------- DBus scaffolding ----------
class InvalidArgsException(dbus.exceptions.DBusException):
    _dbus_error_name = 'org.freedesktop.DBus.Error.InvalidArgs'
class NotSupportedException(dbus.exceptions.DBusException):
    _dbus_error_name = 'org.bluez.Error.NotSupported'

class Application(dbus.service.Object):
    def __init__(self, bus):
        self.path = '/app'; self.services = []; super().__init__(bus, self.path)
    def get_path(self): return dbus.ObjectPath(self.path)
    def add_service(self, s): self.services.append(s)
    @dbus.service.method(DBUS_OM_IFACE, out_signature='a{oa{sa{sv}}}')
    def GetManagedObjects(self):
        out = {}
        for s in self.services:
            out[s.get_path()] = s.get_properties()
            for c in s.get_characteristics(): out[c.get_path()] = c.get_properties()
        return out

class Service(dbus.service.Object):
    def __init__(self, bus, index, uuid, primary=True):
        self.path=f'/app/service{index}'; self.uuid=uuid; self.primary=primary; self.characteristics=[]
        super().__init__(bus, self.path)
    def get_path(self): return dbus.ObjectPath(self.path)
    def add_characteristic(self, ch): self.characteristics.append(ch)
    def get_characteristics(self): return self.characteristics
    def get_properties(self):
        return {'org.bluez.GattService1': {'UUID': self.uuid, 'Primary': self.primary, 'Includes': dbus.Array(signature='o')}}

class Characteristic(dbus.service.Object):
    def __init__(self, bus, index, uuid, flags, service):
        self.path = service.get_path()+f'/char{index}'; self.uuid=uuid; self.flags=flags; self.service=service; self.notifying=False
        super().__init__(bus, self.path)
    def get_path(self): return dbus.ObjectPath(self.path)
    def get_properties(self):
        return {'org.bluez.GattCharacteristic1': {'Service': self.service.get_path(),'UUID': self.uuid,'Flags': dbus.Array(self.flags, signature='s'),'Descriptors': dbus.Array(signature='o')}}
    @dbus.service.method('org.bluez.GattCharacteristic1', in_signature='a{sv}', out_signature='ay')
    def ReadValue(self, options): return dbus.Array([], signature='y')
    @dbus.service.method('org.bluez.GattCharacteristic1', in_signature='aya{sv}')
    def WriteValue(self, value, options): raise NotSupportedException("Write not supported")
    @dbus.service.method('org.bluez.GattCharacteristic1')
    def StartNotify(self):
        if 'notify' not in self.flags: raise NotSupportedException("Notify not supported")
        self.notifying=True
    @dbus.service.method('org.bluez.GattCharacteristic1')
    def StopNotify(self): self.notifying=False
    def send_notification(self, payload: bytes):
        if not self.notifying: return
        arr=dbus.Array([dbus.Byte(b) for b in payload], signature='y')
        self.PropertiesChanged('org.bluez.GattCharacteristic1', {'Value': arr}, [])
    @dbus.service.signal(DBUS_PROP_IFACE, signature='sa{sv}as')
    def PropertiesChanged(self, interface, changed, invalidated): pass

class NusRxCharacteristic(Characteristic):
    def __init__(self, bus, index, service):
        super().__init__(bus, index, NUS_RX_UUID, ['write','write-without-response'], service)
        self.on_line = None   # callback for parsed JSON / text lines
    @dbus.service.method('org.bluez.GattCharacteristic1', in_signature='aya{sv}')
    def WriteValue(self, value, options):
        try:
            s = bytes(bytearray(value)).decode('utf-8', errors='ignore')
            for line in s.splitlines():
                line=line.strip()
                if not line: continue
                # Try JSON first; if not JSON, pass raw text
                try:
                    obj = json.loads(line)
                    print('\n[BLE][RX]', obj)
                    if self.on_line:
                        self.on_line(obj)
                except Exception:
                    print('\n[BLE][RX] text:', line)
                    if self.on_line:
                        self.on_line(line)
        except Exception as e:
            print('\n[BLE][RX] parse-error:', e)

class NusTxCharacteristic(Characteristic):
    def __init__(self, bus, index, service):
        super().__init__(bus, index, NUS_TX_UUID, ['notify'], service)
        self.on_start_notify=None; self.on_stop_notify=None
    @dbus.service.method('org.bluez.GattCharacteristic1')
    def StartNotify(self):
        super().StartNotify()
        try: self.on_start_notify and self.on_start_notify()
        except Exception as e: print('\n[BLE] on_start_notify error:', e)
    @dbus.service.method('org.bluez.GattCharacteristic1')
    def StopNotify(self):
        super().StopNotify()
        try: self.on_stop_notify and self.on_stop_notify()
        except Exception as e: print('\n[BLE] on_stop_notify error:', e)

class NusService(Service):
    def __init__(self, bus, index):
        super().__init__(bus, index, NUS_SERVICE_UUID, True)
        self.tx = NusTxCharacteristic(bus,0,self); self.rx = NusRxCharacteristic(bus,1,self)
        self.add_characteristic(self.tx); self.add_characteristic(self.rx)

class Advertisement(dbus.service.Object):
    PATH_BASE = '/app/advertisement'
    def __init__(self, bus, index, ad_type, service_uuids):
        self.path = self.PATH_BASE + str(index); self.ad_type = ad_type; self.service_uuids = service_uuids
        super().__init__(bus, self.path)
    def get_path(self): return dbus.ObjectPath(self.path)
    @dbus.service.method('org.freedesktop.DBus.Properties', out_signature='a{sv}')
    def GetAll(self, interface):
        if interface != 'org.bluez.LEAdvertisement1':
            raise InvalidArgsException()
        props = {
            'Type': self.ad_type,
            'LocalName': DEVICE_NAME,
            'IncludeTxPower': dbus.Boolean(True),
        }
        if ADVERTISE_SERVICE_UUIDS and self.service_uuids:
            props['ServiceUUIDs'] = dbus.Array(self.service_uuids, signature='s')
        return props
    @dbus.service.method('org.bluez.LEAdvertisement1')
    def Release(self): pass

# ---------- Adapter & device helpers ----------
def find_adapter(bus):
    om = dbus.Interface(bus.get_object(BLUEZ_SERVICE_NAME, '/'), DBUS_OM_IFACE)
    for path, ifaces in om.GetManagedObjects().items():
        if 'org.bluez.Adapter1' in ifaces:
            return path
    raise RuntimeError('Bluetooth adapter not found')

def disconnect_all_connected(bus):
    om = dbus.Interface(bus.get_object(BLUEZ_SERVICE_NAME, '/'), DBUS_OM_IFACE)
    for path, ifaces in om.GetManagedObjects().items():
        dev = ifaces.get(DEVICE_IFACE)
        if dev and bool(dev.get('Connected', False)):
            try:
                dbus.Interface(bus.get_object(BLUEZ_SERVICE_NAME, path), DEVICE_IFACE).Disconnect()
                print('\n[BLE] Disconnect requested:', path)
            except Exception as e:
                print('\n[BLE] Disconnect error:', e)

# ---------- Main ----------
def main():
    dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
    bus = dbus.SystemBus()

    adapter_path = find_adapter(bus)
    adapter_props = dbus.Interface(bus.get_object(BLUEZ_SERVICE_NAME, adapter_path), DBUS_PROP_IFACE)
    adapter_props.Set('org.bluez.Adapter1', 'Powered', dbus.Boolean(True))

    app = Application(bus); nus = NusService(bus, 0); app.add_service(nus)
    service_mgr = dbus.Interface(bus.get_object(BLUEZ_SERVICE_NAME, adapter_path), GATT_MANAGER_IFACE)
    ad_mgr = dbus.Interface(bus.get_object(BLUEZ_SERVICE_NAME, adapter_path), LE_ADVERTISING_MANAGER_IFACE)
    adv = Advertisement(bus, 0, 'peripheral', [NUS_SERVICE_UUID])

    service_mgr.RegisterApplication(app.get_path(), {}, reply_handler=lambda: print('\n[BLE] GATT app registered'),
                                    error_handler=lambda e: print('\n[BLE] GATT app register error:', e))

    # State
    state = {
        'adv_registered': False,
        'client_connected': False,
        't_adv_start': None,
        't_conn_start': None,
        'spin_idx': 0,
        'hb_src': None,   # GLib source id for heartbeat
    }

    def _register_adv():
        if state['adv_registered']: return False
        def ok():
            state['adv_registered'] = True
            state['t_adv_start'] = time.time()
            print('\n[BLE] Advertising started')
            return False
        def err(e):
            print('\n[BLE] Adv register error:', e); return False
        ad_mgr.RegisterAdvertisement(adv.get_path(), {}, reply_handler=ok, error_handler=err)
        return False

    def _unregister_adv():
        if not state['adv_registered']: return False
        try:
            ad_mgr.UnregisterAdvertisement(adv.get_path())
            print('\n[BLE] Advertising stopped')
        except Exception as e:
            print('\n[BLE] Adv unregister error:', e)
        state['adv_registered'] = False
        state['t_adv_start'] = None
        return False

    def enter_pairing_mode():
        _pause_services_async()
        GLib.idle_add(_register_adv)

    def exit_pairing_mode():
        GLib.idle_add(_unregister_adv)
        if not state['client_connected']:
            _resume_services_async()

    # Button handlers
    try:
        from gpiozero import Button
        button = Button(GPIO_BUTTON_PIN, pull_up=True, bounce_time=0.05, hold_time=BUTTON_HOLD_SEC)
        _held = {'v': False}
        def on_held():
            _held['v'] = True
            GLib.idle_add(lambda: (disconnect_all_connected(bus) if state['client_connected'] else _unregister_adv()))
            GLib.timeout_add(200, lambda: (_resume_services_async() or False))
        def on_released():
            if _held['v']:
                _held['v'] = False; return
            if state['adv_registered'] or state['client_connected']:
                return
            enter_pairing_mode()
        button.when_held = on_held
        button.when_released = on_released
        print(f"\n[BLE] Button on GPIO{GPIO_BUTTON_PIN}: short=pair, hold(≥{BUTTON_HOLD_SEC}s)=disconnect")
    except Exception as e:
        print(f"\n[BLE] gpiozero unavailable ({e}); you can still pair via manual call to enter_pairing_mode()")
        GLib.idle_add(_register_adv)

    # ---- Built-in RX command handler (fallback if no external handler set) ----
    def on_rx_cmd(obj: Union[dict, str, bytes]):
        if isinstance(obj, dict):
            cmd = str(obj.get('cmd', '')).lower()
            if cmd == 'ping':
                ble_send_label('pong')
            elif cmd == 'echo':
                msg = obj.get('msg', '')
                ble_send_label('echo', msg=msg)
            elif cmd == 'hb' and 'enable' in obj:
                enable = bool(obj['enable'])
                if enable and state['hb_src'] is None:
                    state['hb_src'] = GLib.timeout_add_seconds(DEMO_HEARTBEAT_SEC, heartbeat_tick)
                    ble_send_label('hb_on')
                elif not enable and state['hb_src'] is not None:
                    GLib.source_remove(state['hb_src']); state['hb_src'] = None
                    ble_send_label('hb_off')
        elif isinstance(obj, str):
            # example text command: "ping"
            if obj.strip().lower() == 'ping':
                ble_send_label('pong')

    # Prefer transport-provided RX handler if present
    nus.rx.on_line = _rx_handler if _rx_handler else on_rx_cmd

    # Connection hooks via notifications
    def start_heartbeat():
        if DEMO_HEARTBEAT and state['hb_src'] is None:
            state['hb_src'] = GLib.timeout_add_seconds(DEMO_HEARTBEAT_SEC, heartbeat_tick)

    def stop_heartbeat():
        if state['hb_src'] is not None:
            GLib.source_remove(state['hb_src']); state['hb_src'] = None

    def heartbeat_tick():
        # runs only while subscribed
        ble_send_label('hb', demo=True)
        return True  # keep timer

    def on_client_subscribed():
        state['client_connected'] = True
        state['t_conn_start'] = time.time()
        print('\n[BLE] Client subscribed (connected)')
        GLib.idle_add(_unregister_adv)  # stop advertising once connected
        # send immediate hello + start heartbeat
        ble_send_label('hello')
        start_heartbeat()

    def on_client_unsubscribed():
        state['client_connected'] = False
        state['t_conn_start'] = None
        print('\n[BLE] Client unsubscribed (disconnected)')
        stop_heartbeat()
        exit_pairing_mode()

    nus.tx.on_start_notify = on_client_subscribed
    nus.tx.on_stop_notify  = on_client_unsubscribed

    # TX worker (chunk & notify)
    def tx_worker():
        while True:
            try:
                payload = _tx_queue.get()
                # keep under typical ~180 bytes/app-layer chunk
                max_len = 180
                for i in range(0, len(payload), max_len):
                    nus.tx.send_notification(payload[i:i+max_len])
                    time.sleep(0.005)
            except Exception as e:
                print('\n[BLE][TX] error:', e); time.sleep(0.5)
    threading.Thread(target=tx_worker, daemon=True).start()

    # Rolling status ticker
    def _status_tick():
        state['spin_idx'] = (state['spin_idx'] + 1) % len(_spinner_frames)
        sp = _spinner_frames[state['spin_idx']]
        now = time.monotonic()
        adv = state['adv_registered']; conn = state['client_connected']
        adv_dur  = _fmt_dur((now - state['t_adv_start']) if adv and state['t_adv_start'] else 0)
        conn_dur = _fmt_dur((now - state['t_conn_start']) if conn and state['t_conn_start'] else 0)
        line = f"\r[BLE]{sp} adv={'ON ' if adv else 'OFF'} {adv_dur} | conn={'YES' if conn else 'NO '} {conn_dur}    "
        try: sys.stdout.write(line); sys.stdout.flush()
        except Exception: pass
        return True
    if RUN_LIVE_BLE_DIAGNOSTICS:
        GLib.timeout_add(500, _status_tick)

    # Clean shutdown
    def _handle_sig(*_):
        print('\n[BLE] Shutting down...')
        try: _unregister_adv()
        except Exception: pass
        try: MAIN_LOOP.quit()
        except Exception: pass
    try:
        import threading as _t
        if _t.current_thread() is _t.main_thread():
            signal.signal(signal.SIGINT, _handle_sig)
            signal.signal(signal.SIGTERM, _handle_sig)
    except Exception as _e:
        print('\n[BLE] Signal setup skipped:', _e)

    global MAIN_LOOP
    MAIN_LOOP = GLib.MainLoop()
    try: MAIN_LOOP.run()
    finally:
        try: _unregister_adv()
        except Exception: pass

if __name__ == "__main__":
    main()
