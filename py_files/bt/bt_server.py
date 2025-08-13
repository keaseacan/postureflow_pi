#!/usr/bin/env python3
"""
BLE Peripheral (NUS) with "press to pair, hold to disconnect" on GPIO17.

Short press:
  - Pause your pipelines (pause_cb)
  - Start advertising until the phone connects (no timeout)

Hold (>=1.2s):
  - If connected: disconnect the phone
  - Else if advertising: stop advertising
  - In both cases, resume pipelines (resume_cb) once fully out of BLE mode

Phone app disconnect:
  - Auto resume pipelines
"""

import json, queue, threading, time, signal
from typing import Callable, Optional
import dbus, dbus.exceptions, dbus.mainloop.glib, dbus.service
from gi.repository import GLib

# ---------- Config ----------
DEVICE_NAME = "PosturePi"
GPIO_BUTTON_PIN = 17      # BCM (pin 11)
BUTTON_HOLD_SEC = 1.2     # "hold" threshold

DEMO_HEARTBEAT = False
DEMO_HEARTBEAT_SEC = 5

# NUS UUIDs
NUS_SERVICE_UUID = '6E400001-B5A3-F393-E0A9-E50E24DCCA9E'
NUS_RX_UUID      = '6E400002-B5A3-F393-E0A9-E50E24DCCA9E'
NUS_TX_UUID      = '6E400003-B5A3-F393-E0A9-E50E24DCCA9E'

# --- ADV privacy toggle ---
ADVERTISE_SERVICE_UUIDS = True  # True = include UUIDs in ADV; False = omit (stealth scan-by-name)

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

def ble_send(obj: dict):
    """Queue a JSON line for TX notifications."""
    _tx_queue.put((json.dumps(obj) + "\n").encode("utf-8"))

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
    def __init__(self, bus, index, service): super().__init__(bus, index, NUS_RX_UUID, ['write','write-without-response'], service)
    @dbus.service.method('org.bluez.GattCharacteristic1', in_signature='aya{sv}')
    def WriteValue(self, value, options):
        try:
            s = bytes(bytearray(value)).decode('utf-8', errors='ignore')
            for line in s.splitlines():
                line=line.strip()
                if not line: continue
                print('[BLE][RX]', json.loads(line))
        except Exception as e: print('[BLE][RX] parse-error:', e)

class NusTxCharacteristic(Characteristic):
    def __init__(self, bus, index, service):
        super().__init__(bus, index, NUS_TX_UUID, ['notify'], service)
        self.on_start_notify=None; self.on_stop_notify=None
    @dbus.service.method('org.bluez.GattCharacteristic1')
    def StartNotify(self):
        super().StartNotify()
        try: self.on_start_notify and self.on_start_notify()
        except Exception as e: print('[BLE] on_start_notify error:', e)
    @dbus.service.method('org.bluez.GattCharacteristic1')
    def StopNotify(self):
        super().StopNotify()
        try: self.on_stop_notify and self.on_stop_notify()
        except Exception as e: print('[BLE] on_stop_notify error:', e)

class NusService(Service):
    def __init__(self, bus, index):
        super().__init__(bus, index, NUS_SERVICE_UUID, True)
        self.tx = NusTxCharacteristic(bus,0,self); self.rx = NusRxCharacteristic(bus,1,self)
        self.add_characteristic(self.tx); self.add_characteristic(self.rx)

class Advertisement(dbus.service.Object):
    PATH_BASE = '/app/advertisement'

    def __init__(self, bus, index, ad_type, service_uuids):
        self.path = self.PATH_BASE + str(index)
        self.ad_type = ad_type
        self.service_uuids = service_uuids
        super().__init__(bus, self.path)

    def get_path(self):
        return dbus.ObjectPath(self.path)

    @dbus.service.method('org.freedesktop.DBus.Properties', out_signature='a{sv}')
    def GetAll(self, interface):
        if interface != 'org.bluez.LEAdvertisement1':
            raise InvalidArgsException()

        props = {
            'Type': self.ad_type,
            'LocalName': DEVICE_NAME,
            'IncludeTxPower': dbus.Boolean(True),
            # Optional extras you can add:
            # 'Appearance': dbus.UInt16(0),   # generic
            # 'Duration': dbus.UInt16(0),     # 0 = no auto-timeout (BlueZ-specific)
            # 'Timeout': dbus.UInt16(0),
        }
        if ADVERTISE_SERVICE_UUIDS and self.service_uuids:
            props['ServiceUUIDs'] = dbus.Array(self.service_uuids, signature='s')

        return props

    @dbus.service.method('org.bluez.LEAdvertisement1')
    def Release(self):
        pass


# ---------- Adapter & device helpers ----------
def find_adapter(bus):
    om = dbus.Interface(bus.get_object(BLUEZ_SERVICE_NAME, '/'), DBUS_OM_IFACE)
    for path, ifaces in om.GetManagedObjects().items():
        if 'org.bluez.Adapter1' in ifaces:
            return path
    raise RuntimeError('Bluetooth adapter not found')

def disconnect_all_connected(bus):
    """Attempt to disconnect any connected device(s)."""
    om = dbus.Interface(bus.get_object(BLUEZ_SERVICE_NAME, '/'), DBUS_OM_IFACE)
    for path, ifaces in om.GetManagedObjects().items():
        dev = ifaces.get(DEVICE_IFACE)
        if dev and bool(dev.get('Connected', False)):
            try:
                dbus.Interface(bus.get_object(BLUEZ_SERVICE_NAME, path), DEVICE_IFACE).Disconnect()
                print(f'[BLE] Disconnect requested: {path}')
            except Exception as e:
                print('[BLE] Disconnect error:', e)

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

    service_mgr.RegisterApplication(app.get_path(), {}, reply_handler=lambda: print('[BLE] GATT app registered'),
                                    error_handler=lambda e: print('[BLE] GATT app register error:', e))

    # State
    state = {'adv_registered': False, 'client_connected': False}

    def _register_adv():
        if state['adv_registered']: return False
        def ok(): print('[BLE] Advertising'); return False
        def err(e): print('[BLE] Adv register error:', e); return False
        ad_mgr.RegisterAdvertisement(adv.get_path(), {}, reply_handler=ok, error_handler=err)
        state['adv_registered'] = True
        return False

    def _unregister_adv():
        if not state['adv_registered']: return False
        try:
            ad_mgr.UnregisterAdvertisement(adv.get_path())
            print('[BLE] Advertising stopped')
        except Exception as e:
            print('[BLE] Adv unregister error:', e)
        state['adv_registered'] = False
        return False

    def enter_pairing_mode():
        """Pause pipelines and start advertising indefinitely."""
        _pause_services_async()
        GLib.idle_add(_register_adv)

    def exit_pairing_mode():
        """Stop advertising and resume pipelines if not connected."""
        GLib.idle_add(_unregister_adv)
        if not state['client_connected']:
            _resume_services_async()

    # Button: short press to pair, hold to disconnect/cancel
    try:
        from gpiozero import Button
        button = Button(GPIO_BUTTON_PIN, pull_up=True, bounce_time=0.05, hold_time=BUTTON_HOLD_SEC)
        _held = {'v': False}

        def on_held():
            _held['v'] = True
            # If connected -> disconnect central; else cancel advertising.
            GLib.idle_add(lambda: (disconnect_all_connected(bus) if state['client_connected'] else _unregister_adv()))
            # Resume services after the above completes
            GLib.timeout_add(200, lambda: (_resume_services_async() or False))

        def on_released():
            # Short press if the hold callback didn't fire
            if _held['v']:
                _held['v'] = False
                return
            # If already advertising or connected, ignore
            if state['adv_registered'] or state['client_connected']:
                return
            enter_pairing_mode()

        button.when_held = on_held
        button.when_released = on_released
        print(f"[BLE] Button on GPIO{GPIO_BUTTON_PIN}: short=pair, hold(≥{BUTTON_HOLD_SEC}s)=disconnect")
    except Exception as e:
        print(f"[BLE] gpiozero unavailable ({e}); you can still pair via manual call to enter_pairing_mode()")
        # Optional: auto-start advertising if no button available
        GLib.idle_add(_register_adv)

    # Connection hooks via notifications
    def on_client_subscribed():
        state['client_connected'] = True
        GLib.idle_add(_unregister_adv)     # stop advertising once connected
        # services remain paused while connected

    def on_client_unsubscribed():
        state['client_connected'] = False
        exit_pairing_mode()                 # resume after disconnect

    nus.tx.on_start_notify = on_client_subscribed
    nus.tx.on_stop_notify  = on_client_unsubscribed

    # TX worker
    def tx_worker():
        while True:
            try:
                payload = _tx_queue.get()
                max_len = 180
                for i in range(0, len(payload), max_len):
                    nus.tx.send_notification(payload[i:i+max_len])
                    time.sleep(0.005)
            except Exception as e:
                print('[BLE][TX] error:', e); time.sleep(0.5)
    threading.Thread(target=tx_worker, daemon=True).start()

    # Optional heartbeat
    if DEMO_HEARTBEAT:
        def heartbeat():
            ble_send({"ts_ms": int(time.time()*1000), "class_idx": 2, "conf": 0.99, "demo": True})
            return True
        GLib.timeout_add_seconds(DEMO_HEARTBEAT_SEC, heartbeat)

    # Clean shutdown
    def _handle_sig(*_):
        print('[BLE] Shutting down...')
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
        print('[BLE] Signal setup skipped:', _e)

    global MAIN_LOOP
    MAIN_LOOP = GLib.MainLoop()
    try: MAIN_LOOP.run()
    finally:
        try: _unregister_adv()
        except Exception: pass
