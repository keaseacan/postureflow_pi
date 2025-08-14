# dependencies
import json
from typing import Optional, Any

# functions
from py_files.core import start_services, stop_services, _services_running
from py_files.bt.bt_transport import ble_send
from py_files.data_output.fn_data_outbox import ack as outbox_ack


def _handle_ble_command(msg: Any):
  """
  Supports "start" / "stop" / "status".
  Normalize bytes/str/json to a 'cmd' string.
  Replies via ble_send() if available.
  """
  cmd: Optional[str] = None
  def reply(payload):
    ble_send(payload)  # wrapper-safe (no-op if BLE not started)

  try:
    if isinstance(msg, (bytes, bytearray)):
      s = msg.decode('utf-8', errors='ignore').strip()
      try:
        obj = json.loads(s); cmd = (obj.get('cmd') or s).strip().lower()
      except Exception:
        cmd = s.lower()
    elif isinstance(msg, str):
      s = msg.strip()
      try:
        obj = json.loads(s); cmd = (obj.get('cmd') or s).strip().lower()
      except Exception:
        cmd = s.lower()
    elif isinstance(msg, dict):
      cmd = str(msg.get('cmd', '')).strip().lower()
    elif cmd == "ack":
      ids = obj.get("ids", [])
      try:
        outbox_ack([int(i) for i in ids])
        reply({"ok": True})
      except Exception as e:
        reply({"ok": False, "err": f"bad_ack:{e}"})
  except Exception:
    cmd = None

  if not cmd:
    reply({"ok": False, "err": "bad_command"}); return

  if cmd == "start":
    start_services(); reply({"ok": True, "state": "running"})
  elif cmd == "stop":
    stop_services(); reply({"ok": True, "state": "stopped"})
  elif cmd == "status":
    reply({"ok": True, "status": {"running": _services_running}})
  else:
    reply({"ok": False, "err": f"unknown_cmd:{cmd}"})
