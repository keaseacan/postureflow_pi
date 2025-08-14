# dependencies
import time
import sys
import signal
import threading
import traceback
import json
from typing import Optional, Any

# functions
from py_files.bt.bt_transport import init_ble, start_ble, stop_ble, join_ble, ble_send
from py_files.data_output.fn_data_outbox import init_outbox, reset_session, emit as emit_classification, close_outbox, ack as outbox_ack
from py_files.data_output.fn_data_transport import ChangeEventTransport
from py_files.record_process_audio.fn_record_main import start_audio_pipeline, stop_audio_pipeline
from py_files.model.fn_classification_main import start_classification, stop_classification
from py_files.time.time_softclock import setup_i2c
from py_files.time.time_main import init_clock, apply_phone_time_sync, _now_ms

# diagnostic constants
from py_files.fn_cfg import RUN_CORE_DIAGNOSTICS


_shutdown_once = False
_shutdown_ev = threading.Event()

# Track service lifecycle so BLE can pause/resume them
global _services_running = False
_services_lock = threading.Lock()
_feat_q = None

# ---------- Hardware + outbox ----------
def pi_setup():
  print("Setup: initializing hardware...")
  setup_i2c()
  if RUN_CORE_DIAGNOSTICS: print("[OK] setup_i2c")
  init_clock()
  if RUN_CORE_DIAGNOSTICS: print("[OK] clock initialised")

  try:
    init_outbox(transport=ChangeEventTransport)
    if RUN_CORE_DIAGNOSTICS: print("[OK] init_outbox")
    reset_session()
    if RUN_CORE_DIAGNOSTICS: print("[OK] reset_session")
  except Exception as e:
    if RUN_CORE_DIAGNOSTICS: print("[FAIL] init_outbox:", repr(e))
    traceback.print_exc()
    raise

# ---------- Pipelines start/stop (pause/resume hooks) ----------
def start_services():
  """Start audio + classifier; emit to outbox ONLY."""
  global _services_running, _feat_q
  with _services_lock:
    if _services_running:
      if RUN_CORE_DIAGNOSTICS: print("[SERV] start_services: already running")
      return
    try:
      _feat_q = start_audio_pipeline()
      if RUN_CORE_DIAGNOSTICS: print("[OK] start_audio_pipeline")

      def _on_emit(idx: int, ts_ms: int):
        try:
          emit_classification(int(idx), int(ts_ms))
        except Exception as e:
          if RUN_CORE_DIAGNOSTICS: print("[OUTBOX] emit error:", repr(e))

      try:
        start_classification(_feat_q, on_emit=_on_emit)
        if RUN_CORE_DIAGNOSTICS: print("[OK] start_classification (with on_emit)")
      except TypeError:
        start_classification(_feat_q)
        if RUN_CORE_DIAGNOSTICS:
          print("[WARN] start_classification() did not accept on_emit; "
                "outbox will not receive live events from classifier.")

      _services_running = True
    except Exception as e:
      if RUN_CORE_DIAGNOSTICS: print("[SERV] start_services error:", repr(e))
      traceback.print_exc()

def stop_services():
  """Stop classifier then audio (idempotent)."""
  global _services_running, _feat_q
  with _services_lock:
    if not _services_running:
      if RUN_CORE_DIAGNOSTICS: print("[SERV] stop_services: already stopped")
      return
    try:
      stop_classification()
      if RUN_CORE_DIAGNOSTICS: print("[OK] stop_classification")
    except Exception as e:
      if RUN_CORE_DIAGNOSTICS: print("[SERV] stop_classification error:", repr(e))
    try:
      stop_audio_pipeline()
      if RUN_CORE_DIAGNOSTICS: print("[OK] stop_audio_pipeline")
    except Exception as e:
      if RUN_CORE_DIAGNOSTICS: print("[SERV] stop_audio_pipeline error:", repr(e))
    _feat_q = None
    _services_running = False

# ---------- Shutdown handling ----------
def _graceful_shutdown(_sig=None, _frame=None):
  """Ensure threads stop and outbox is flushed (idempotent)."""
  global _shutdown_once
  if _shutdown_once:
    return
  _shutdown_once = True
  _shutdown_ev.set()
  try:
    stop_services()
  except Exception:
    pass
  try:
    stop_ble()
  except Exception:
    pass
  try:
    close_outbox()
  except Exception:
    pass

def _on_signal(sig, frame):
  if RUN_CORE_DIAGNOSTICS: print(f"[SHUTDOWN] got signal {sig}; stopping…")
  _shutdown_ev.set()   # actual cleanup in finally

signal.signal(signal.SIGINT, _on_signal)
signal.signal(signal.SIGTERM, _on_signal)

# ---- BLE reply handling ----
def _handle_ble_command(msg: Any):
  """
  Supports: start | stop | status | ack | time_sync
  msg may be bytes/str/json/dict. Replies via ble_send(...) as JSON (dict).
  """
  def reply(obj: dict):
      try:
          ble_send(obj)  # <-- pass dict; ble_send will json.dumps for you
      except Exception:
          pass

  # ---- Normalize input to a dict 'obj' ----
  obj: Optional[dict] = None
  if isinstance(msg, (bytes, bytearray)):
      s = msg.decode("utf-8", "ignore").strip()
      try:
          obj = json.loads(s)
      except Exception:
          obj = {"cmd": s}
  elif isinstance(msg, str):
      s = msg.strip()
      try:
          obj = json.loads(s)
      except Exception:
          obj = {"cmd": s}
  elif isinstance(msg, dict):
      obj = msg.copy()
  else:
      reply({"ok": False, "err": "bad_type"})
      return

  cmd = str(obj.get("cmd", "")).strip().lower()
  if not cmd:
      reply({"ok": False, "err": "bad_command"})
      return

  # ---- Commands ----
  if cmd == "ack":
      ids = obj.get("ids", [])
      if not isinstance(ids, list):
          reply({"ok": False, "err": "ack.ids_not_list"})
          return
      if outbox_ack is None:
          reply({"ok": False, "err": "ack_handler_unavailable"})
          return
      try:
          outbox_ack([int(i) for i in ids])
          reply({"ok": True, "cmd": "ack", "n": len(ids)})
      except Exception as e:
          reply({"ok": False, "err": f"bad_ack:{e}"})
      return

  if cmd == "time_sync":
      # Phone should send: {"cmd":"time_sync","epoch_ms": <utc_ms>}
      epoch_ms = obj.get("epoch_ms")
      if epoch_ms is None:
          reply({"ok": False, "err": "missing_epoch_ms"})
          return
      try:
          epoch_ms = int(epoch_ms)
      except Exception:
          reply({"ok": False, "err": "bad_epoch_ms"})
          return

      drift_ms = abs(_now_ms() - epoch_ms)
      if drift_ms > 1000:
          sys_ok, rtc_ok = apply_phone_time_sync(epoch_ms)
      else:
          # Already in sync (≤1s drift); avoid thrashing system/RTC
          sys_ok = rtc_ok = True
      reply({"ok": True, "cmd": "time_sync", "drift_ms": drift_ms, "sys_ok": sys_ok, "rtc_ok": rtc_ok})
      return

  if cmd == "start":
      start_services()
      reply({"ok": True, "state": "running"})
      return

  if cmd == "stop":
      stop_services()
      reply({"ok": True, "state": "stopped"})
      return

  if cmd == "status":
      reply({"ok": True, "status": {"running": _services_running}})
      return

  # Unknown
  reply({"ok": False, "err": f"unknown_cmd:{cmd}"})


# ---------- Main ----------
def main():
  exit_code = 0
  try:
    pi_setup()
    start_services()

    # Init + start BLE transport (wrapper style)
    init_ble(
      on_command=_handle_ble_command,
      service_hooks=(stop_services, start_services),
      device_name="PosturePi",
      demo_heartbeat=False,
      diag=RUN_CORE_DIAGNOSTICS,
    )
    start_ble()

    if RUN_CORE_DIAGNOSTICS: print("[MAIN] services running; BLE transport ready. Press GPIO17 to pair.")
    while not _shutdown_ev.is_set():
      time.sleep(0.5)

  except BaseException as e:
    exit_code = 1
    if RUN_CORE_DIAGNOSTICS: print("[FATAL] Uncaught exception:", repr(e))
    traceback.print_exc()
  finally:
    _graceful_shutdown()
    try:
      join_ble(timeout=1.5)
    except Exception:
      pass
    sys.exit(exit_code)

from core import main

if __name__ == "__main__":
  main()