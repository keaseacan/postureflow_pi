#!/usr/bin/env python3
import time
import sys
import signal
import threading
import traceback
import json
from typing import Any, Optional

# ⬇️ use wrapper-style BLE API
from py_files.bt.bt_transport import init_ble, start_ble, stop_ble, join_ble, ble_send, BleTransport
from py_files.record_process_audio.fn_record_main import start_audio_pipeline, stop_audio_pipeline
from py_files.model.fn_classification_main import start_classification, stop_classification
from py_files.fn_time import setup_i2c, write_to_pi
from py_files.data_output.fn_data_outbox import init_outbox, reset_session, emit as emit_classification, close_outbox
from py_files.fn_cfg import RUN_CORE_DIAGNOSTICS

_shutdown_once = False
_shutdown_ev = threading.Event()

# Track service lifecycle so BLE can pause/resume them
_services_running = False
_services_lock = threading.Lock()
_feat_q = None

# ---------- Hardware + outbox ----------
def pi_setup():
  print("Setup: initializing hardware...")
  setup_i2c()
  if RUN_CORE_DIAGNOSTICS: print("[OK] setup_i2c")
  write_to_pi()
  if RUN_CORE_DIAGNOSTICS: print(f"[OK] write_to_pi")

  try:
    init_outbox(transport=BleTransport)
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
          emit_classification({"idx": int(idx), "ts_ms": int(ts_ms)})
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

# ---------- Optional: BLE command handler ----------
def _handle_ble_command(msg: Any):
  """
  Supports "start" / "stop" / "status".
  Normalize bytes/str/json to a 'cmd' string.
  Replies via ble_send() if available.
  """
  cmd: Optional[str] = None

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
  except Exception:
    cmd = None

  def reply(payload):
    ble_send(payload)  # wrapper-safe (no-op if BLE not started)

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
      join_ble(timeout=1.5)   # ⬅️ wrapper
    except Exception:
      pass
    sys.exit(exit_code)

if __name__ == "__main__":
  main()
