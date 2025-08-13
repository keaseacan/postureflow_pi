#!/usr/bin/env python3
import time
import sys
import signal
import threading
import traceback

import py_files.bt.bt_server as ble
from py_files.record_process_audio.fn_record_main import start_audio_pipeline, stop_audio_pipeline
from py_files.model.fn_classification_main import start_classification, stop_classification
from py_files.fn_time import setup_i2c, write_to_pi
from py_files.data_output.fn_data_outbox import init_outbox, reset_session, emit as emit_classification, close_outbox

from py_files.fn_cfg import RUN_CORE_DIAGNOSTICS

_shutdown_once = False
_shutdown_ev = threading.Event()

# Track service lifecycle so BLE can pause/resume them
_services_running = False
_feat_q = None
_ble_thread = None

# ---------- Hardware + outbox ----------
def pi_setup():
  print("Setup: initializing hardware...")
  setup_i2c()
  if RUN_CORE_DIAGNOSTICS: print("[OK] setup_i2c")
  write_to_pi()
  if RUN_CORE_DIAGNOSTICS: print(f"[OK] write_to_pi")

  try:
    init_outbox()
    if RUN_CORE_DIAGNOSTICS: print("[OK] init_outbox")
    reset_session()
    if RUN_CORE_DIAGNOSTICS: print("[OK] reset_session")

  except Exception as e:
    if RUN_CORE_DIAGNOSTICS: print("[FAIL] init_outbox:", repr(e))
    traceback.print_exc()
    raise

# ---------- Pipelines start/stop (pause/resume hooks) ----------
def start_services():
  """Start audio + classifier and route emits to outbox."""
  global _services_running, _feat_q
  if _services_running:
    if RUN_CORE_DIAGNOSTICS: print("[SERV] start_services: already running")
    return
  try:
    _feat_q = start_audio_pipeline(); 
    if RUN_CORE_DIAGNOSTICS: print("[OK] start_audio_pipeline")
    try:
      # Preferred signature with outbox emit
      start_classification(_feat_q, on_emit=emit_classification)
      if RUN_CORE_DIAGNOSTICS: print("[OK] start_classification (with on_emit)")
    except TypeError:
      start_classification(_feat_q)
      if RUN_CORE_DIAGNOSTICS: print("[WARN] start_classification() did not accept on_emit; events may not reach outbox.")
    _services_running = True
  except Exception as e:
    if RUN_CORE_DIAGNOSTICS: print("[SERV] start_services error:", repr(e))
    traceback.print_exc()

def stop_services():
  """Stop classifier then audio (idempotent)."""
  global _services_running
  if not _services_running:
    if RUN_CORE_DIAGNOSTICS: print("[SERV] stop_services: already stopped")
    return
  try:
    stop_classification()
    if RUN_CORE_DIAGNOSTICS: print("[OK] stop_classification")
  except Exception as e:
    if RUN_CORE_DIAGNOSTICS: print("[SERV] stop_classification error:", repr(e))
  try:
    stop_audio_pipeline();
    if RUN_CORE_DIAGNOSTICS: print("[OK] stop_audio_pipeline")
  except Exception as e:
    if RUN_CORE_DIAGNOSTICS: print("[SERV] stop_audio_pipeline error:", repr(e))
  _services_running = False

# ---------- BLE integration ----------
def _run_ble():
  """Run BLE GLib loop in a background thread."""
  # Configure BLE behavior
  ble.DEVICE_NAME = "PosturePi"
  ble.DEMO_HEARTBEAT = False  # no demo traffic
  # Button/flow already defined inside nus_gatt_server:
  # - Short press GPIO17 -> enter pairing (advertise until connected)
  # - Hold GPIO17       -> disconnect (or cancel advertising)
  # - Phone disconnect  -> resume services

  # Give BLE the hooks so it can pause/resume our services.
  ble.set_service_hooks(stop_services, start_services)

  # Block this thread inside GLib main loop
  ble.main()

def start_ble_thread():
  global _ble_thread
  if _ble_thread and _ble_thread.is_alive():
    return
  _ble_thread = threading.Thread(target=_run_ble, name="ble-thread", daemon=True)
  _ble_thread.start()
  if RUN_CORE_DIAGNOSTICS: print("[BLE] thread started")

def _stop_ble():
  """Ask BLE loop to quit gracefully (best-effort)."""
  try:
    # Our BLE module exposes MAIN_LOOP; quitting it will end the thread.
    if getattr(ble, "MAIN_LOOP", None) is not None:
      ble.MAIN_LOOP.quit()
  except Exception:
    pass

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
    _stop_ble()
  except Exception:
    pass
  try:
    close_outbox()
  except Exception:
    pass

def _on_signal(sig, frame):
  if RUN_CORE_DIAGNOSTICS: print(f"[SHUTDOWN] got signal {sig}; stopping…")
  _shutdown_ev.set()   # actual cleanup happens in finally

signal.signal(signal.SIGINT, _on_signal)
signal.signal(signal.SIGTERM, _on_signal)

# ---------- Main ----------
def main():
  exit_code = 0
  try:
    pi_setup()
    # Start your services normally (idle mode)
    start_services()
    # Spin up BLE (button controls pairing/disconnect)
    start_ble_thread()

    if RUN_CORE_DIAGNOSTICS: print("[MAIN] services running; BLE ready. Press GPIO17 to pair.")
    while not _shutdown_ev.is_set():
      time.sleep(0.5)
  except BaseException as e:
    exit_code = 1
    if RUN_CORE_DIAGNOSTICS: print("[FATAL] Uncaught exception:", repr(e))
    traceback.print_exc()
  finally:
    _graceful_shutdown()
    # Give BLE thread a moment to exit
    if _ble_thread and _ble_thread.is_alive():
      _ble_thread.join(timeout=1.5)
    sys.exit(exit_code)

if __name__ == "__main__":
  main()
