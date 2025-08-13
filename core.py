#!/usr/bin/env python3
import time
import sys
import signal
import threading
import traceback
from typing import Any, Optional, Tuple

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

# ---------------- BLE forwarding helpers (NEW) ----------------
def _now_ms() -> int:
  return int(time.time() * 1000)

def _normalize_emit_for_ble(ev: Any) -> Optional[Tuple[str, int, Optional[float], dict]]:
  """
  Return (label, ts_ms, conf_or_None, extra) or None if unusable.
  Accepts:
    - dict: {'label': 'Standing', 'conf': 0.92, 'ts_ms': 169..., ...}
    - tuple/list: ('Standing', 0.92) or ('Standing',)
    - str: 'Standing'
  """
  try:
    if isinstance(ev, dict):
      label = str(ev.get('label') or ev.get('state') or ev.get('posture') or '').strip()
      if not label:
        return None
      ts_any = ev.get('ts_ms', ev.get('ts'))
      ts_ms = int(ts_any) if ts_any is not None else _now_ms()
      # common confidence keys
      conf_any = ev.get('conf', ev.get('score'), ev.get('prob'))
      conf = float(conf_any) if conf_any is not None else None
      extra = {k: v for k, v in ev.items() if k not in ('label', 'ts_ms', 'ts', 'conf', 'score', 'prob')}
      return (label, ts_ms, conf, extra)

    if isinstance(ev, (list, tuple)) and len(ev) >= 1:
      label = str(ev[0]).strip()
      if not label:
        return None
      conf = None
      if len(ev) >= 2:
        try:
          conf = float(ev[1])
        except Exception:
          conf = None
      return (label, _now_ms(), conf, {})

    if isinstance(ev, str):
      label = ev.strip()
      if not label:
        return None
      return (label, _now_ms(), None, {})

  except Exception:
    pass
  return None

def _forward_to_ble(ev: Any):
  """Best-effort: never raise."""
  try:
    norm = _normalize_emit_for_ble(ev)
    if not norm:
      return
    label, ts_ms, conf, extra = norm
    if conf is not None:
      extra = {**extra, "conf": conf}
    # Single line JSON; BLE layer chunks if needed
    ble.ble_send_label(label, ts_ms=ts_ms, **extra)
  except Exception as e:
    if RUN_CORE_DIAGNOSTICS:
      print("[BLE] forward error:", repr(e))

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
  """Start audio + classifier and route emits to outbox & BLE."""
  global _services_running, _feat_q
  if _services_running:
    if RUN_CORE_DIAGNOSTICS: print("[SERV] start_services: already running")
    return
  try:
    _feat_q = start_audio_pipeline()
    if RUN_CORE_DIAGNOSTICS: print("[OK] start_audio_pipeline")

    # ---- emit wrapper: write to outbox, also mirror to BLE ----
    def _on_emit(ev):
      try:
        emit_classification(ev)    # existing behavior
      except Exception as e:
        if RUN_CORE_DIAGNOSTICS: print("[OUTBOX] emit error:", repr(e))
      _forward_to_ble(ev)          # BLE mirror (best-effort)

    try:
      # Preferred signature with outbox+BLE emit
      start_classification(_feat_q, on_emit=_on_emit)
      if RUN_CORE_DIAGNOSTICS: print("[OK] start_classification (with on_emit)")
    except TypeError:
      # Fallback when classifier has no on_emit param
      start_classification(_feat_q)
      if RUN_CORE_DIAGNOSTICS:
        print("[WARN] start_classification() did not accept on_emit; "
              "BLE/outbox will not receive live events from classifier.")

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
    stop_audio_pipeline()
    if RUN_CORE_DIAGNOSTICS: print("[OK] stop_audio_pipeline")
  except Exception as e:
    if RUN_CORE_DIAGNOSTICS: print("[SERV] stop_audio_pipeline error:", repr(e))
  _services_running = False

# ---------- BLE integration ----------
def _run_ble():
  """Run BLE GLib loop in a background thread."""
  # Configure BLE behavior
  ble.DEVICE_NAME = "PosturePi"
  ble.DEMO_HEARTBEAT = False  # no periodic demo traffic

  # Pause/resume hooks so pairing doesn't fight the audio/classifier
  ble.set_service_hooks(stop_services, start_services)

  # Enter the GLib main loop (blocks this thread)
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
