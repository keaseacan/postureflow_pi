import time
import sys
import signal
import threading
import traceback

from py_files.record_process_audio.fn_record_main import start_audio_pipeline, stop_audio_pipeline
from py_files.model.fn_classification_main import start_classification, stop_classification
from py_files.fn_time import setup_i2c, write_to_pi
from py_files.data_output.fn_data_outbox import init_outbox, reset_session, emit as emit_classification, close_outbox

_shutdown_once = False
_shutdown_ev = threading.Event()

def pi_setup():
  print("Setup: initializing hardware...")
  setup_i2c();                print("[OK] setup_i2c")
  ok = write_to_pi();         print(f"[OK] write_to_pi -> {ok}")

  try:
    init_outbox();   print("[OK] init_outbox")
    reset_session(); print("[OK] reset_session")
  except Exception as e:
    print("[FAIL] init_outbox:", repr(e))
    traceback.print_exc()
    raise

  feat_q = start_audio_pipeline();  print("[OK] start_audio_pipeline")

  # Start classifier with outbox emit callback if supported
  try:
    start_classification(feat_q, on_emit=emit_classification); print("[OK] start_classification (with on_emit)")
  except TypeError:
    start_classification(feat_q)
    print("[WARN] start_classification() did not accept on_emit; "
          "wire the callback in when you can so events reach the outbox.")
  return feat_q

def _graceful_shutdown(_sig=None, _frame=None):
  """Ensure threads stop and outbox is flushed (idempotent)."""
  global _shutdown_once
  if _shutdown_once:
    return
  _shutdown_once = True
  _shutdown_ev.set()
  try:
    stop_classification()
  except Exception:
    pass
  try:
    stop_audio_pipeline()
  except Exception:
    pass
  try:
    close_outbox()
  except Exception:
    pass

def _on_signal(sig, frame):
  print(f"[SHUTDOWN] got signal {sig}; stopping…")
  _shutdown_ev.set()   # just set the flag; actual cleanup happens in finally

signal.signal(signal.SIGINT, _on_signal)
signal.signal(signal.SIGTERM, _on_signal)

def main():
  exit_code = 0
  try:
    pi_setup()
    print("[MAIN] after pi_setup; entering idle loop")
    while not _shutdown_ev.is_set():
      time.sleep(0.5)
  except BaseException as e:
    exit_code = 1
    print("[FATAL] Uncaught exception:", repr(e))
    traceback.print_exc()
  finally:
    _graceful_shutdown()
    sys.exit(exit_code)

if __name__ == "__main__":
  main()
