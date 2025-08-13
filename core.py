import time
import sys
import signal

from py_files.record_process_audio.fn_record_main import start_audio_pipeline, stop_audio_pipeline
from py_files.model.fn_classification_main import start_classification, stop_classification
from py_files.fn_time import setup_i2c, write_to_pi

# outbox helpers (your wrapper module)
from py_files.data_output.fn_data_outbox import (
    init_outbox, reset_session, emit as emit_classification, close_outbox
)

def pi_setup():
  print("Setup: initializing hardware...")
  setup_i2c();                print("[OK] setup_i2c")
  ok = write_to_pi();         print(f"[OK] write_to_pi -> {ok}")

  try:
    init_outbox()
    print("[OK] init_outbox")
    reset_session()
    print("[OK] reset_session")
  except Exception as e:
    print("[FAIL] init_outbox:", repr(e))
    import traceback; traceback.print_exc()
    raise

  feat_q = start_audio_pipeline();  print("[OK] start_audio_pipeline")

  # 3) Start classifier thread and give it the emit callback
  try:
    # preferred signature (updated classifier): start_classification(feat_q, on_emit=callable)
    start_classification(feat_q, on_emit=emit_classification)
    print("[OK] start_classification (with on_emit)")
  except TypeError:
    # fallback if your current start_classification doesn't accept on_emit yet
    start_classification(feat_q)
    print("[WARN] start_classification() did not accept on_emit; "
          "wire the callback in when you can so events reach the outbox.")

  return feat_q

def _graceful_shutdown(_sig=None, _frame=None):
  """Ensure threads stop and outbox is flushed."""
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
  finally:
    sys.exit(0)

# Ctrl+C / SIGTERM
signal.signal(signal.SIGINT, _graceful_shutdown)
signal.signal(signal.SIGTERM, _graceful_shutdown)

import traceback

if __name__ == "__main__":
    exit_code = 0
    try:
        _ = pi_setup()
        print("[MAIN] after pi_setup; entering idle loop")
        while True:
            time.sleep(0.5)
    except BaseException as e:
        exit_code = 1
        print("[FATAL] Uncaught exception:", repr(e))
        traceback.print_exc()
    finally:
        # don't hide the exception; pass the exit code through
        try:
            _graceful_shutdown()
        finally:
            sys.exit(exit_code)