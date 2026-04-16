import signal
import sys
import threading
import time
import traceback

from py_files.fn_cfg import RUN_CORE_DIAGNOSTICS
from py_files.record_process_audio.fn_record_main_realtime import start_audio_pipeline, stop_audio_pipeline
from py_files.model.fn_classification_direct import make_segment_callback

# Optional hardware/time init. Comment out if not needed for demo.
from py_files.time.time_softclock import setup_i2c
from py_files.time.time_main import init_clock

_shutdown_once = False
_shutdown_ev = threading.Event()


def pi_setup():
    print("Setup: initializing hardware...")
    setup_i2c()
    if RUN_CORE_DIAGNOSTICS:
        print("[OK] setup_i2c")
    init_clock()
    if RUN_CORE_DIAGNOSTICS:
        print("[OK] clock initialised")


def _graceful_shutdown(_sig=None, _frame=None):
    global _shutdown_once
    if _shutdown_once:
        return
    _shutdown_once = True
    _shutdown_ev.set()
    try:
        stop_audio_pipeline()
    except Exception:
        pass


def _on_signal(sig, frame):
    if RUN_CORE_DIAGNOSTICS:
        print(f"[SHUTDOWN] got signal {sig}; stopping...")
    _shutdown_ev.set()


signal.signal(signal.SIGINT, _on_signal)
signal.signal(signal.SIGTERM, _on_signal)


def main():
    exit_code = 0
    try:
        pi_setup()

        on_segment = make_segment_callback()
        start_audio_pipeline(on_segment=on_segment)

        if RUN_CORE_DIAGNOSTICS:
            print("[MAIN] realtime demo pipeline running")

        while not _shutdown_ev.is_set():
            time.sleep(0.2)

    except BaseException as e:
        exit_code = 1
        print("[FATAL] Uncaught exception:", repr(e))
        traceback.print_exc()
    finally:
        _graceful_shutdown()
        sys.exit(exit_code)


if __name__ == "__main__":
    main()