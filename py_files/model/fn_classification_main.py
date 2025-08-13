import numpy as np
import time
import threading
import queue
from typing import Optional, Callable

from py_files.model.fn_classification_model import classify_idx, classify_imfs, class_map
from py_files.fn_cfg import RUN_CLASSIFICATION_DIAGNOSTICS

_cls_thread = None
_cls_stop_evt = threading.Event()

def start_classification(
  feat_q: queue.Queue,
  handler: Optional[Callable[[dict], None]] = None,
  on_emit: Optional[Callable[[int, int], None]] = None  # <- NEW: (idx, window_start_ms)
):
  """
  Start a background worker that consumes IMF dicts from feat_q.

  - If `on_emit` is provided, it's called for EVERY window with:
      on_emit(idx: int, window_start_ms: int)  # UTC epoch ms
    (Your outbox/ChangeEventEmitter will only enqueue when the label CHANGES.)

  - `handler(out)` is still supported for logging/diagnostics:
      out = {
        "idx": int, "label": str, "dur_ms": float, "env": str,
        "t": float,            # wall-clock seconds (epoch)
        "IMF": list[float],    # pass-through; remove if not needed
      }
  """

  global _cls_thread
  _cls_stop_evt.clear()

  def _default_handler(out: dict):
    print(f"[PRED] idx={out['idx']} {out['label']} "
          f"dur={out['dur_ms']:.0f} ms env={out['env']} t={out['t']:.3f}")

  if handler is None:
    handler = _default_handler

  def _worker():
    # Establish a stable mapping from monotonic -> wall clock
    t0_wall = time.time()        # epoch seconds (wall clock)
    t0_mono = time.monotonic()   # monotonic seconds

    while not _cls_stop_evt.is_set():
      try:
        res = feat_q.get(timeout=0.5)
      except queue.Empty:
        continue

      # ---- Model inference -------------------------------------------------
      imfs = np.asarray(res["IMF"], dtype=np.float32)
      idx = classify_idx(imfs)

      # Window start time in WALL-CLOCK (epoch) ms:
      # res["t_abs_start"] is expected to be monotonic seconds at window start.
      try:
        t_wall = t0_wall + (res["t_abs_start"] - t0_mono)  # seconds (float)
      except KeyError:
        t_wall = time.time()  # fallback if producer didn't include t_abs_start
      window_start_ms = int(round(t_wall * 1000.0))

      out = {
        "idx": int(idx),
        "label": class_map[idx],
        "dur_ms": float(res["Duration_ms"]),
        "env": res.get("EnvProfile", "unknown"),
        "t": t_wall,             # keep as seconds like before
        "IMF": res["IMF"],       # drop if you don't need it downstream
      }

      # ---- Optional diagnostics -------------------------------------------
      if RUN_CLASSIFICATION_DIAGNOSTICS:
        try:
          _, _, score = classify_imfs(imfs)
          print(f"[PRED] idx={out['idx']} {out['label']} score={score:.3f} "
                f"dur={out['dur_ms']:.0f} ms env={out['env']} t={out['t']:.3f}")
        except Exception:
          pass

      # ---- Emit to outbox (change-event gate happens inside your outbox) ---
      if on_emit is not None:
        try:
          on_emit(int(idx), window_start_ms)
        except Exception as e:
          print(f"[CLS] on_emit error: {e}")

      # ---- Legacy/console handler ------------------------------------------
      try:
        handler(out)
      except Exception as e:
        print(f"[CLS] handler error: {e}")

  _cls_thread = threading.Thread(target=_worker, daemon=True, name="classification")
  _cls_thread.start()
  return _cls_thread

def stop_classification(timeout: float = 1.0):
  _cls_stop_evt.set()
  if _cls_thread is not None:
    try:
      _cls_thread.join(timeout=timeout)
    except RuntimeError:
      pass
