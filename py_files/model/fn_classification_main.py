import numpy as np
import time
import threading
import queue

from py_files.model.fn_classification_model import classify_idx, classify_imfs, class_map
from py_files.fn_cfg import RUN_CLASSIFICATION_DIAGNOSTICS

_cls_thread = None
_cls_stop_evt = threading.Event()

def start_classification(feat_q: queue.Queue, handler=None):
    """
    Start a background worker that consumes IMF dicts from feat_q and emits
    classified results via handler(out). Returns the Thread.
    out = {
      "idx": int, "label": str, "dur_ms": float, "env": str,
      "t": float,            # wall-clock seconds
      "IMF": list[float],    # included for convenience; remove if you don't want it
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
      t0_wall = time.time()
      t0_mono = time.monotonic()
      while not _cls_stop_evt.is_set():
        try:
          res = feat_q.get(timeout=0.5)
        except queue.Empty:
          continue

        imfs = np.asarray(res["IMF"], dtype=np.float32)
        idx = classify_idx(imfs)

        out = {
          "idx": int(idx),
          "label": class_map[idx],
          "dur_ms": float(res["Duration_ms"]),
          "env": res.get("EnvProfile", "unknown"),
          "t": t0_wall + (res["t_abs_start"] - t0_mono),
          "IMF": res["IMF"],   
          }

        # for debugging
        if RUN_CLASSIFICATION_DIAGNOSTICS:
          _, _, score = classify_imfs(imfs)
          print(f"[PRED] idx={out['idx']} {out['label']} score={score:.3f} "
          f"dur={out['dur_ms']:.0f} ms env={out['env']} t={out['t']:.3f}")

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
