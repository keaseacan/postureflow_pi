# dependencies
import time
from typing import Dict, Any, Optional

# functions
from py_files.data_output.fn_data_events import ChangeEventEmitter
from py_files.data_output.fn_data_spool import Spool, SpoolWorker, ClassEvent, _now_ms
from py_files.data_output.fn_data_transport import ChangeEventTransport
from py_files.model.fn_classification_model import class_map

class JsonOutbox:
  """
  Usage:
  outbox = JsonOutbox()
  outbox.reset_session(session_start_ms=acq_start_ms)  # first event will use this t
  ...
  outbox.on_classification(idx=pred_idx, ts_ms=window_start_ms)
  ...
  outbox.close()
  """
  def __init__(self, db_path: str = "posture_spool.db"):
    self.spool = Spool(db_path)
    self.emitter = ChangeEventEmitter()
    self.worker = SpoolWorker(self.spool, ChangeEventTransport(include_label=False),
                              batch_size=32)
    self.worker.start()
    # First-event timestamp override (to anchor first segment at session start)
    self._first_override_ms: Optional[int] = None
    self._first_emitted: bool = False

  def reset_session(self, session_start_ms: Optional[int] = None) -> None:
    """
    Call at the start of a recording/capture session.
    If session_start_ms is provided, the very first emitted change-event
    will use that timestamp instead of the model window start.
    """
    self.emitter.reset()
    self._first_override_ms = session_start_ms
    self._first_emitted = False

  def on_classification(self, idx: int, ts_ms: Optional[int] = None,
                        label: Optional[str] = None, score: Optional[float] = None,
                        meta: Optional[Dict[str, Any]] = None) -> None:
    """
    Feed each classification decision (ideally once per window).
    Emits only on changes; queues a minimal event in the spool.
    """
    t = ts_ms if ts_ms is not None else _now_ms()
    if self.emitter.feed(t, idx):
      if not self._first_emitted and self._first_override_ms is not None:
        t = self._first_override_ms
      self._first_emitted = True

      self.spool.enqueue(ClassEvent(
        ts_ms=t,
        cls_idx=idx,
        cls_label=label if label is not None else (class_map.get(idx, "") or ""),
        score=score if score is not None else 0.0,
        meta=meta
      ))

  # called on startup and intervaled to limit SQL table size.
  def prune(self, max_rows: int = 100000) -> None:
    self.spool.prune_keep_most_recent(max_rows)

  def close(self) -> None:
    """Stop worker thread and close DB neatly"""
    self.worker.stop()
    time.sleep(0.25)  # allow worker to exit sleep loop
    self.spool.wal_checkpoint()
    self.spool.close()

# ---------- Optional: self-test ----------
if __name__ == "__main__":
  import random
  outbox = JsonOutbox()
  try:
    t0 = _now_ms()
    outbox.reset_session(session_start_ms=t0)   # pin first segment to session start
    idx = 0
    for k in range(30):  # simulate 30 windows (1s each)
      if random.random() < 0.3:
        idx = random.choice([0, 1, 2])
      outbox.on_classification(idx=idx, ts_ms=t0 + k * 1000)
      time.sleep(0.02)
  finally:
    outbox.close()
