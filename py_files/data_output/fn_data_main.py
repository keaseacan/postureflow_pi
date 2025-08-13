# dependencies
import time
from typing import Dict, Any, Optional

# functions
from py_files.data_output.fn_data_events import ChangeEventEmitter
from py_files.data_output.fn_data_spool import Spool, SpoolWorker, ClassEvent, _now_ms
from py_files.data_output.fn_data_transport import ChangeEventTransport
from py_files.model.fn_classification_model import class_map

class JsonOutbox:
  def __init__(self, db_path: str = "posture_spool.db", transport=None, batch_size: int = 32):
    self.spool = Spool(db_path)
    self.emitter = ChangeEventEmitter()

    # default to stdout JSON if no transport provided
    if transport is None:
      transport = ChangeEventTransport(include_label=False)

    self.worker = SpoolWorker(self.spool, transport, batch_size=batch_size)
    self.worker.start()

    self._first_override_ms: Optional[int] = None
    self._first_emitted: bool = False

  def reset_session(self, session_start_ms: Optional[int] = None) -> None:
    self.emitter.reset()
    self._first_override_ms = session_start_ms
    self._first_emitted = False

  def on_classification(self, idx: int, ts_ms: Optional[int] = None,
                      label: Optional[str] = None, score: Optional[float] = None,
                      meta: Optional[Dict[str, Any]] = None) -> None:
    t = ts_ms if ts_ms is not None else now_ms()
    if self.emitter.feed(t, idx):
      if not self._first_emitted and self._first_override_ms is not None:
        t = self._first_override_ms
      self._first_emitted = True
      self.spool.enqueue(ClassEvent(
        ts_ms=t, cls_idx=idx, cls_label=label or "", score=score or 0.0, meta=meta
      ))

  def prune(self, max_rows: int = 100_000) -> None:
    self.spool.prune_keep_most_recent(max_rows)

  def close(self) -> None:
    self.worker.stop()
    time.sleep(0.25)
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
