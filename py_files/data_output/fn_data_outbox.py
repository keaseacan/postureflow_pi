# dependencies
from typing import Optional, Sequence

# functions
from py_files.data_output.fn_data_main import JsonOutbox # uses the spool/worker/transport we set up
from py_files.time.time_main import _now_ms

_outbox: Optional[JsonOutbox] = None  # module-level singleton

def init_outbox(db_path: str = "posture_spool.db", transport=None, max_rows: int = 100_000) -> JsonOutbox:
  """Create (once) and return the durable JSON outbox.
  - transport: pass a BleNotifyTransport(...) later; None uses stdout transport.
  - max_rows: safety cap; old queued rows beyond this are pruned.
  """
  global _outbox
  if _outbox is None:
      _outbox = JsonOutbox(db_path=db_path, transport=transport)
      _outbox.prune(max_rows=max_rows)  # safety valve at startup
  return _outbox

def reset_session(session_start_ms: Optional[int] = None) -> None:
  """Anchor the FIRST emitted segment to session_start_ms (defaults to now)."""
  if _outbox is None:
      raise RuntimeError("init_outbox() must be called before reset_session()")
  if session_start_ms is None:
      session_start_ms = _now_ms()
  _outbox.reset_session(session_start_ms=session_start_ms)

def emit(idx: int, window_start_ms: int) -> None:
  """Call once per analysis window; only enqueues when label CHANGES."""
  if _outbox is None:
      return  # not initialized yet; drop quietly
  _outbox.on_classification(idx=idx, ts_ms=window_start_ms)

def close_outbox() -> None:
  """Stop worker, checkpoint WAL, close DB."""
  global _outbox
  if _outbox is not None:
    _outbox.close()
    _outbox = None

def ack(ids: Sequence[int]) -> None:
  """Delete delivered rows after phone confirms receipt."""
  if _outbox is None:
    return
  _outbox.spool.ack(list(ids))