# dependencies
from typing import Optional

# remembers last change to so that another row can added to the SQL table.
class ChangeEventEmitter:
  """Emit an event only when label index changes."""
  def __init__(self):
    self._last_idx: Optional[int] = None

  def reset(self) -> None:
    self._last_idx = None

  def feed(self, ts_ms: int, idx: int) -> bool:
    """
    Returns True if a change-event should be emitted at ts_ms for idx.
    """
    if self._last_idx is None or idx != self._last_idx:
      self._last_idx = idx
      return True
    return False
